from typing import OrderedDict
import torch
import torch.nn as nn

import numpy as np

import os
BFP_HOME = os.environ["BFP_HOME"]

from util.bfp.bfp_config import BfpConfig, PrecisionFlag
from util.custom_transpose import custom_transpose_2d
from util.bfp.bfp_gemm import BfpGemm
from util.bfp.fast_bfp_gemm import FastBfpGemm

from util.reprod_util import set_reproducibility


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        inputs, 
        weights, 
        bias, 
        precision_flag, 
        bfp_gemms, 
        intermediate_memory, 
        id,
        global_id
        ):
        if precision_flag == PrecisionFlag.FP:
            prec_name = "fp"
        elif precision_flag == PrecisionFlag.BFP:
            prec_name = "bfp"
        else:
            raise ValueError(f"not supported precision flag: {precision_flag}")

        ctx.save_for_backward(inputs, weights, bias)
        ctx.precision_flag = precision_flag
        ctx.bfp_gemms = bfp_gemms
        ctx.intermediate_memory = intermediate_memory
        ctx.id = id
        ctx.global_id = global_id
        ctx.prec_name = prec_name

        if precision_flag == PrecisionFlag.FP:
            outputs = torch.matmul(inputs, weights.t())

        elif precision_flag == PrecisionFlag.BFP:
            if bfp_gemms["fwd"] is None:
                if BfpConfig.is_fast:
                    bfp_gemms["fwd"] = FastBfpGemm(
                        inputs.shape,
                        weights.shape,
                        use_stochastic_rounding=BfpConfig.f_st,
                        use_flex_bfp=BfpConfig.use_flex_bfp,
                        layer_index=global_id,
                        name=f"linear-{id}-fwd")
                else:
                    bfp_gemms["fwd"] = BfpGemm(
                        inputs.shape,
                        weights.shape,
                        use_stochastic_rounding=BfpConfig.f_st,
                        use_multi_exp=BfpConfig.use_multi_exp,
                        apply_thresholding=BfpConfig.f_thres,
                        threshold=BfpConfig.threshold)

            bfp_gemm = bfp_gemms["fwd"]
            outputs = bfp_gemm.run(inputs, weights)

            # if BfpConfig.should_log:
            #     with open(f"{BfpConfig.log_path}/fwd-{global_id}-A.npy", "wb") as f:
            #         np.save(f, inputs.cpu().clone().detach().numpy(), allow_pickle=False)

            #     with open(f"{BfpConfig.log_path}/fwd-{global_id}-W.npy", "wb") as f:
            #         np.save(f, weights.cpu().clone().detach().numpy(), allow_pickle=False)

            if BfpConfig.should_log:
                with open(f"{BfpConfig.log_path}/fwd-{global_id}-A.npy", "wb") as f:
                    np.save(f, inputs.cpu().clone().detach().numpy(), allow_pickle=False)

                with open(f"{BfpConfig.log_path}/fwd-{global_id}-W.npy", "wb") as f:
                    np.save(f, weights.cpu().clone().detach().numpy(), allow_pickle=False)


        else:
            raise ValueError(f"not supported precision flag: {precision_flag}")

        if bias is not None:
            outputs += bias.unsqueeze(0).expand_as(outputs)

        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        precision_flag = ctx.precision_flag
        bfp_gemms = ctx.bfp_gemms
        intermediate_memory = ctx.intermediate_memory
        id = ctx.id
        global_id = ctx.global_id
        prec_name = ctx.prec_name
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            if precision_flag == PrecisionFlag.FP:
                grad_input = torch.matmul(grad_output, weight)

            elif precision_flag == PrecisionFlag.BFP:
                if intermediate_memory["weight-t"] is None:
                    intermediate_memory["weight-t"] = torch.empty(
                        size=(weight.shape[1], weight.shape[0]),
                        dtype=torch.float,
                        device="cuda"
                    )
                weight_t = intermediate_memory["weight-t"]
                custom_transpose_2d(dst=weight_t, src=weight.contiguous())

                if bfp_gemms["grad-a"] is None:
                    if BfpConfig.is_fast:
                        bfp_gemms["grad-a"] = FastBfpGemm(
                            grad_output.shape,
                            weight_t.shape,
                            use_stochastic_rounding=BfpConfig.a_st,
                            use_flex_bfp=BfpConfig.use_flex_bfp,
                            layer_index=global_id,
                            name=f"linear-{id}-grad-a")
                    else:
                        bfp_gemms["grad-a"] = BfpGemm(
                            grad_output.shape,
                            weight_t.shape,
                            use_stochastic_rounding=BfpConfig.a_st,
                            use_multi_exp=BfpConfig.use_multi_exp,
                            apply_thresholding=BfpConfig.a_thres,
                            threshold=BfpConfig.threshold)

                bfp_gemm = bfp_gemms["grad-a"]
                grad_input = bfp_gemm.run(grad_output.contiguous(), weight_t)

                # if BfpConfig.should_log:
                #     with open(f"{BfpConfig.log_path}/bwd-{global_id}-dA.npy", "wb") as f:
                #         np.save(f, grad_input.cpu().clone().detach().numpy(), allow_pickle=False)
                if BfpConfig.should_log:
                    with open(f"{BfpConfig.log_path}/bwd-{global_id}-dA.npy", "wb") as f:
                        np.save(f, grad_input.cpu().clone().detach().numpy(), allow_pickle=False)

            else:
                raise ValueError(f"not supported precision flag: {precision_flag}")

        if ctx.needs_input_grad[1]:
            if len(grad_output.shape) > 2:
                grad_output_size = (
                    int(torch.prod(torch.tensor(grad_output.shape[:-1])).item()),
                    int(grad_output.shape[-1])
                )
            else:
                grad_output_size = (
                    int(grad_output.shape[0]),
                    int(grad_output.shape[1])
                )
            
            if len(input.shape) > 2:
                input_size = (
                    int(torch.prod(torch.tensor(input.shape[:-1])).item()),
                    int(input.shape[-1])
                )
            else:
                input_size = (
                    int(input.shape[0]),
                    int(input.shape[1])
                )

            if precision_flag == PrecisionFlag.FP:
                grad_weight = grad_output.reshape(grad_output_size).t().mm(input.reshape(input_size))

            elif precision_flag == PrecisionFlag.BFP:
                if intermediate_memory["grad-output-t"] is None:
                    intermediate_memory["grad-output-t"] = torch.empty(
                        size=(grad_output_size[1], grad_output_size[0]),
                        dtype=torch.float,
                        device="cuda"
                    )
                grad_output_t = intermediate_memory["grad-output-t"]
                custom_transpose_2d(dst=grad_output_t, src=grad_output.contiguous())
                
                if intermediate_memory["input-t"] is None:
                    intermediate_memory["input-t"] = torch.empty(
                        size=(input_size[1], input_size[0]),
                        dtype=torch.float,
                        device="cuda"
                    )
                input_t = intermediate_memory["input-t"]
                custom_transpose_2d(dst=input_t, src=input.contiguous())

                if bfp_gemms["grad-w"] is None:
                    if BfpConfig.is_fast:
                        bfp_gemms["grad-w"] = FastBfpGemm(
                            grad_output_t.shape,
                            input_t.shape,
                            use_stochastic_rounding=BfpConfig.w_st,
                            use_flex_bfp=BfpConfig.use_flex_bfp,
                            layer_index=global_id,
                            name=f"linear-{id}-grad-w")
                    else:
                        bfp_gemms["grad-w"] = BfpGemm(
                            grad_output_t.shape,
                            input_t.shape,
                            use_stochastic_rounding=BfpConfig.w_st,
                            use_multi_exp=BfpConfig.use_multi_exp,
                            apply_thresholding=BfpConfig.w_thres,
                            threshold=BfpConfig.threshold)

                bfp_gemm = bfp_gemms["grad-w"]
                grad_weight = bfp_gemm.run(grad_output_t, input_t)

                # if BfpConfig.should_log:
                #     with open(f"{BfpConfig.log_path}/bwd-{global_id}-dW.npy", "wb") as f:
                #         np.save(f, grad_weight.cpu().clone().detach().numpy(), allow_pickle=False)
                if BfpConfig.should_log:
                    with open(f"{BfpConfig.log_path}/bwd-{global_id}-dW.npy", "wb") as f:
                        np.save(f, grad_weight.cpu().clone().detach().numpy(), allow_pickle=False)

            else:
                raise ValueError(f"not supported precision flag: {precision_flag}")

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
            if BfpConfig.should_log:
                with open(f"{BfpConfig.log_path}/bwd-{global_id}-dB.npy", "wb") as f:
                    np.save(f, grad_bias.cpu().clone().detach().numpy(), allow_pickle=False)
            

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class CustomLinear(nn.Module):
    current_id = 0

    def __init__(self, input_features, output_features, bias, precision_flag: PrecisionFlag, global_id: int, config=None):
        super(CustomLinear, self).__init__()
        self.id = CustomLinear.current_id
        CustomLinear.current_id += 1

        self.global_id = global_id

        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.empty(output_features, input_features))

        self.bfp_gemms = {
            "fwd": None,
            "grad-a": None,
            "grad-w": None
        }

        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.precision_flag = precision_flag

        self.intermediate_memory = {
            "weight-t": None,
            "grad-output-t": None,
            "input-t": None
        }

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if self.precision_flag == PrecisionFlag.FP or self.precision_flag == PrecisionFlag.BFP:
            return LinearFunction.apply(input, self.weight, self.bias, self.precision_flag, self.bfp_gemms, self.intermediate_memory, self.id, self.global_id)
        else:
            raise ValueError(f"not supported precision flag: {self.precision_flag}")


if __name__ == "__main__":
    # input_features = 123
    # hidden_features = 34
    # output_features = 5
    # BfpConfig.bfp_M_Bit = 24

    # set_reproducibility(12345)

    # max_val = 5
    # min_val = -5

    # linear = CustomLinear(input_features=input_features,
    #                       output_features=hidden_features, 
    #                       bias=True, 
    #                       precision_flag=PrecisionFlag.BFP,
    #                       global_id=0)

    # linear_fp = nn.Linear(in_features=input_features,
    #                       out_features=hidden_features,
    #                       bias=True)

    # linear_fp.weight = nn.Parameter(torch.randint(low=min_val, high=max_val, size=linear_fp.weight.shape, dtype=torch.float))
    # linear_fp.bias = nn.Parameter(torch.randint(low=min_val, high=max_val, size=linear_fp.bias.shape, dtype=torch.float))
    
    # linear.weight = nn.Parameter(linear_fp.weight.data.clone().detach())
    # linear.bias = nn.Parameter(linear_fp.bias.data.clone().detach())

    # linear = linear.to("cuda")
    # linear_fp = linear_fp.to("cuda")

    # # inputs = torch.randn((8, 4, input_features)).to("cuda")
    # inputs = torch.randint(low=min_val, high=max_val, size=(8, 4, input_features), dtype=torch.float).to("cuda")

    # outputs = linear(inputs)
    # outputs_fp = linear_fp(inputs)

    # print(f"BFP: {outputs.shape}")
    # print(f"FP : {outputs_fp.shape}")

    # print(f"mean diff: {torch.mean(torch.abs(outputs - outputs_fp))}")
    

    # exit()

    batch_size = 32
    input_features = 123 # 16
    hidden_features = 243 # 8
    output_features =  12 # 4
    BfpConfig.group_size = 16
    BfpConfig.use_flex_bfp = False
    BfpConfig.bfp_M_Bit = 8

    BfpConfig.f_thres = True
    BfpConfig.a_thres = True
    BfpConfig.w_thres = True
    BfpConfig.threshold = 4


    set_reproducibility(12345)

    max_val = 5
    min_val = -5

    l_ref = nn.Sequential(OrderedDict([
        ("linear0", nn.Linear(in_features=input_features, out_features=hidden_features, bias=True)),
        ("linear1", nn.Linear(in_features=hidden_features, out_features=output_features, bias=True)),
    ]))

    # l_ref.linear0.weight = nn.Parameter(torch.randint(low=min_val, high=max_val, size=l_ref.linear0.weight.shape, dtype=torch.float))
    # l_ref.linear0.bias = nn.Parameter(torch.randint(low=min_val, high=max_val, size=l_ref.linear0.bias.shape, dtype=torch.float))

    # l_ref.linear1.weight = nn.Parameter(torch.randint(low=min_val, high=max_val, size=l_ref.linear1.weight.shape, dtype=torch.float))
    # l_ref.linear1.bias = nn.Parameter(torch.randint(low=min_val, high=max_val, size=l_ref.linear1.bias.shape, dtype=torch.float))

    l_fp = nn.Sequential(OrderedDict([
        ("linear0", CustomLinear(input_features=input_features,
                                 output_features=hidden_features, 
                                 bias=True, 
                                 precision_flag=PrecisionFlag.FP,
                                 global_id=0)),
        ("linear1", CustomLinear(input_features=hidden_features,
                                 output_features=output_features, 
                                 bias=True, 
                                 precision_flag=PrecisionFlag.FP,
                                 global_id=1)),
    ]))

    l_bfp = nn.Sequential(OrderedDict([
        ("linear0", CustomLinear(input_features=input_features,
                                 output_features=hidden_features, 
                                 bias=True, 
                                 precision_flag=PrecisionFlag.BFP,
                                 global_id=0)),
        ("linear1", CustomLinear(input_features=hidden_features,
                                 output_features=output_features, 
                                 bias=True, 
                                 precision_flag=PrecisionFlag.BFP,
                                 global_id=1)),
    ]))


    l_fp.linear0.weight = nn.Parameter(l_ref.linear0.weight.data.clone().detach())
    l_fp.linear0.bias = nn.Parameter(l_ref.linear0.bias.data.clone().detach())
    l_fp.linear1.weight = nn.Parameter(l_ref.linear1.weight.data.clone().detach())
    l_fp.linear1.bias = nn.Parameter(l_ref.linear1.bias.data.clone().detach())

    l_bfp.linear0.weight = nn.Parameter(l_ref.linear0.weight.data.clone().detach())
    l_bfp.linear0.bias = nn.Parameter(l_ref.linear0.bias.data.clone().detach())
    l_bfp.linear1.weight = nn.Parameter(l_ref.linear1.weight.data.clone().detach())
    l_bfp.linear1.bias = nn.Parameter(l_ref.linear1.bias.data.clone().detach())

    models = [l_ref.to("cuda"), l_bfp.to("cuda")]
    # criterions = [torch.nn.CrossEntropyLoss() for _ in range(len(models))]
    optimizers = []
    for model in models:
        optimizers.append(torch.optim.SGD(model.parameters(), lr=1.0))

    inputs = torch.randn(size=(batch_size, input_features)).to("cuda")
    # inputs = torch.randint(low=min_val, high=max_val, size=(8, 4, input_features), dtype=torch.float).to("cuda")
    # inputs = torch.randint(low=min_val, high=max_val, size=(batch_size, 18, input_features), dtype=torch.float).to("cuda")

    # targets = torch.randn(size=(batch_size, output_features), dtype=torch.float).to("cuda")
    # targets = torch.randint(low=min_val, high=max_val, size=(8, 4, output_features), dtype=torch.float).to("cuda")
    targets = torch.randint(low=min_val, high=max_val, size=(batch_size, output_features), dtype=torch.float).to("cuda")
    # targets = torch.randint(low=min_val, high=max_val, size=(batch_size, 18, output_features), dtype=torch.float).to("cuda")
    
    # prev_weight = l_fp.weight.data.clone().detach()

    for i in range(len(models)):
        # print("[FWD]")
        outputs = models[i](inputs)

        # print(outputs.shape)
        # print(targets.shape)

        # loss = criterions[i](outputs, targets)
        loss = torch.sum(torch.abs(outputs - targets))

        # print("[BWD]")
        optimizer = optimizers[i]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # w0_diff_fp = torch.abs(l_ref.linear0.weight.flatten() - l_fp.linear0.weight.flatten())
    # w1_diff_fp = torch.abs(l_ref.linear1.weight.flatten() - l_fp.linear1.weight.flatten())

    w0_diff_bfp = torch.abs(l_ref.linear0.weight.flatten() - l_bfp.linear0.weight.flatten())
    w1_diff_bfp = torch.abs(l_ref.linear1.weight.flatten() - l_bfp.linear1.weight.flatten())

    # print(f"[diff fp]")
    # print(f"w0 : {torch.mean(w0_diff_fp):.10f}")
    # print(f"w1 : {torch.mean(w1_diff_fp):.10f}")

    print(f"[diff bfp]")
    print(f"w0   : {torch.mean(w0_diff_bfp):.10f}")
    print(f"w1   : {torch.mean(w1_diff_bfp):.10f}")
    print()

    # print(f"[std]")
    # print(f"fp   : {torch.std(diff_fp):.10f}")
    # print(f"bfp  : {torch.std(diff_bfp):.10f}")
    # print()

    # print(f"[max]")
    # print(f"fp   : {torch.max(diff_fp):.10f}")
    # print(f"bfp  : {torch.max(diff_bfp):.10f}")
    # print()

    # print(f"[min]")
    # print(f"fp   : {torch.min(diff_fp):.10f}")
    # print(f"bfp  : {torch.min(diff_bfp):.10f}")
    # print()




