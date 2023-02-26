from typing import OrderedDict
import torch
import torch.nn as nn

import numpy as np

import os
BFP_HOME = os.environ["BFP_HOME"]

from util.bfp.bfp_config import BfpConfig, PrecisionFlag
from util.custom_transpose import custom_transpose_2d, custom_transpose_4d
# from util.bfp.bfp_gemm import BfpGemm
from util.bfp.bfp_gemm_for_cmm import BfpGemmForCustomMatMul
from util.bfp.fast_bfp_gemm import FastBfpGemm
from util.bfp.cuda_bfp_wrapper import CudaBfpWrapper

from util.reprod_util import set_reproducibility
from typing import List


class MatMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        lhs_mat,
        rhs_mat,
        precision_flag,
        bfp_gemms,
        intermediate_memory,
        id,
        global_id,
        shapes,
        wrappers,
        is_deberta
        ):
        if precision_flag == PrecisionFlag.FP:
            prec_name = "fp"
        elif precision_flag == PrecisionFlag.BFP:
            prec_name = "bfp"
        else:
            raise ValueError(f"not supported precision flag: {precision_flag}")

        ctx.save_for_backward(lhs_mat, rhs_mat)
        ctx.precision_flag = precision_flag
        ctx.bfp_gemms = bfp_gemms
        ctx.intermediate_memory = intermediate_memory
        ctx.id = id
        ctx.global_id = global_id
        ctx.prec_name = prec_name
        ctx.shapes = shapes
        ctx.wrappers = wrappers
        ctx.is_deberta = is_deberta

        if precision_flag == PrecisionFlag.FP:
            outputs = torch.matmul(lhs_mat, rhs_mat)

        elif precision_flag == PrecisionFlag.BFP:
            rhs_mat_t = intermediate_memory["rhs-mat-t"]

            custom_transpose_4d(dst=rhs_mat_t, src=rhs_mat)

            if bfp_gemms["fwd"] is None:
                if BfpConfig.is_fast:
                    raise ValueError(f"unsupported BFP option: is_fast")
                else:
                    bfp_gemms["fwd"] = BfpGemmForCustomMatMul(
                        lhs_mat.shape,
                        rhs_mat_t.shape,
                        shapes["output"],
                        use_stochastic_rounding=BfpConfig.f_st,
                        use_multi_exp=BfpConfig.use_multi_exp,
                        apply_thresholding=BfpConfig.f_thres,
                        threshold=BfpConfig.threshold
                    )
            bfp_gemm = bfp_gemms["fwd"]
            outputs = bfp_gemm.run(lhs_mat, rhs_mat_t)

        else:
            raise ValueError(f"not supported precision flag: {precision_flag}")

        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        lhs_mat, rhs_mat = ctx.saved_tensors
        precision_flag = ctx.precision_flag
        bfp_gemms = ctx.bfp_gemms
        intermediate_memory = ctx.intermediate_memory
        id = ctx.id
        global_id = ctx.global_id
        prec_name = ctx.prec_name
        shapes = ctx.shapes
        wrappers = ctx.wrappers
        is_deberta = ctx.is_deberta
        grad_lhs_mat = grad_rhs_mat = None

        # print(f"grad_out:\n{grad_output}")

        if ctx.needs_input_grad[0]:
            if precision_flag == PrecisionFlag.FP:
                # grad_lhs_mat = torch.matmul(grad_output, rhs_mat.t())
                grad_lhs_mat = torch.matmul(grad_output, rhs_mat.transpose(-1, -2))

            elif precision_flag == PrecisionFlag.BFP:
                if bfp_gemms["grad-lhs-mat"] is None:
                    if BfpConfig.is_fast:
                        raise ValueError(f"unsupported BFP option: is_fast")
                    else:
                        bfp_gemms["grad-lhs-mat"] = BfpGemmForCustomMatMul(
                            grad_output.shape,
                            rhs_mat.shape,
                            shapes["lhs"],
                            use_stochastic_rounding=BfpConfig.a_st,
                            use_multi_exp=BfpConfig.use_multi_exp,
                            apply_thresholding=BfpConfig.a_thres,
                            threshold=BfpConfig.threshold)

                bfp_gemm = bfp_gemms["grad-lhs-mat"]
                grad_lhs_mat = bfp_gemm.run(grad_output.contiguous(), rhs_mat.contiguous())
            else:
                raise ValueError(f"not supported precision flag: {precision_flag}")
        if ctx.needs_input_grad[1]:
            if precision_flag == PrecisionFlag.FP:
                # grad_rhs_mat = torch.matmul(lhs_mat.t(), grad_output)
                grad_rhs_mat = torch.matmul(lhs_mat.transpose(-1, -2), grad_output)

            elif precision_flag == PrecisionFlag.BFP:
                lhs_mat_t = intermediate_memory["lhs-mat-t"]
                custom_transpose_4d(dst=lhs_mat_t, src=lhs_mat.contiguous())

                grad_output_t = intermediate_memory["grad-output-t"]
                custom_transpose_4d(dst=grad_output_t, src=grad_output.contiguous())

                if bfp_gemms["grad-rhs-mat"] is None:
                    if BfpConfig.is_fast:
                        raise ValueError(f"unsupported BFP option: is_fast")
                    else:
                        bfp_gemms["grad-rhs-mat"] = BfpGemmForCustomMatMul(
                            lhs_mat_t.shape,
                            grad_output_t.shape,
                            shapes["rhs"] if not is_deberta else shapes["rhs-intermediate"],
                            use_stochastic_rounding=BfpConfig.a_st,
                            use_multi_exp=BfpConfig.use_multi_exp,
                            apply_thresholding=BfpConfig.a_thres,
                            threshold=BfpConfig.threshold)
                bfp_gemm = bfp_gemms["grad-rhs-mat"]
                grad_rhs_mat = bfp_gemm.run(lhs_mat_t, grad_output_t)

                if is_deberta:
                    grad_rhs_mat = torch.unsqueeze(torch.sum(grad_rhs_mat, dim=0), dim=0)

            else:
                raise ValueError(f"not supported precision flag: {precision_flag}")

        return grad_lhs_mat, grad_rhs_mat, None, None, None, None, None, None, None, None


class CustomMatMul(nn.Module):
    current_id = 0

    def __init__(
        self,
        lhs_shape: torch.Size,
        rhs_shape: torch.Size,
        precision_flag: PrecisionFlag, 
        global_id: int,
        use_multi_exp: bool,
        apply_thresholding: bool,
        threshold: int,
        config=None):
        super(CustomMatMul, self).__init__()

        if len(lhs_shape) != 4 and len(rhs_shape) != 4:
            raise ValueError("both lhs/rhs must have 4-dim")

        self.id = CustomMatMul.current_id
        CustomMatMul.current_id += 1

        self.global_id = global_id
        self.precision_flag = precision_flag

        self.bfp_gemms = {
            "fwd": None,
            "grad-lhs-mat": None,
            "grad-rhs-mat": None
        }
        self.intermediate_memory = {
            "rhs-mat-t": None,
            "grad-output-t": None,
            "lhs-mat-t": None
        }

        self.device = "cuda"

        # info
        self.use_multi_exp = use_multi_exp

        self.ori_lhs_shape = lhs_shape
        self.ori_lhs_t_shape = list(self.ori_lhs_shape)
        self.ori_lhs_t_shape[-1] = self.ori_lhs_shape[-2]
        self.ori_lhs_t_shape[-2] = self.ori_lhs_shape[-1]
        self.ori_lhs_t_shape = torch.Size(self.ori_lhs_t_shape)

        self.ori_rhs_shape = rhs_shape
        self.ori_rhs_t_shape = list(self.ori_rhs_shape)
        self.ori_rhs_t_shape[-1] = self.ori_rhs_shape[-2]
        self.ori_rhs_t_shape[-2] = self.ori_rhs_shape[-1]
        self.ori_rhs_t_shape = torch.Size(self.ori_rhs_t_shape)


        self.output_shape = self.__generate_output_shape(lhs_shape, rhs_shape)
        self.output_t_shape = list(self.output_shape)
        self.output_t_shape[-1] = self.output_shape[-2]
        self.output_t_shape[-2] = self.output_shape[-1]
        self.output_t_shape = torch.Size(self.output_t_shape)

        self.lhs_t = torch.empty(
            size=self.ori_lhs_t_shape,
            dtype=torch.float,
            device=self.device)

        self.rhs_t = torch.empty(
            size=self.ori_rhs_t_shape,
            dtype=torch.float,
            device=self.device)

        self.output = torch.empty(
            size=self.output_shape,
            dtype=torch.float,
            device=self.device)

        self.output_t = torch.empty(
            size=self.output_t_shape,
            dtype=torch.float,
            device=self.device)

        self.lhs_wrapper = CudaBfpWrapper(
            fp_tensor_shape=self.ori_lhs_shape, 
            use_multi_exp=use_multi_exp,
            apply_thresholding=apply_thresholding,
            threshold=threshold)
        self.lhs_t_wrapper = CudaBfpWrapper(
            fp_tensor_shape=self.ori_lhs_t_shape, 
            use_multi_exp=use_multi_exp,
            apply_thresholding=apply_thresholding,
            threshold=threshold)

        self.rhs_wrapper = CudaBfpWrapper(
            fp_tensor_shape=self.ori_rhs_shape, 
            use_multi_exp=use_multi_exp,
            apply_thresholding=apply_thresholding,
            threshold=threshold)
        self.rhs_t_wrapper = CudaBfpWrapper(
            fp_tensor_shape=self.ori_rhs_t_shape, 
            use_multi_exp=use_multi_exp,
            apply_thresholding=apply_thresholding,
            threshold=threshold)

        self.intermediate_memory = {
            "rhs-mat-t": self.rhs_t,
            "grad-output-t": self.output_t,
            "lhs-mat-t": self.lhs_t,
            "output": self.output,
        }

        rhs_intermediate_shape = list(self.ori_rhs_shape)
        rhs_intermediate_shape[0] = self.ori_lhs_shape[0]
        rhs_intermediate_shape = torch.Size(rhs_intermediate_shape)

        self.shapes = {
            "lhs": self.ori_lhs_shape,
            "lhs-t": self.ori_lhs_t_shape,
            "rhs": self.ori_rhs_shape,
            "rhs-intermediate": rhs_intermediate_shape,
            "rhs-t": self.ori_rhs_t_shape,
            "output": self.output_shape,
            "output-t": self.output_t_shape

        }

        self.wrappers = {
            "lhs": self.lhs_wrapper,
            "lhs-t": self.lhs_t_wrapper,
            "rhs": self.rhs_wrapper,
            "rhs-t": self.rhs_t_wrapper
        }

        self.is_deberta = False
        if self.shapes["rhs"][0] != self.shapes["rhs-intermediate"][0]:
            self.is_deberta = True

        # self.config = config

    def __generate_output_shape(
        self,
        lhs_shape: torch.Size,
        rhs_shape: torch.Size) -> torch.Size:
        lhs = torch.zeros(size=lhs_shape)
        rhs = torch.zeros(size=rhs_shape)
        output = torch.matmul(lhs, rhs)

        return torch.Size(list(output.shape))
        

    def forward(self, lhs_mat, rhs_mat):
        if self.precision_flag == PrecisionFlag.FP or self.precision_flag == PrecisionFlag.BFP:
            return MatMulFunction.apply(lhs_mat, rhs_mat, self.precision_flag, self.bfp_gemms, self.intermediate_memory, self.id, self.global_id, self.shapes, self.wrappers, self.is_deberta)
        else:
            raise ValueError(f"not supported precision flag: {self.precision_flag}")




if __name__ == "__main__":
    set_reproducibility(128)
    BfpConfig.group_size = 16
    BfpConfig.bfp_M_Bit = 8
    BfpConfig.use_shift = False
    use_multi_exp = False
    apply_thresholding = False
    threshold = 1
    use_SR = True

    dev = "cuda"

    l = torch.randint(low=-5, high=5, size=(2, 2, 2, 3), requires_grad=True, dtype=torch.float, device=dev)
    r = torch.randint(low=-5, high=5, size=(1, 2, 3, 2), requires_grad=True, dtype=torch.float, device=dev)

    param = torch.randint(
        low=-5,
        high=5,
        size=(1, 16), 
        dtype=torch.float
    )

    target = torch.randint(low=-5, high=5, size=(1, 1)).to(dev)

    print(f"L:\n{l}\nR:\n{r}")

    def run_bfp(lhs: torch.Tensor, rhs: torch.Tensor) -> List[torch.Tensor]:
        custom_matmul = CustomMatMul(
            lhs_shape=l.shape,
            rhs_shape=r.shape,
            precision_flag=PrecisionFlag.BFP,
            global_id=0,
            use_multi_exp=False,
            apply_thresholding=False,
            threshold=1).to(dev)

        linear = nn.Linear(in_features=16, out_features=1, bias=False)
        linear.weight = nn.Parameter(param.clone().detach())
        linear = linear.to(dev)

        res = custom_matmul(l, r)
        out = linear(res.flatten())

        loss = torch.sum(out - target.clone().detach())
        loss.backward()

        print(f"l.grad:\n{l.grad}")
        print(f"r.grad:\n{r.grad}")

        return [res, l.grad, r.grad]

    def run_fp(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        linear = nn.Linear(in_features=16, out_features=1, bias=False)
        linear.weight = nn.Parameter(param.clone().detach())
        linear = linear.to(dev)

        res = torch.matmul(l, r)
        out = linear(res.flatten())

        loss = torch.sum(out - target.clone().detach())
        loss.backward()

        print(f"l.grad:\n{l.grad}")
        print(f"r.grad:\n{r.grad}")

        return [res, l.grad, r.grad]


    bfp_res = run_bfp(l.clone().detach(), r.clone().detach())
    fp_res = run_fp(l.clone().detach(), r.clone().detach())

    for i in range(3):
        print(f"abs diff sum: {torch.sum(torch.abs(bfp_res[i].flatten() - fp_res[i].flatten())).cpu().item()}")


    # print(f"res\n{res}")

    # linear = nn.Linear(in_features=8, out_features=1, bias=False)
    # # print(linear.weight.shape)
    # linear.weight = nn.Parameter(torch.randint(low=-5, high=5, size=(1, 8), dtype=torch.float))
    # linear = linear.to(dev)
    # print(f"linear weight\n{linear.weight}")

    # out = linear(res.flatten())
    # print(f"out\n{out}")

    # target = torch.randint(low=-5, high=5, size=(1, 1)).to(dev)
    # print(f"target\n{target}")

    # loss = torch.sum(out - target)
    # loss.backward()

    # print(f"L grad:\n{l.retain_grad()}\nR grad:\n{r.retain_grad()}")
    # print(f"L grad:\n{l.grad}\nR grad:\n{r.grad}")