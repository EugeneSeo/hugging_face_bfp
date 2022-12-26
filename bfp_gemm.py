import numpy as np
import sys
import torch
import time
from ctypes import *

# if "/home/yskim/projects/sparse-bfp" not in sys.path:
    # sys.path.append("/home/yskim/projects/sparse-bfp")

from util.bfp.bfp_config import BfpConfig
from util.reprod_util import set_reproducibility
from util.custom_transpose import custom_transpose_2d
from util.bfp.cuda_bfp_wrapper import CudaBfpWrapper
# from util.bfp.cuda.cuda_memory_helper import CudaFloatStorage

import os
BFP_HOME = os.environ["BFP_HOME"]

cuda_lib = CDLL(f"{BFP_HOME}/util/bfp/cuda/lib_not_sorted_gemm_cuda.so")
cuda_thres_lib = CDLL(f"{BFP_HOME}/util/bfp/cuda/lib_bfp_thresholding_gemm_cuda.so")
# cuda_optim_lib = CDLL(f"{BFP_HOME}/util/bfp/cuda/lib_not_sorted_gemm_optimized_cuda.so")
cuda_multi_exp_lib = CDLL(f"{BFP_HOME}/util/bfp/cuda/lib_multi_exp_gemm_cuda.so")
# cuda_zero_indexed_lib = CDLL(f"{BFP_HOME}/util/bfp/cuda/lib_cuda_zero_indexed_gemm.so")
cuda_multi_exp_shift_lib = CDLL(f"{BFP_HOME}/util/bfp/cuda/lib_multi_exp_shift_gemm_cuda.so")

class BfpGemm:
    def __init__(
        self, lhs_shape: torch.Size, 
        rhs_shape: torch.Size, 
        use_stochastic_rounding: bool, 
        use_multi_exp: bool,
        apply_thresholding: bool,
        threshold: int, 
        config = None
        ):
        self.device = "cuda"

        self.use_multi_exp = use_multi_exp
        self.apply_thresholding = apply_thresholding

        if self.use_multi_exp and self.apply_thresholding:
            raise ValueError(f"use_multi_exp and apply_thresholding cannot be turn on together")

        self.lhs_wrapper = CudaBfpWrapper(
            fp_tensor_shape=lhs_shape, 
            use_multi_exp=self.use_multi_exp,
            apply_thresholding=apply_thresholding,
            threshold=threshold)
        self.rhs_wrapper = CudaBfpWrapper(
            fp_tensor_shape=rhs_shape, 
            use_multi_exp=self.use_multi_exp,
            apply_thresholding=apply_thresholding,
            threshold=threshold)

        shape_cnt = len(lhs_shape)
        if shape_cnt > 2:
            out_size = list(lhs_shape[:-1])
            out_size.append(rhs_shape[0])
        else:
            out_size = (lhs_shape[0], rhs_shape[0])

        self.output_shape = out_size

        self.output = torch.zeros(
            size=out_size,
            dtype=torch.float,
            device=self.device)

        self.use_stochastic_rounding = use_stochastic_rounding

        if self.lhs_wrapper.bfp_M.shape[-1] != self.rhs_wrapper.bfp_M.shape[-1]:
            raise ValueError(f"cannot matched dim: {self.lhs_wrapper.bfp_M.shape} x {self.rhs_wrapper.bfp_M.shape}")

        self.M = self.lhs_wrapper.bfp_M.shape[0]
        self.K = self.lhs_wrapper.bfp_M.shape[1]
        self.N = self.rhs_wrapper.bfp_M.shape[0]
        # self.bfp_M_bit = BfpConfig.bfp_M_Bit
        # self.group_size = BfpConfig.group_size
        Config = config if config is not None else BfpConfig
        if config is not None:
            BfpConfig.use_bfp = config.use_bfp
            BfpConfig.use_flex_bfp = config.use_flex_bfp
            BfpConfig.use_multi_exp = config.use_multi_exp
            BfpConfig.group_size = config.group_size
            BfpConfig.bfp_M_Bit = config.bfp_M_Bit
            BfpConfig.thread_num = config.thread_num
            BfpConfig.is_fast = config.is_fast
            BfpConfig.f_st = config.f_st
            BfpConfig.a_st = config.a_st 
            BfpConfig.w_st = config.w_st
            BfpConfig.threshold = config.threshold
            BfpConfig.f_thres = config.f_thres
            BfpConfig.a_thres = config.a_thres
            BfpConfig.w_thres = config.w_thres
            BfpConfig.use_shift = config.use_shift
        self.bfp_M_bit = Config.bfp_M_Bit
        self.group_size = Config.group_size

        self.use_shift = BfpConfig.use_shift

    def run(self, lhs: torch.Tensor, rhs: torch.Tensor):
        self.lhs_wrapper.run_convert_fp_to_bfp(src_tensor=lhs, is_stochastic_rounding=self.use_stochastic_rounding)
        self.rhs_wrapper.run_convert_fp_to_bfp(src_tensor=rhs, is_stochastic_rounding=self.use_stochastic_rounding)

        if self.use_multi_exp and not self.use_shift:
            cuda_multi_exp_lib.bfp_gemm_2d(
                c_void_p(self.lhs_wrapper.bfp_S.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_E.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_M.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_E_index_ptr.data_ptr()),


                c_void_p(self.rhs_wrapper.bfp_S.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_E.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_M.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_E_index_ptr.data_ptr()),

                c_void_p(self.output.data_ptr()),

                c_int32(self.M),
                c_int32(self.K),
                c_int32(self.N),

                c_int32(self.group_size),
                c_int32(self.bfp_M_bit),
                c_int32(self.bfp_M_bit)
            )
        elif self.use_multi_exp and self.use_shift:
            cuda_multi_exp_shift_lib.bfp_gemm_2d(
                c_void_p(self.lhs_wrapper.bfp_S.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_E.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_M.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_E_index_ptr.data_ptr()),


                c_void_p(self.rhs_wrapper.bfp_S.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_E.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_M.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_E_index_ptr.data_ptr()),

                c_void_p(self.output.data_ptr()),

                c_int32(self.M),
                c_int32(self.K),
                c_int32(self.N),

                c_int32(self.group_size),
                c_int32(self.bfp_M_bit),
                c_int32(self.bfp_M_bit)
            )
        elif self.apply_thresholding:
            cuda_thres_lib.bfp_gemm_2d(
            # cuda_optim_lib.bfp_gemm_2d(
                c_void_p(self.lhs_wrapper.bfp_S.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_E.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_M.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_M_bit_flag.data_ptr()),

                c_void_p(self.rhs_wrapper.bfp_S.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_E.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_M.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_M_bit_flag.data_ptr()),

                c_void_p(self.output.data_ptr()),

                c_int32(self.M),
                c_int32(self.K),
                c_int32(self.N),
                c_int32(self.group_size),

                c_int32(self.lhs_wrapper.bfp_M_bit),
                c_int32(self.lhs_wrapper.bfp_M_bit_to_thres),
                c_int32(self.rhs_wrapper.bfp_M_bit),
                c_int32(self.rhs_wrapper.bfp_M_bit_to_thres)
            )
        else:
            cuda_lib.bfp_gemm_2d(
            # cuda_optim_lib.bfp_gemm_2d(
                c_void_p(self.lhs_wrapper.bfp_S.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_E.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_M.data_ptr()),

                c_void_p(self.rhs_wrapper.bfp_S.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_E.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_M.data_ptr()),

                c_void_p(self.output.data_ptr()),

                c_int32(self.M),
                c_int32(self.K),
                c_int32(self.N),

                c_int32(self.group_size),
                c_int32(self.lhs_wrapper.bfp_M_bit),
                c_int32(self.rhs_wrapper.bfp_M_bit)
            )


        return self.output

def load_inout(path_root, phase, layer_index):
    path = f"{path_root}/{phase}-{layer_index}-fp"

    lhs = np.load(f"{path}-lhs.npy", allow_pickle=False)
    rhs = np.load(f"{path}-rhs.npy", allow_pickle=False)

    output = np.matmul(lhs, rhs)
    # print(f"lhs: {lhs.shape} X rhs: {rhs.shape} = output: {output.shape}")
    
    # lhs = lhs[n_index,0:k_cnt]
    # rhs = rhs[0:k_cnt,m_index]
    # output = np.dot(lhs, rhs)
    # print(f"lhs: {lhs.shape} X rhs: {rhs.shape} = output: {output.shape}")

    return lhs, rhs, output

if __name__ == "__main__":
    set_reproducibility(1234)
    BfpConfig.group_size = 16
    BfpConfig.bfp_M_Bit = 4
    use_multi_exp = True
    apply_thresholding = False
    threshold = 1
    use_SR = True
    BfpConfig.use_shift = True
    # BfpConfig.chunk_size_to_sort = 256

    log_path = f"{BFP_HOME}/resnet50-inout-log/epoch20-lr0.001"
    lhs, rhs, output = load_inout(log_path, "fwd", 0)
    # print(lhs.shape)
    # print(rhs.shape)
    # lhs = lhs[:,0:16]
    # rhs = rhs[0:16,:]

    M = lhs.shape[0] # 1024 # 1
    K = lhs.shape[1] # 2024 * 4 # 2
    N = rhs.shape[1] # 1024 # 1


    # M, K, N = 512, 512, 512

    print(f"{'bfp only' if not use_multi_exp else 'multi exp'}")
    print(f"dims: [{M}, {K}] x [{K}, {N}]")

    lhs = torch.tensor(lhs, dtype=torch.float32)
    # lhs = torch.randn(size=(M, K), dtype=torch.float32)
    lhs = lhs.to('cuda')

    rhs = torch.tensor(rhs, dtype=torch.float32)
    # rhs = torch.randn(size=(K, N), dtype=torch.float32)
    rhs = rhs.to('cuda')
    rhs_t = torch.empty((N, K), dtype=torch.float).to('cuda')
    custom_transpose_2d(dst=rhs_t, src=rhs)
    
    bfp_gemm = BfpGemm(
        lhs.shape, 
        rhs_t.shape, 
        use_stochastic_rounding=use_SR, 
        use_multi_exp=use_multi_exp,
        apply_thresholding=apply_thresholding,
        threshold=threshold)

    # print(f"lhs: {bfp_gemm.lhs_wrapper.bfp_M_bit}-bit, rhs: {bfp_gemm.rhs_wrapper.bfp_M_bit}-bit")

    # outputs_cuda = bfp_gemm.run(lhs, rhs_t)

    iterations = 1

    start = time.time_ns()
    for i in range(iterations):
        outputs_cuda = bfp_gemm.run(lhs, rhs_t)
    end = time.time_ns()

    print(f"{(end - start) / iterations / 1000_000:.5f}ms")

    # print(f"output: {outputs_cuda.cpu()[0,0]:.10f}")

    # print(f"lhs S: {bfp_gemm.lhs_wrapper.bfp_S.cpu().numpy()}")
    # if not use_multi_exp:
    #     print(f"lhs E: {bfp_gemm.lhs_wrapper.bfp_E.cpu().numpy()}")
    # print(f"lhs M: {bfp_gemm.lhs_wrapper.bfp_M.cpu().numpy()} ({(bfp_gemm.lhs_wrapper.bfp_M.count_nonzero() / (M * K)) * 100.:.2f}%)")

    # print(f"rhs S: {bfp_gemm.rhs_wrapper.bfp_S.cpu().numpy()}")
    # if not use_multi_exp:
    #     print(f"rhs E: {bfp_gemm.rhs_wrapper.bfp_E.cpu().numpy()}")
    # print(f"rhs M: {bfp_gemm.rhs_wrapper.bfp_M.cpu().numpy()}")

    # if use_multi_exp:
    #     print(f"lhs Es: {bfp_gemm.lhs_wrapper.bfp_E[0][0:4].cpu().numpy()}, rhs Es: {bfp_gemm.rhs_wrapper.bfp_E[0][0:4].cpu().numpy()}")
    #     print(f"lhs E index: {bfp_gemm.lhs_wrapper.bfp_E_index_ptr.cpu().numpy()}")
    #     print(f"rhs E index: {bfp_gemm.rhs_wrapper.bfp_E_index_ptr.cpu().numpy()}")
    # else:
    #     print(f"lhs ori M: {[f'{e:032b}' for e in bfp_gemm.lhs_wrapper.bfp_32_bit_M.cpu().numpy()[0]]}")
    #     print(f"rhs ori M: {[f'{e:032b}' for e in bfp_gemm.rhs_wrapper.bfp_32_bit_M.cpu().numpy()[0]]}")

    # print(f"output: {outputs_cuda.cpu().numpy()[0,0]:.15f}")

    # outputs_cuda_added = outputs_cuda + output_add

    fp_outputs = torch.matmul(lhs, rhs)
    # print(f"answer: {fp_outputs.cpu().numpy()[0,0]:.15f}")

    # diff = torch.abs(fp_outputs.flatten() - outputs.flatten())
    diff_cuda = torch.abs(fp_outputs.flatten() - outputs_cuda.flatten())
    errors = torch.abs((outputs_cuda - fp_outputs) / (fp_outputs + 1e-15)) * 100.

    # diff_cuda_added = torch.abs(fp_outputs.flatten() - outputs_cuda_added.flatten())
    # errors_added = torch.abs((outputs_cuda_added - fp_outputs) / (fp_outputs + 1e-15)) * 100.

    # print(f"{errors.cpu().item():.2f}%")

    # print(mask.nonzero().cpu().numpy())

    # nans = torch.isnan(outputs_cuda)
    # print(nans.shape)
    # print(nans[1,1])

    # mask = errors > 100.
    # mask_cnt = torch.count_nonzero(mask).cpu().item()
    print(f"avg error mean  : {torch.mean(errors.flatten()).item():.2f}")
    # print(f"avg error mean* : {torch.mean(errors_added.flatten()).item():.2f}")
    # print(f"mask cnt: {mask_cnt} ({(mask_cnt / (N * M)) * 100.:.2f}%)")

    print(f"[diffs]\nmean: {torch.mean(diff_cuda):.20f}\nmax:  {torch.max(diff_cuda):.20f}\nmin:  {torch.min(diff_cuda):.20f}")
    # print(f"[diffs added]\nmean: {torch.mean(diff_cuda_added):.15f}\nmax:  {torch.max(diff_cuda_added):.15f}\nmin:  {torch.min(diff_cuda_added):.15f}")
