import numpy as np
import sys
import torch
import time
from ctypes import *

from util.bfp.bfp_config import BfpConfig
from util.reprod_util import set_reproducibility
from util.custom_transpose import custom_transpose_2d
from util.bfp.cuda_bfp_wrapper import CudaBfpWrapper

import os
BFP_HOME = os.environ["BFP_HOME"]

cuda_lib = CDLL(f"{BFP_HOME}/util/bfp/cuda/lib_not_sorted_gemm_cuda.so")
cuda_multi_exp_lib = CDLL(f"{BFP_HOME}/util/bfp/cuda/lib_multi_exp_gemm_cuda.so")
cuda_zero_bfp_to_fp_lib = CDLL(f"{BFP_HOME}/util/bfp/cuda/lib_cuda_zero_bfp_to_fp_gemm.so")
memory_helper_lib = CDLL(f"{BFP_HOME}/util/bfp/cuda/lib_cuda_memory_helper.so")
cuda_multi_exp_shift_lib = CDLL(f"{BFP_HOME}/util/bfp/cuda/lib_multi_exp_shift_gemm_cuda.so")


class CustomMatMul:
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

        lhs_shape_cnt = len(lhs_shape)
        if lhs_shape_cnt > 2:
            lhs_batch = int(torch.prod(torch.tensor(lhs_shape[:-2])).item())
            lhs_rows = int(lhs_shape[-2])
            lhs_cols = int(lhs_shape[-1])
        else:
            lhs_batch = 1
            lhs_rows = int(lhs_shape[0])
            lhs_cols = int(lhs_shape[1])

        rhs_shape_cnt = len(rhs_shape)
        if rhs_shape_cnt > 2:
            rhs_batch = int(torch.prod(torch.tensor(rhs_shape[:-2])).item())
            rhs_rows = int(rhs_shape[-2])
            rhs_cols = int(rhs_shape[-1])
        else:
            rhs_batch = 1
            rhs_rows = int(rhs_shape[0])
            rhs_cols = int(rhs_shape[1])

        self.M = lhs_rows
        self.K = lhs_cols
        self.N = rhs_cols

        if not (lhs_batch == 1 or rhs_batch == 1 or lhs_batch == rhs_batch):
            raise ValueError(f"CustomMatMul invalid lhs/rhs shapes: ({lhs_batch}, {lhs_rows}, {lhs_cols}) x ({rhs_batch}, {rhs_rows}, {rhs_cols})")

        self.lhs_B = lhs_batch
        self.rhs_B = rhs_batch
        
        if lhs_batch == 1:
            out_size = (rhs_batch, lhs_rows, rhs_cols)
        elif rhs_batch == 1:
            out_size = (lhs_batch, lhs_rows, rhs_cols)
        elif lhs_batch == rhs_batch:
            if len(lhs_shape) == 4:
                if not (lhs_shape[0] == rhs_shape[0] and lhs_shape[1] == rhs_shape[1]):
                    raise ValueError(f"not matched batch dims: {lhs_shape} x {rhs_shape}")
                out_size = (lhs_shape[0], lhs_shape[1], lhs_rows, rhs_cols)
            else:
                out_size = (lhs_batch, lhs_rows, rhs_cols)
        else:
            out_size = (lhs_rows, rhs_cols)

        self.output_shape = out_size

        self.rhs_t = torch.zeros(
            size=(rhs_batch, rhs_cols, rhs_rows),
            dtype=torch.float,
            device=self.device
        )

        self.lhs_wrapper = CudaBfpWrapper(
            fp_tensor_shape=lhs_shape, 
            use_multi_exp=self.use_multi_exp,
            apply_thresholding=apply_thresholding,
            threshold=threshold)
        self.rhs_wrapper = CudaBfpWrapper(
            fp_tensor_shape=self.rhs_t.shape,
            use_multi_exp=self.use_multi_exp, 
            apply_thresholding=apply_thresholding,
            threshold=threshold)

        self.output = torch.zeros(
            size=out_size,
            dtype=torch.float,
            device=self.device)

        self.output_add = torch.zeros(
            size=out_size,
            dtype=torch.float,
            device=self.device)

        self.use_stochastic_rounding = use_stochastic_rounding
        self.config = config if config is not None else BfpConfig

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

        self.use_shift = BfpConfig.use_shift

    def run(self, lhs: torch.Tensor, rhs: torch.Tensor):
        self.lhs_wrapper.run_convert_fp_to_bfp(src_tensor=lhs, is_stochastic_rounding=self.use_stochastic_rounding)

        memory_helper_lib.transpose_4d(
            c_void_p(self.rhs_t.data_ptr()),
            c_void_p(rhs.data_ptr()),
            c_size_t(1),
            c_size_t(self.rhs_t.shape[0]),
            c_size_t(self.rhs_t.shape[2]),
            c_size_t(self.rhs_t.shape[1])
        )

        self.rhs_wrapper.run_convert_fp_to_bfp(src_tensor=self.rhs_t, is_stochastic_rounding=self.use_stochastic_rounding)

        if self.lhs_wrapper.bfp_M.shape[-1] != self.rhs_wrapper.bfp_M.shape[-1]:
            raise ValueError(f"cannot matched dim: {self.lhs_wrapper.bfp_M.shape} x {self.rhs_wrapper.bfp_M.shape}")

        bfp_M_bit = self.config.bfp_M_Bit
        group_size = self.config.group_size

        if self.use_multi_exp and not self.use_shift:
            cuda_multi_exp_lib.bfp_gemm_2d_batched(
                c_void_p(self.lhs_wrapper.bfp_S.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_E.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_M.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_E_index_ptr.data_ptr()),


                c_void_p(self.rhs_wrapper.bfp_S.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_E.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_M.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_E_index_ptr.data_ptr()),

                c_void_p(self.output.data_ptr()),

                c_int32(self.lhs_B),
                c_int32(self.rhs_B),
                c_int32(self.M),
                c_int32(self.K),
                c_int32(self.N),

                c_int32(group_size),
                c_int32(self.lhs_wrapper.bfp_M_bit),
                c_int32(self.rhs_wrapper.bfp_M_bit)
            )
        elif self.use_multi_exp and self.use_shift:
            cuda_multi_exp_shift_lib.bfp_gemm_2d_batched(
                c_void_p(self.lhs_wrapper.bfp_S.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_E.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_M.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_E_index_ptr.data_ptr()),


                c_void_p(self.rhs_wrapper.bfp_S.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_E.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_M.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_E_index_ptr.data_ptr()),

                c_void_p(self.output.data_ptr()),

                c_int32(self.lhs_B),
                c_int32(self.rhs_B),
                c_int32(self.M),
                c_int32(self.K),
                c_int32(self.N),

                c_int32(group_size),
                c_int32(self.lhs_wrapper.bfp_M_bit),
                c_int32(self.rhs_wrapper.bfp_M_bit)
            )
        else:
            cuda_lib.bfp_gemm_2d_batched(
                c_void_p(self.lhs_wrapper.bfp_S.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_E.data_ptr()),
                c_void_p(self.lhs_wrapper.bfp_M.data_ptr()),

                c_void_p(self.rhs_wrapper.bfp_S.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_E.data_ptr()),
                c_void_p(self.rhs_wrapper.bfp_M.data_ptr()),

                c_void_p(self.output.data_ptr()),

                c_int32(self.lhs_B),
                c_int32(self.rhs_B),
                c_int32(self.M),
                c_int32(self.K),
                c_int32(self.N),

                c_int32(group_size),
                c_int32(self.lhs_wrapper.bfp_M_bit),
                c_int32(self.rhs_wrapper.bfp_M_bit)
            )

        return self.output

def load_inout(path_root, phase, layer_index):
    path = f"{path_root}/{phase}-{layer_index}-fp"

    lhs = np.load(f"{path}-lhs.npy", allow_pickle=False)
    rhs = np.load(f"{path}-rhs.npy", allow_pickle=False)
    output = np.matmul(lhs, rhs)

    return lhs, rhs, output

if __name__ == "__main__":
    set_reproducibility(1234)
    BfpConfig.group_size = 16
    BfpConfig.bfp_M_Bit = 4
    BfpConfig.use_shift = True
    use_multi_exp = True
    apply_thresholding = False
    threshold = 1
    use_SR = True

    # B = 8
    # M = 4
    # K = 128
    # N = 32
    
    # low = -5
    # high = 5
    # lhs = torch.randint(low=low, high=high, size=(M, K), dtype=torch.float).to("cuda")
    # rhs = torch.randint(low=low, high=high, size=(B, K, N), dtype=torch.float).to("cuda")
    # lhs = torch.randn(size=(M, K), dtype=torch.float).to("cuda")
    # rhs = torch.randn(size=(B, K, N), dtype=torch.float).to("cuda")

    lhs = np.load("matmul_npy/0_lhs.npy", allow_pickle=False)
    rhs = np.load("matmul_npy/0_rhs.npy", allow_pickle=False)
    fp_output = np.load("matmul_npy/0_output.npy", allow_pickle=False)

    lhs = torch.tensor(lhs, dtype=torch.float).to("cuda")
    rhs = torch.tensor(rhs, dtype=torch.float).to("cuda")

    print(f"{lhs.shape} x {rhs.shape} = {fp_output.shape}")

    print(f"{'bfp only' if not use_multi_exp else 'multi exp'}")
    # print(f"dims: [{M}, {K}] x [{K}, {N}]")

    lhs = torch.tensor(lhs, dtype=torch.float32)
    lhs = lhs.to('cuda')

    rhs = torch.tensor(rhs, dtype=torch.float32)
    rhs = rhs.to('cuda')
    
    bfp_matmul = CustomMatMul(
        lhs.shape, 
        rhs.shape, 
        use_stochastic_rounding=use_SR, 
        use_multi_exp=use_multi_exp,
        apply_thresholding=apply_thresholding,
        threshold=threshold)

    # print(f"lhs: {bfp_gemm.lhs_wrapper.bfp_M_bit}-bit, rhs: {bfp_gemm.rhs_wrapper.bfp_M_bit}-bit")

    outputs_cuda = bfp_matmul.run(lhs, rhs)

    fp_outputs = torch.matmul(lhs, rhs)

    print(f"bfp: {outputs_cuda.shape}, torch: {fp_outputs.shape}")

    # print(f"answer: {fp_outputs.cpu().numpy()[0,0]:.15f}")

    # diff = torch.abs(fp_outputs.flatten() - outputs.flatten())
    diff_cuda = torch.abs(fp_outputs.flatten() - outputs_cuda.flatten())
    errors = torch.abs((outputs_cuda - fp_outputs) / (fp_outputs + 1e-15)) * 100.

    print(f"avg error mean  : {torch.mean(errors.flatten()).item():.2f}")
    print(f"[diffs]\nmean: {torch.mean(diff_cuda):.15f}\nmax:  {torch.max(diff_cuda):.15f}\nmin:  {torch.min(diff_cuda):.15f}")