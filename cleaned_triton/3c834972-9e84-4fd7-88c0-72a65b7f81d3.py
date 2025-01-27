
import torch 
import triton 
import triton .language as tl 
from torch ._inductor .runtime .triton_heuristics import (
grid ,
)
from torch ._C import _cuda_getCurrentRawStream as get_raw_stream 
from torch ._C import _cuda_getCurrentRawStream as get_raw_stream 

aten =torch .ops .aten 
inductor_ops =torch .ops .inductor 
_quantized =torch .ops ._quantized 
assert_size_stride =torch ._C ._dynamo .guards .assert_size_stride 
empty_strided_cpu =torch ._C ._dynamo .guards ._empty_strided_cpu 
empty_strided_cuda =torch ._C ._dynamo .guards ._empty_strided_cuda 
empty_strided_xpu =torch ._C ._dynamo .guards ._empty_strided_xpu 
reinterpret_tensor =torch ._C ._dynamo .guards ._reinterpret_tensor 
alloc_from_pool =torch .ops .inductor ._alloc_from_pool 

empty_strided_p2p =torch ._C ._distributed_c10d ._SymmetricMemory .empty_strided_p2p 

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused__native_batch_norm_legit_0 (in_ptr0 ,out_ptr2 ,ks0 ,ks1 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =128 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    tmp2_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp2_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp2_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        roffset =r0_offset 
        r0_1 =r0_index 
        tmp0 =tl .load (in_ptr0 +(r0_1 +ks0 *ks1 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
        tmp2_mean_next ,tmp2_m2_next ,tmp2_weight_next =triton_helpers .welford_reduce (
        tmp1 ,tmp2_mean ,tmp2_m2 ,tmp2_weight ,roffset ==0 
        )
        tmp2_mean =tl .where (r0_mask &xmask ,tmp2_mean_next ,tmp2_mean )
        tmp2_m2 =tl .where (r0_mask &xmask ,tmp2_m2_next ,tmp2_m2 )
        tmp2_weight =tl .where (r0_mask &xmask ,tmp2_weight_next ,tmp2_weight )
    tmp5 ,tmp6 ,tmp7 =triton_helpers .welford (tmp2_mean ,tmp2_m2 ,tmp2_weight ,1 )
    tmp2 =tmp5 [:,None ]
    tmp3 =tmp6 [:,None ]
    tmp4 =tmp7 [:,None ]
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        roffset =r0_offset 
        r0_1 =r0_index 
        tmp8 =tl .load (in_ptr0 +(r0_1 +ks0 *ks1 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp9 =tmp8 -tmp2 
        tmp10 =ks0 *ks1 
        tmp11 =tmp10 .to (tl .float32 )
        tmp12 =tmp3 /tmp11 
        tmp13 =1e-05 
        tmp14 =tmp12 +tmp13 
        tmp15 =libdevice .rsqrt (tmp14 )
        tmp16 =tmp9 *tmp15 
        tl .store (out_ptr2 +(r0_1 +ks0 *ks1 *x0 ),tmp16 ,r0_mask &xmask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 =args 
    args .clear ()
    s1 =arg0_1 
    s2 =arg1_1 
    assert_size_stride (arg2_1 ,(1 ,128 ,s1 ,s2 ),(128 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf3 =empty_strided_cuda ((1 ,128 ,s1 *s2 ),(128 *s1 *s2 ,s1 *s2 ,1 ),torch .float32 )

        s1 *s2 
        get_raw_stream (0 )
        triton_red_fused__native_batch_norm_legit_0 [grid (128 )](arg2_1 ,buf3 ,64 ,64 ,128 ,4096 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del arg2_1 
    return (reinterpret_tensor (buf3 ,(1 ,s1 *s2 ,128 ),(128 *s1 *s2 ,1 ,s1 *s2 ),0 ),s1 ,s2 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =64 
    arg1_1 =64 
    arg2_1 =rand_strided ((1 ,128 ,64 ,64 ),(524288 ,4096 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
