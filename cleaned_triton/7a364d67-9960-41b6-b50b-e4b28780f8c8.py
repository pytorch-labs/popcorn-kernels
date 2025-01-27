
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
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused_abs_add_exp_mean_mul_norm_sub_0 (in_out_ptr0 ,in_ptr0 ,ks0 ,ks1 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    _tmp7 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_0 =r0_index 
        tmp0 =tl .load (in_ptr0 +(r0_0 ),r0_mask ,eviction_policy ='evict_last',other =0.0 )
        tmp1 =tl .load (in_ptr0 +(r0_0 +ks0 *ks1 ),r0_mask ,eviction_policy ='evict_first',other =0.0 )
        tmp2 =tmp0 -tmp1 
        tmp3 =1e-06 
        tmp4 =tmp2 +tmp3 
        tmp5 =tmp4 *tmp4 
        tmp6 =tl .broadcast_to (tmp5 ,[XBLOCK ,R0_BLOCK ])
        tmp8 =_tmp7 +tmp6 
        _tmp7 =tl .where (r0_mask ,tmp8 ,_tmp7 )
    tmp7 =tl .sum (_tmp7 ,1 )[:,None ]
    tmp9 =libdevice .sqrt (tmp7 )
    tmp10 =tl_math .abs (tmp9 )
    tmp11 =1.0 
    tmp12 =tmp10 +tmp11 
    tmp13 =tmp9 /tmp12 
    tmp14 =tl_math .exp (tmp13 )
    tmp15 =tmp14 -tmp13 
    tmp16 =tmp15 /tmp11 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp16 ,None )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 =args 
    args .clear ()
    s1 =arg0_1 
    s2 =arg1_1 
    assert_size_stride (arg2_1 ,(1 ,2 ,s1 ,s2 ),(2 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,),(1 ,),torch .float32 )
        buf1 =reinterpret_tensor (buf0 ,(),(),0 );del buf0 

        s1 *s2 
        get_raw_stream (0 )
        triton_red_fused_abs_add_exp_mean_mul_norm_sub_0 [grid (1 )](buf1 ,arg2_1 ,64 ,64 ,1 ,4096 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del arg2_1 
    return (buf1 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =64 
    arg1_1 =64 
    arg2_1 =rand_strided ((1 ,2 ,64 ,64 ),(8192 ,4096 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
