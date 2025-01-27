
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
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused__softmax_0 (in_ptr0 ,out_ptr0 ,out_ptr1 ,ks0 ,ks1 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    _tmp2 =tl .full ([XBLOCK ,R0_BLOCK ],float ("-inf"),tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_1 =r0_index 
        tmp0 =tl .load (in_ptr0 +(x0 +ks0 *ks1 *r0_1 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
        tmp3 =triton_helpers .maximum (_tmp2 ,tmp1 )
        _tmp2 =tl .where (r0_mask &xmask ,tmp3 ,_tmp2 )
    tmp2 =triton_helpers .max2 (_tmp2 ,1 )[:,None ]
    tl .store (out_ptr0 +(x0 ),tmp2 ,xmask )
    _tmp8 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_1 =r0_index 
        tmp4 =tl .load (in_ptr0 +(x0 +ks0 *ks1 *r0_1 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp5 =tmp4 -tmp2 
        tmp6 =tl_math .exp (tmp5 )
        tmp7 =tl .broadcast_to (tmp6 ,[XBLOCK ,R0_BLOCK ])
        tmp9 =_tmp8 +tmp7 
        _tmp8 =tl .where (r0_mask &xmask ,tmp9 ,_tmp8 )
    tmp8 =tl .sum (_tmp8 ,1 )[:,None ]
    tl .store (out_ptr1 +(x0 ),tmp8 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__softmax_1 (in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr0 ,ks0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x2 =xindex 
    x0 =(xindex %ks0 )
    tmp0 =tl .load (in_ptr0 +(x2 ),xmask ,eviction_policy ='evict_last')
    tmp1 =tl .load (in_ptr1 +(x0 ),xmask ,eviction_policy ='evict_last')
    tmp4 =tl .load (in_ptr2 +(x0 ),xmask ,eviction_policy ='evict_last')
    tmp2 =tmp0 -tmp1 
    tmp3 =tl_math .exp (tmp2 )
    tmp5 =tmp3 /tmp4 
    tl .store (out_ptr0 +(x2 ),tmp5 ,xmask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,1 ,s1 ,s2 ),(s1 *s2 ,s1 *s2 ,s2 ,1 ),torch .float32 )
        buf1 =empty_strided_cuda ((1 ,1 ,s1 ,s2 ),(s1 *s2 ,s1 *s2 ,s2 ,1 ),torch .float32 )

        triton_red_fused__softmax_0_xnumel =s1 *s2 
        get_raw_stream (0 )
        triton_red_fused__softmax_0 [grid (triton_red_fused__softmax_0_xnumel )](arg3_1 ,buf0 ,buf1 ,64 ,64 ,4096 ,3 ,XBLOCK =128 ,R0_BLOCK =4 ,num_warps =4 ,num_stages =1 )
        s1 *s2 
        buf2 =empty_strided_cuda ((1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ),torch .float32 )

        triton_poi_fused__softmax_1_xnumel =s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused__softmax_1 [grid (triton_poi_fused__softmax_1_xnumel )](arg3_1 ,buf0 ,buf1 ,buf2 ,4096 ,12288 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        del buf0 
        del buf1 
    return (reinterpret_tensor (buf2 ,(1 ,s1 *s2 ,s0 ),(s0 *s1 *s2 ,1 ,s1 *s2 ),0 ),s1 ,s2 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =3 
    arg1_1 =64 
    arg2_1 =64 
    arg3_1 =rand_strided ((1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
