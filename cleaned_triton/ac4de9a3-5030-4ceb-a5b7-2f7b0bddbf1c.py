
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
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_bernoulli_0 (in_ptr0 ,out_ptr0 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =x0 
    tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tl .store (out_ptr0 +(x0 ),tmp2 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_bernoulli_1 (in_ptr0 ,out_ptr0 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    tl .full ([XBLOCK ],True ,tl .int1 )
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =tl .full ([1 ],0 ,tl .int32 )
    tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tl .store (out_ptr0 +(tl .full ([XBLOCK ],0 ,tl .int32 )),tmp2 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__to_copy_add_bernoulli_mul_2 (in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr0 ,ks0 ,ks1 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp1 =tl .load (in_ptr1 +(x0 //(ks0 *ks1 )),xmask ,eviction_policy ='evict_last')
    tmp8 =tl .load (in_ptr2 +(0 ))
    tmp9 =tl .broadcast_to (tmp8 ,[XBLOCK ])
    tmp2 =0.5 
    tmp3 =tmp1 <tmp2 
    tmp4 =tmp3 .to (tl .float32 )
    tmp5 =2.0 
    tmp6 =tmp4 *tmp5 
    tmp7 =tmp0 *tmp6 
    tmp10 =tmp9 <tmp2 
    tmp11 =tmp10 .to (tl .float32 )
    tmp12 =0.8864048946659319 
    tmp13 =tmp11 *tmp12 
    tmp14 =tmp7 *tmp13 
    tmp15 =-1.0 
    tmp16 =tmp11 +tmp15 
    tmp17 =1.558387861036063 
    tmp18 =tmp16 *tmp17 
    tmp19 =0.7791939305180315 
    tmp20 =tmp18 +tmp19 
    tmp21 =tmp14 +tmp20 
    tl .store (out_ptr0 +(x0 ),tmp21 ,xmask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((2 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[2 ],out =buf0 )
        buf1 =empty_strided_cuda ((1 ,s0 ,1 ,1 ),(s0 ,1 ,s0 ,s0 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_bernoulli_0 [grid (s0 )](buf0 ,buf1 ,0 ,3 ,XBLOCK =4 ,num_warps =1 ,num_stages =1 )
        buf2 =empty_strided_cuda ((1 ,1 ,1 ),(1 ,1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_bernoulli_1 [grid (1 )](buf0 ,buf2 ,1 ,1 ,XBLOCK =1 ,num_warps =1 ,num_stages =1 )
        del buf0 
        buf3 =empty_strided_cuda ((1 ,1 ,s0 *s1 *s2 ),(s0 *s1 *s2 ,s0 *s1 *s2 ,1 ),torch .float32 )

        triton_poi_fused__to_copy_add_bernoulli_mul_2_xnumel =s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused__to_copy_add_bernoulli_mul_2 [grid (triton_poi_fused__to_copy_add_bernoulli_mul_2_xnumel )](arg3_1 ,buf1 ,buf2 ,buf3 ,64 ,64 ,12288 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        del buf1 
        del buf2 
    return (reinterpret_tensor (buf3 ,(1 ,s0 *s1 *s2 ),(s0 *s1 *s2 ,1 ),0 ),)

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
