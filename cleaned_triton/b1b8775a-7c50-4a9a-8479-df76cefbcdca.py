
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
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_replication_pad3d_1 (in_ptr0 ,in_ptr1 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,ks6 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%ks1 )
    x2 =xindex //ks2 
    x5 =xindex //ks6 
    x6 =xindex 
    tmp0 =tl .load (in_ptr0 +(ks5 *(((-1 )+ks4 )*(((-1 )+ks4 )<=(((0 )*((0 )>=((-1 )+x1 ))+((-1 )+x1 )*(((-1 )+x1 )>(0 )))))+(((0 )*((0 )>=((-1 )+x1 ))+((-1 )+x1 )*(((-1 )+x1 )>(0 ))))*((((0 )*((0 )>=((-1 )+x1 ))+((-1 )+x1 )*(((-1 )+x1 )>(0 ))))<((-1 )+ks4 )))+ks4 *ks5 *(((-1 )+ks3 )*(((-1 )+ks3 )<=(((0 )*((0 )>=((-1 )+x2 ))+((-1 )+x2 )*(((-1 )+x2 )>(0 )))))+(((0 )*((0 )>=((-1 )+x2 ))+((-1 )+x2 )*(((-1 )+x2 )>(0 ))))*((((0 )*((0 )>=((-1 )+x2 ))+((-1 )+x2 )*(((-1 )+x2 )>(0 ))))<((-1 )+ks3 )))+(((-1 )+ks5 )*(((-1 )+ks5 )<=(((0 )*((0 )>=((-1 )+x0 ))+((-1 )+x0 )*(((-1 )+x0 )>(0 )))))+(((0 )*((0 )>=((-1 )+x0 ))+((-1 )+x0 )*(((-1 )+x0 )>(0 ))))*((((0 )*((0 )>=((-1 )+x0 ))+((-1 )+x0 )*(((-1 )+x0 )>(0 ))))<((-1 )+ks5 )))),xmask ,eviction_policy ='evict_last')
    tmp6 =tl .load (in_ptr1 +((((-1 )+ks3 )*(((-1 )+ks3 )<=(((0 )*((0 )>=((-1 )+x5 ))+((-1 )+x5 )*(((-1 )+x5 )>(0 )))))+(((0 )*((0 )>=((-1 )+x5 ))+((-1 )+x5 )*(((-1 )+x5 )>(0 ))))*((((0 )*((0 )>=((-1 )+x5 ))+((-1 )+x5 )*(((-1 )+x5 )>(0 ))))<((-1 )+ks3 )))),xmask ,eviction_policy ='evict_last')
    tmp1 =tl_math .abs (tmp0 )
    tmp2 =0.5 
    tmp3 =tmp1 <=tmp2 
    tmp4 =0.0 
    tmp5 =tl .where (tmp3 ,tmp4 ,tmp0 )
    tmp7 =tmp6 <tmp2 
    tmp8 =tmp7 .to (tl .float32 )
    tmp9 =0.8864048946659319 
    tmp10 =tmp8 *tmp9 
    tmp11 =tmp5 *tmp10 
    tmp12 =-1.0 
    tmp13 =tmp8 +tmp12 
    tmp14 =1.558387861036063 
    tmp15 =tmp13 *tmp14 
    tmp16 =0.7791939305180315 
    tmp17 =tmp15 +tmp16 
    tmp18 =tmp11 +tmp17 
    tl .store (out_ptr0 +(x6 ),tmp18 ,xmask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[1 ],out =buf0 )
        buf1 =empty_strided_cuda ((1 ,s0 ,1 ,1 ),(s0 ,1 ,s0 ,s0 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_bernoulli_0 [grid (s0 )](buf0 ,buf1 ,0 ,10 ,XBLOCK =16 ,num_warps =1 ,num_stages =1 )
        del buf0 
        2 +s2 
        2 +s1 
        4 +2 *s1 +2 *s2 +s1 *s2 
        4 +2 *s1 +2 *s2 +s1 *s2 
        buf2 =empty_strided_cuda ((1 ,1 ,2 +s0 ,2 +s1 ,2 +s2 ),(8 +4 *s0 +4 *s1 +4 *s2 +2 *s0 *s1 +2 *s0 *s2 +2 *s1 *s2 +s0 *s1 *s2 ,8 +4 *s0 +4 *s1 +4 *s2 +2 *s0 *s1 +2 *s0 *s2 +2 *s1 *s2 +s0 *s1 *s2 ,4 +2 *s1 +2 *s2 +s1 *s2 ,2 +s2 ,1 ),torch .float32 )

        triton_poi_fused_replication_pad3d_1_xnumel =8 +4 *s0 +4 *s1 +4 *s2 +2 *s0 *s1 +2 *s0 *s2 +2 *s1 *s2 +s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_replication_pad3d_1 [grid (triton_poi_fused_replication_pad3d_1_xnumel )](arg3_1 ,buf1 ,buf2 ,34 ,34 ,1156 ,10 ,32 ,32 ,1156 ,13872 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        del buf1 
    return (reinterpret_tensor (buf2 ,(1 ,1 ,8 +4 *s0 +4 *s1 +4 *s2 +2 *s0 *s1 +2 *s0 *s2 +2 *s1 *s2 +s0 *s1 *s2 ),(8 +4 *s0 +4 *s1 +4 *s2 +2 *s0 *s1 +2 *s0 *s2 +2 *s1 *s2 +s0 *s1 *s2 ,8 +4 *s0 +4 *s1 +4 *s2 +2 *s0 *s1 +2 *s0 *s2 +2 *s1 *s2 +s0 *s1 *s2 ,1 ),0 ),)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =10 
    arg1_1 =32 
    arg2_1 =32 
    arg3_1 =rand_strided ((1 ,10 ,32 ,32 ),(10240 ,1024 ,32 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
