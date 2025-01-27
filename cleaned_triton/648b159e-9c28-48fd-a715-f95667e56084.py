
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
def triton_poi_fused_rand_0 (in_ptr0 ,out_ptr0 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =3 
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
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_abs_add_div_1 (in_ptr0 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =512 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp1 =tl_math .abs (tmp0 )
    tmp2 =1.0 
    tmp3 =tmp1 +tmp2 
    tmp4 =tmp0 /tmp3 
    tl .store (out_ptr0 +(x0 ),tmp4 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_abs_add_div_rrelu_with_noise_functional_2 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =512 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_out_ptr0 +(x0 ),xmask )
    tmp7 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp8 =tl .load (in_ptr1 +(x0 ),xmask )
    tmp1 =tl_math .abs (tmp0 )
    tmp2 =1.0 
    tmp3 =tmp1 +tmp2 
    tmp4 =tmp0 /tmp3 
    tmp5 =0.0 
    tmp6 =tmp4 <=tmp5 
    tmp9 =tmp7 *tmp8 
    tmp10 =tl .where (tmp6 ,tmp9 ,tmp7 )
    tl .store (in_out_ptr0 +(x0 ),tmp10 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_rrelu_with_noise_functional_sub_tanh_3 (in_out_ptr0 ,in_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =512 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_out_ptr0 +(x0 ),xmask )
    tmp3 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp1 =0.0 
    tmp2 =tmp0 <=tmp1 
    tmp4 =tmp0 *tmp3 
    tmp5 =tl .where (tmp2 ,tmp4 ,tmp0 )
    tmp6 =libdevice .tanh (tmp5 )
    tmp7 =tmp5 -tmp6 
    tl .store (in_out_ptr0 +(x0 ),tmp7 ,xmask )

def call (args ):
    arg0_1 ,=args 
    args .clear ()
    assert_size_stride (arg0_1 ,(1 ,1 ,16 ,16 ,16 ),(4096 ,4096 ,256 ,16 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[1 ],out =buf0 )
        buf1 =empty_strided_cuda ((1 ,1 ,3 ),(3 ,3 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_rand_0 [grid (3 )](buf0 ,buf1 ,0 ,3 ,XBLOCK =4 ,num_warps =1 ,num_stages =1 )
        del buf0 

        buf2 =torch .ops .aten .fractional_max_pool3d .default (arg0_1 ,[2 ,2 ,2 ],[8 ,8 ,8 ],buf1 )
        del arg0_1 
        del buf1 
        buf3 =buf2 [0 ]
        del buf2 
        buf5 =empty_strided_cuda ((1 ,1 ,8 ,8 ,8 ),(512 ,512 ,64 ,8 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_abs_add_div_1 [grid (512 )](buf3 ,buf5 ,512 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )

        buf6 =torch .ops .aten .uniform .default (buf5 ,0.125 ,0.3333333333333333 )
        buf7 =buf6 
        del buf6 
        buf8 =buf3 ;del buf3 

        get_raw_stream (0 )
        triton_poi_fused_abs_add_div_rrelu_with_noise_functional_2 [grid (512 )](buf8 ,buf5 ,buf7 ,512 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf5 
        del buf7 

        buf9 =torch .ops .aten .uniform .default (buf8 ,0.125 ,0.3333333333333333 )
        buf10 =buf9 
        del buf9 
        buf11 =reinterpret_tensor (buf8 ,(1 ,1 ,8 ,8 ,8 ),(512 ,1 ,64 ,8 ,1 ),0 );del buf8 

        get_raw_stream (0 )
        triton_poi_fused_rrelu_with_noise_functional_sub_tanh_3 [grid (512 )](buf11 ,buf10 ,512 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf10 
    return (reinterpret_tensor (buf11 ,(1 ,512 ),(512 ,1 ),0 ),)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =rand_strided ((1 ,1 ,16 ,16 ,16 ),(4096 ,4096 ,256 ,16 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
