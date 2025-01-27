
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
def triton_per_fused_constant_pad_nd_mean_var_0 (in_out_ptr0 ,in_out_ptr1 ,in_ptr0 ,xnumel ,r0_numel ):
    XBLOCK :tl .constexpr =1 
    R0_BLOCK :tl .constexpr =1024 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    tl .full ([1 ],xoffset ,tl .int32 )
    tl .full ([R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[:]
    tl .full ([R0_BLOCK ],True ,tl .int1 )
    r0_1 =r0_index //32 
    r0_0 =(r0_index %32 )
    tmp0 =(-2 )+r0_1 
    tmp1 =tl .full ([1 ],0 ,tl .int64 )
    tmp2 =tmp0 >=tmp1 
    tmp3 =tl .full ([1 ],28 ,tl .int64 )
    tmp4 =tmp0 <tmp3 
    tmp5 =(-2 )+r0_0 
    tmp6 =tmp5 >=tmp1 
    tmp7 =tmp5 <tmp3 
    tmp8 =tmp2 &tmp4 
    tmp9 =tmp8 &tmp6 
    tmp10 =tmp9 &tmp7 
    tmp11 =tl .load (in_ptr0 +(tl .broadcast_to ((-58 )+r0_0 +28 *r0_1 ,[R0_BLOCK ])),tmp10 ,other =0.0 )
    tmp12 =tl .broadcast_to (tmp11 ,[R0_BLOCK ])
    tmp14 =triton_helpers .promote_to_tensor (tl .sum (tmp12 ,0 ))
    tmp16 =tl .broadcast_to (tmp12 ,[R0_BLOCK ])
    tmp18 =triton_helpers .promote_to_tensor (tl .sum (tmp16 ,0 ))
    tmp19 =tl .full ([1 ],1024 ,tl .int32 )
    tmp20 =tmp19 .to (tl .float32 )
    tmp21 =tmp18 /tmp20 
    tmp22 =tmp12 -tmp21 
    tmp23 =tmp22 *tmp22 
    tmp24 =tl .broadcast_to (tmp23 ,[R0_BLOCK ])
    tmp26 =triton_helpers .promote_to_tensor (tl .sum (tmp24 ,0 ))
    tmp27 =1024.0 
    tmp28 =tmp14 /tmp27 
    tmp29 =1023.0 
    tmp30 =tmp26 /tmp29 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([1 ],0 ,tl .int32 )),tmp28 ,None )
    tl .debug_barrier ()
    tl .store (in_out_ptr1 +(tl .full ([1 ],0 ,tl .int32 )),tmp30 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_randn_like_1 (in_ptr0 ,out_ptr0 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    tl .full ([XBLOCK ],True ,tl .int1 )
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =tl .full ([1 ],0 ,tl .int32 )
    tmp2 =tl .randn (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tl .store (out_ptr0 +(tl .full ([XBLOCK ],0 ,tl .int32 )),tmp2 ,None )

def call (args ):
    arg0_1 ,=args 
    args .clear ()
    assert_size_stride (arg0_1 ,(1 ,784 ),(784 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,1 ,1 ,1 ),(1 ,1 ,1 ,1 ),torch .float32 )
        buf5 =empty_strided_cuda ((1 ,1 ,1 ,1 ),(1 ,1 ,1 ,1 ),torch .float32 )
        buf1 =buf0 ;del buf0 
        buf7 =buf5 ;del buf5 

        get_raw_stream (0 )
        triton_per_fused_constant_pad_nd_mean_var_0 [grid (1 )](buf1 ,buf7 ,arg0_1 ,1 ,1024 ,num_warps =8 ,num_stages =1 )
        del arg0_1 
        buf2 =empty_strided_cuda ((1 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[1 ],out =buf2 )
        buf3 =empty_strided_cuda ((1 ,1 ),(1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_randn_like_1 [grid (1 )](buf2 ,buf3 ,0 ,1 ,XBLOCK =1 ,num_warps =1 ,num_stages =1 )
        del buf2 
    return (reinterpret_tensor (buf1 ,(1 ,1 ),(1 ,1 ),0 ),buf3 ,reinterpret_tensor (buf7 ,(1 ,1 ),(1 ,1 ),0 ),)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =rand_strided ((1 ,784 ),(784 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
