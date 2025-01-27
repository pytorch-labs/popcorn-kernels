
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
def triton_poi_fused_bernoulli_0 (in_ptr0 ,out_ptr0 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
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
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
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
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__to_copy_bernoulli_div_log_sigmoid_forward_mul_2 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp12 =tl .load (in_ptr1 +(0 ))
    tmp13 =tl .broadcast_to (tmp12 ,[XBLOCK ])
    tmp33 =tl .load (in_ptr2 +(0 ))
    tmp34 =tl .broadcast_to (tmp33 ,[XBLOCK ])
    tmp1 =libdevice .tanh (tmp0 )
    tmp2 =tmp0 -tmp1 
    tmp3 =3.0 
    tmp4 =tmp2 +tmp3 
    tmp5 =0.0 
    tmp6 =triton_helpers .maximum (tmp4 ,tmp5 )
    tmp7 =6.0 
    tmp8 =triton_helpers .minimum (tmp6 ,tmp7 )
    tmp9 =tmp2 *tmp8 
    tmp10 =0.16666666666666666 
    tmp11 =tmp9 *tmp10 
    tmp14 =0.5 
    tmp15 =tmp13 <tmp14 
    tmp16 =tmp15 .to (tl .float32 )
    tmp17 =2.0 
    tmp18 =tmp16 *tmp17 
    tmp19 =tmp11 *tmp18 
    tmp20 =triton_helpers .minimum (tmp5 ,tmp19 )
    tmp21 =tl_math .abs (tmp19 )
    tmp22 =-tmp21 
    tmp23 =tl_math .exp (tmp22 )
    tmp24 =libdevice .log1p (tmp23 )
    tmp25 =tmp20 -tmp24 
    tmp26 =libdevice .tanh (tmp25 )
    tmp27 =tmp25 -tmp26 
    tmp28 =tmp27 +tmp3 
    tmp29 =triton_helpers .maximum (tmp28 ,tmp5 )
    tmp30 =triton_helpers .minimum (tmp29 ,tmp7 )
    tmp31 =tmp27 *tmp30 
    tmp32 =tmp31 *tmp10 
    tmp35 =tmp34 <tmp14 
    tmp36 =tmp35 .to (tl .float32 )
    tmp37 =tmp36 *tmp17 
    tmp38 =tmp32 *tmp37 
    tmp39 =triton_helpers .minimum (tmp5 ,tmp38 )
    tmp40 =tl_math .abs (tmp38 )
    tmp41 =-tmp40 
    tmp42 =tl_math .exp (tmp41 )
    tmp43 =libdevice .log1p (tmp42 )
    tmp44 =tmp39 -tmp43 
    tl .store (in_out_ptr0 +(x0 ),tmp44 ,xmask )

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
        buf1 =empty_strided_cuda ((1 ,1 ,1 ),(1 ,1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_bernoulli_0 [grid (1 )](buf0 ,buf1 ,0 ,1 ,XBLOCK =1 ,num_warps =1 ,num_stages =1 )
        buf2 =empty_strided_cuda ((1 ,1 ,1 ),(1 ,1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_bernoulli_1 [grid (1 )](buf0 ,buf2 ,1 ,1 ,XBLOCK =1 ,num_warps =1 ,num_stages =1 )
        del buf0 
        buf3 =empty_strided_cuda ((1 ,1 ,s0 *s1 *s2 ),(s0 *s1 *s2 ,s0 *s1 *s2 ,1 ),torch .float32 )
        buf4 =reinterpret_tensor (buf3 ,(1 ,s0 *s1 *s2 ),(s0 *s1 *s2 ,1 ),0 );del buf3 

        triton_poi_fused__to_copy_bernoulli_div_log_sigmoid_forward_mul_2_xnumel =s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused__to_copy_bernoulli_div_log_sigmoid_forward_mul_2 [grid (triton_poi_fused__to_copy_bernoulli_div_log_sigmoid_forward_mul_2_xnumel )](buf4 ,arg3_1 ,buf1 ,buf2 ,12288 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        del buf1 
        del buf2 
    return (buf4 ,)

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
