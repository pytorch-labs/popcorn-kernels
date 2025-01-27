
import torch 
from torch ._inductor .select_algorithm import extern_kernels 
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
def triton_poi_fused_convolution_0 (in_out_ptr0 ,in_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =10816 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x2 =xindex 
    x1 =xindex //676 
    tmp0 =tl .load (in_out_ptr0 +(x2 ),xmask )
    tmp1 =tl .load (in_ptr0 +(x1 ),xmask ,eviction_policy ='evict_last')
    tmp2 =tmp0 +tmp1 
    tl .store (in_out_ptr0 +(x2 ),tmp2 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_constant_pad_nd_1 (in_ptr0 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =13520 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x1 =xindex //676 
    x2 =xindex 
    tmp0 =(-2 )+x1 
    tmp1 =tl .full ([1 ],0 ,tl .int64 )
    tmp2 =tmp0 >=tmp1 
    tmp3 =tl .full ([1 ],16 ,tl .int64 )
    tmp4 =tmp0 <tmp3 
    tmp5 =tmp2 &tmp4 
    tmp6 =tl .load (in_ptr0 +((-1352 )+x2 ),tmp5 &xmask ,other =0.0 )
    tmp7 =0.0 
    tmp8 =triton_helpers .maximum (tmp6 ,tmp7 )
    tmp9 =6.0 
    tmp10 =triton_helpers .minimum (tmp8 ,tmp9 )
    tmp11 =tmp10 *tmp10 
    tmp12 =tl .full (tmp11 .shape ,0.0 ,tmp11 .dtype )
    tmp13 =tl .where (tmp5 ,tmp11 ,tmp12 )
    tl .store (out_ptr0 +(x2 ),tmp13 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_add_avg_pool3d_div_hardtanh_mul_pow_2 (in_ptr0 ,in_ptr1 ,out_ptr0 ,out_ptr1 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =10816 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp1 =tl .load (in_ptr0 +(676 +x0 ),xmask )
    tmp3 =tl .load (in_ptr0 +(1352 +x0 ),xmask )
    tmp5 =tl .load (in_ptr0 +(2028 +x0 ),xmask )
    tmp7 =tl .load (in_ptr0 +(2704 +x0 ),xmask )
    tmp11 =tl .load (in_ptr1 +(x0 ),xmask )
    tmp2 =tmp1 +tmp0 
    tmp4 =tmp3 +tmp2 
    tmp6 =tmp5 +tmp4 
    tmp8 =tmp7 +tmp6 
    tmp9 =0.2 
    tmp10 =tmp8 *tmp9 
    tmp12 =0.0 
    tmp13 =triton_helpers .maximum (tmp11 ,tmp12 )
    tmp14 =6.0 
    tmp15 =triton_helpers .minimum (tmp13 ,tmp14 )
    tmp16 =0.0001 
    tmp17 =tmp10 *tmp16 
    tmp18 =1.0 
    tmp19 =tmp17 +tmp18 
    tmp20 =0.75 
    tmp21 =libdevice .pow (tmp19 ,tmp20 )
    tmp22 =tmp15 /tmp21 
    tl .store (out_ptr0 +(x0 ),tmp10 ,xmask )
    tl .store (out_ptr1 +(x0 ),tmp22 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__to_copy_3 (out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =52 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =x0 
    tmp1 =tmp0 .to (tl .float32 )
    tmp2 =0.49019607843137253 
    tmp3 =tmp1 *tmp2 
    tmp4 =0.0 
    tmp5 =triton_helpers .maximum (tmp3 ,tmp4 )
    tmp6 =tmp5 .to (tl .int32 )
    tl .store (out_ptr0 +(x0 ),tmp6 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_add_clamp_4 (out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =52 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =x0 
    tmp1 =tmp0 .to (tl .float32 )
    tmp2 =0.49019607843137253 
    tmp3 =tmp1 *tmp2 
    tmp4 =0.0 
    tmp5 =triton_helpers .maximum (tmp3 ,tmp4 )
    tmp6 =tmp5 .to (tl .int32 )
    tmp7 =tl .full ([1 ],1 ,tl .int64 )
    tmp8 =tmp6 +tmp7 
    tmp9 =tl .full ([1 ],25 ,tl .int64 )
    tmp10 =triton_helpers .minimum (tmp8 ,tmp9 )
    tl .store (out_ptr0 +(x0 ),tmp10 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__to_copy_arange_clamp_mul_sub_5 (out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =52 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =x0 
    tmp1 =tmp0 .to (tl .float32 )
    tmp2 =0.49019607843137253 
    tmp3 =tmp1 *tmp2 
    tmp4 =0.0 
    tmp5 =triton_helpers .maximum (tmp3 ,tmp4 )
    tmp6 =tmp5 .to (tl .int32 )
    tmp7 =tmp6 .to (tl .float32 )
    tmp8 =tmp5 -tmp7 
    tmp9 =triton_helpers .maximum (tmp8 ,tmp4 )
    tmp10 =1.0 
    tmp11 =triton_helpers .minimum (tmp9 ,tmp10 )
    tl .store (out_ptr0 +(x0 ),tmp11 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__unsafe_index_add_mul_sub_6 (in_ptr0 ,in_ptr1 ,in_ptr2 ,in_ptr3 ,in_ptr4 ,in_ptr5 ,in_ptr6 ,out_ptr1 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =43264 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x1 =((xindex //52 )%52 )
    x0 =(xindex %52 )
    x2 =xindex //2704 
    (xindex %2704 )
    x4 =xindex 
    tmp0 =tl .load (in_ptr0 +(x1 ),xmask ,eviction_policy ='evict_last')
    tmp5 =tl .load (in_ptr1 +(x0 ),xmask ,eviction_policy ='evict_last')
    tmp10 =tl .load (in_ptr3 +(x0 ),xmask ,eviction_policy ='evict_last')
    tmp16 =tl .load (in_ptr4 +(x0 ),xmask ,eviction_policy ='evict_last')
    tmp19 =tl .load (in_ptr5 +(x1 ),xmask ,eviction_policy ='evict_last')
    tmp29 =tl .load (in_ptr6 +(x1 ),xmask ,eviction_policy ='evict_last')
    tmp1 =tl .full ([XBLOCK ],26 ,tl .int32 )
    tmp2 =tmp0 +tmp1 
    tmp3 =tmp0 <0 
    tmp4 =tl .where (tmp3 ,tmp2 ,tmp0 )
    tmp6 =tmp5 +tmp1 
    tmp7 =tmp5 <0 
    tmp8 =tl .where (tmp7 ,tmp6 ,tmp5 )
    tmp9 =tl .load (in_ptr2 +(tmp8 +26 *tmp4 +676 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp11 =tmp10 +tmp1 
    tmp12 =tmp10 <0 
    tmp13 =tl .where (tmp12 ,tmp11 ,tmp10 )
    tmp14 =tl .load (in_ptr2 +(tmp13 +26 *tmp4 +676 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp15 =tmp14 -tmp9 
    tmp17 =tmp15 *tmp16 
    tmp18 =tmp9 +tmp17 
    tmp20 =tmp19 +tmp1 
    tmp21 =tmp19 <0 
    tmp22 =tl .where (tmp21 ,tmp20 ,tmp19 )
    tmp23 =tl .load (in_ptr2 +(tmp8 +26 *tmp22 +676 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp24 =tl .load (in_ptr2 +(tmp13 +26 *tmp22 +676 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp25 =tmp24 -tmp23 
    tmp26 =tmp25 *tmp16 
    tmp27 =tmp23 +tmp26 
    tmp28 =tmp27 -tmp18 
    tmp30 =tmp28 *tmp29 
    tmp31 =tmp18 +tmp30 
    tl .store (out_ptr1 +(x4 ),tmp31 ,xmask )

def call (args ):
    primals_1 ,primals_2 ,primals_3 =args 
    args .clear ()
    assert_size_stride (primals_1 ,(1 ,784 ),(784 ,1 ))
    assert_size_stride (primals_2 ,(16 ,1 ,3 ,3 ),(9 ,9 ,3 ,1 ))
    assert_size_stride (primals_3 ,(16 ,),(1 ,))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )

        buf0 =extern_kernels .convolution (reinterpret_tensor (primals_1 ,(1 ,1 ,28 ,28 ),(784 ,784 ,28 ,1 ),0 ),primals_2 ,stride =(1 ,1 ),padding =(0 ,0 ),dilation =(1 ,1 ),transposed =False ,output_padding =(0 ,0 ),groups =1 ,bias =None )
        assert_size_stride (buf0 ,(1 ,16 ,26 ,26 ),(10816 ,676 ,26 ,1 ))
        buf1 =buf0 ;del buf0 

        get_raw_stream (0 )
        triton_poi_fused_convolution_0 [grid (10816 )](buf1 ,primals_3 ,10816 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del primals_3 
        buf2 =empty_strided_cuda ((1 ,1 ,20 ,26 ,26 ),(13536 ,13536 ,676 ,26 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_constant_pad_nd_1 [grid (13520 )](buf1 ,buf2 ,13520 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        buf3 =empty_strided_cuda ((1 ,1 ,16 ,26 ,26 ),(10816 ,10816 ,676 ,26 ,1 ),torch .float32 )
        buf4 =empty_strided_cuda ((1 ,16 ,26 ,26 ),(10816 ,676 ,26 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_add_avg_pool3d_div_hardtanh_mul_pow_2 [grid (10816 )](buf2 ,buf1 ,buf3 ,buf4 ,10816 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        buf5 =empty_strided_cuda ((52 ,1 ),(1 ,1 ),torch .int64 )

        get_raw_stream (0 )
        triton_poi_fused__to_copy_3 [grid (52 )](buf5 ,52 ,XBLOCK =64 ,num_warps =1 ,num_stages =1 )
        buf6 =empty_strided_cuda ((52 ,1 ),(1 ,1 ),torch .int64 )

        get_raw_stream (0 )
        triton_poi_fused_add_clamp_4 [grid (52 )](buf6 ,52 ,XBLOCK =64 ,num_warps =1 ,num_stages =1 )
        buf7 =empty_strided_cuda ((52 ,),(1 ,),torch .int64 )

        get_raw_stream (0 )
        triton_poi_fused__to_copy_3 [grid (52 )](buf7 ,52 ,XBLOCK =64 ,num_warps =1 ,num_stages =1 )
        buf8 =empty_strided_cuda ((52 ,),(1 ,),torch .int64 )

        get_raw_stream (0 )
        triton_poi_fused_add_clamp_4 [grid (52 )](buf8 ,52 ,XBLOCK =64 ,num_warps =1 ,num_stages =1 )
        buf9 =empty_strided_cuda ((52 ,),(1 ,),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused__to_copy_arange_clamp_mul_sub_5 [grid (52 )](buf9 ,52 ,XBLOCK =64 ,num_warps =1 ,num_stages =1 )
        buf11 =empty_strided_cuda ((52 ,1 ),(1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused__to_copy_arange_clamp_mul_sub_5 [grid (52 )](buf11 ,52 ,XBLOCK =64 ,num_warps =1 ,num_stages =1 )
        buf12 =empty_strided_cuda ((1 ,16 ,52 ,52 ),(43264 ,2704 ,52 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused__unsafe_index_add_mul_sub_6 [grid (43264 )](buf5 ,buf7 ,buf4 ,buf8 ,buf9 ,buf6 ,buf11 ,buf12 ,43264 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
    return (buf12 ,primals_2 ,reinterpret_tensor (primals_1 ,(1 ,1 ,28 ,28 ),(784 ,784 ,28 ,1 ),0 ),buf1 ,buf2 ,buf3 ,buf4 ,buf5 ,buf6 ,buf7 ,buf8 ,buf9 ,buf11 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    primals_1 =rand_strided ((1 ,784 ),(784 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_2 =rand_strided ((16 ,1 ,3 ,3 ),(9 ,9 ,3 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_3 =rand_strided ((16 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([primals_1 ,primals_2 ,primals_3 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
