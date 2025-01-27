
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
def triton_poi_fused_abs_mul_pow_relu_sign_0 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ynumel ,xnumel ,YBLOCK :tl .constexpr ,XBLOCK :tl .constexpr ):
    yoffset =(tl .program_id (1 )+tl .program_id (2 )*tl .num_programs (1 ))*YBLOCK 
    yindex =yoffset +tl .arange (0 ,YBLOCK )[None ,:]
    ymask =yindex <ynumel 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    x1 =xindex 
    y0 =yindex 
    tl .device_assert (((((x1 //3 )%3 ))+((((2 *y0 )//((-2 )+ks1 ))%((-2 )+ks0 )))<ks0 )|~(xmask &ymask ),"index out of bounds: (((x1 // 3) % 3)) + ((((2*y0) // ((-2) + ks1)) % ((-2) + ks0))) < ks0")
    tl .device_assert ((((x1 %3 ))+(((2 *y0 )%((-2 )+ks1 )))<ks1 )|~(xmask &ymask ),"index out of bounds: ((x1 % 3)) + (((2*y0) % ((-2) + ks1))) < ks1")
    tmp2 =tl .load (in_ptr0 +(ks1 *(((x1 //3 )%3 ))+ks1 *((((2 *y0 )//((-2 )+ks1 ))%((-2 )+ks0 )))+ks0 *ks1 *(x1 //9 )+((x1 %3 ))+(((2 *y0 )%((-2 )+ks1 )))),xmask &ymask ,eviction_policy ='evict_last')
    tl .device_assert (((((x1 //3 )%3 ))+((((1 +2 *y0 )//((-2 )+ks1 ))%((-2 )+ks0 )))<ks0 )|~(xmask &ymask ),"index out of bounds: (((x1 // 3) % 3)) + ((((1 + 2*y0) // ((-2) + ks1)) % ((-2) + ks0))) < ks0")
    tl .device_assert ((((x1 %3 ))+(((1 +2 *y0 )%((-2 )+ks1 )))<ks1 )|~(xmask &ymask ),"index out of bounds: ((x1 % 3)) + (((1 + 2*y0) % ((-2) + ks1))) < ks1")
    tmp6 =tl .load (in_ptr0 +(ks1 *(((x1 //3 )%3 ))+ks1 *((((1 +2 *y0 )//((-2 )+ks1 ))%((-2 )+ks0 )))+ks0 *ks1 *(x1 //9 )+((x1 %3 ))+(((1 +2 *y0 )%((-2 )+ks1 )))),xmask &ymask ,eviction_policy ='evict_last')
    tl .device_assert (((((x1 //3 )%3 ))+((((2 +2 *y0 )//((-2 )+ks1 ))%((-2 )+ks0 )))<ks0 )|~(xmask &ymask ),"index out of bounds: (((x1 // 3) % 3)) + ((((2 + 2*y0) // ((-2) + ks1)) % ((-2) + ks0))) < ks0")
    tl .device_assert ((((x1 %3 ))+(((2 +2 *y0 )%((-2 )+ks1 )))<ks1 )|~(xmask &ymask ),"index out of bounds: ((x1 % 3)) + (((2 + 2*y0) % ((-2) + ks1))) < ks1")
    tmp11 =tl .load (in_ptr0 +(ks1 *(((x1 //3 )%3 ))+ks1 *((((2 +2 *y0 )//((-2 )+ks1 ))%((-2 )+ks0 )))+ks0 *ks1 *(x1 //9 )+((x1 %3 ))+(((2 +2 *y0 )%((-2 )+ks1 )))),xmask &ymask ,eviction_policy ='evict_last')
    tmp3 =tmp2 *tmp2 
    tmp7 =tmp6 *tmp6 
    tmp8 =tmp7 +tmp3 
    tmp12 =tmp11 *tmp11 
    tmp13 =tmp12 +tmp8 
    tmp14 =0.3333333333333333 
    tmp15 =tmp13 *tmp14 
    tmp16 =tl .full ([1 ,1 ],0 ,tl .int32 )
    tmp17 =tmp16 <tmp15 
    tmp18 =tmp17 .to (tl .int8 )
    tmp19 =tmp15 <tmp16 
    tmp20 =tmp19 .to (tl .int8 )
    tmp21 =tmp18 -tmp20 
    tmp22 =tmp21 .to (tmp15 .dtype )
    tmp23 =tl_math .abs (tmp15 )
    tmp24 =triton_helpers .maximum (tmp16 ,tmp23 )
    tmp25 =tmp22 *tmp24 
    tmp26 =3.0 
    tmp27 =tmp25 *tmp26 
    tmp28 =libdevice .sqrt (tmp27 )
    tl .store (out_ptr0 +(y0 +x1 *((3 +ks0 *ks1 )//2 )+((-1 )*ks0 *x1 )+((-1 )*ks1 *x1 )),tmp28 ,xmask &ymask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused_smooth_l1_loss_zeros_like_1 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =7 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    _tmp18 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_1 =r0_index 
        tmp0 =r0_1 +x0 *(triton_helpers .div_floor_integer (6 +((-9 )*ks0 *ks1 )+((-9 )*ks0 *ks2 )+9 *ks0 *((3 +ks1 *ks2 )//2 ),7 ))
        tmp1 =((-9 )*ks0 *ks1 )+((-9 )*ks0 *ks2 )+9 *ks0 *((3 +ks1 *ks2 )//2 )
        tmp2 =tmp0 <tmp1 
        tmp3 =tl .load (in_ptr0 +(r0_1 +x0 *(triton_helpers .div_floor_integer (6 +((-9 )*ks0 *ks1 )+((-9 )*ks0 *ks2 )+9 *ks0 *((3 +ks1 *ks2 )//2 ),7 ))),r0_mask &tmp2 &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp4 =0.0 
        tmp5 =tmp3 -tmp4 
        tmp6 =tl_math .abs (tmp5 )
        tmp7 =1.0 
        tmp8 =tmp6 <tmp7 
        tmp9 =tmp6 *tmp6 
        tmp10 =0.5 
        tmp11 =tmp9 *tmp10 
        tmp12 =tmp11 *tmp7 
        tmp13 =tmp6 -tmp10 
        tmp14 =tl .where (tmp8 ,tmp12 ,tmp13 )
        tmp15 =tl .full (tmp14 .shape ,0 ,tmp14 .dtype )
        tmp16 =tl .where (tmp2 ,tmp14 ,tmp15 )
        tmp17 =tl .broadcast_to (tmp16 ,[XBLOCK ,R0_BLOCK ])
        tmp19 =_tmp18 +tmp17 
        _tmp18 =tl .where (r0_mask &xmask ,tmp19 ,_tmp18 )
    tmp18 =tl .sum (_tmp18 ,1 )[:,None ]
    tl .store (out_ptr0 +(x0 ),tmp18 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused_smooth_l1_loss_zeros_like_2 (in_out_ptr0 ,in_ptr0 ,ks0 ,ks1 ,ks2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    r0_numel =7 
    R0_BLOCK :tl .constexpr =8 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    r0_mask =r0_index <r0_numel 
    r0_0 =r0_index 
    tmp0 =tl .load (in_ptr0 +(r0_0 ),r0_mask ,other =0.0 )
    tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
    tmp3 =tl .where (r0_mask ,tmp1 ,0 )
    tmp4 =tl .sum (tmp3 ,1 )[:,None ]
    tmp5 =((-9 )*ks0 *ks1 )+((-9 )*ks0 *ks2 )+9 *ks0 *((3 +ks1 *ks2 )//2 )
    tmp6 =tmp5 .to (tl .float32 )
    tmp7 =tmp4 /tmp6 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp7 ,None )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,9 *s0 ,((-1 )*s1 )+((-1 )*s2 )+((3 +s1 *s2 )//2 )),(((-9 )*s0 *s1 )+((-9 )*s0 *s2 )+9 *s0 *((3 +s1 *s2 )//2 ),((-1 )*s1 )+((-1 )*s2 )+((3 +s1 *s2 )//2 ),1 ),torch .float32 )

        triton_poi_fused_abs_mul_pow_relu_sign_0_ynumel =((-1 )*s1 )+((-1 )*s2 )+((3 +s1 *s2 )//2 )
        triton_poi_fused_abs_mul_pow_relu_sign_0_xnumel =9 *s0 
        get_raw_stream (0 )
        triton_poi_fused_abs_mul_pow_relu_sign_0 [grid (triton_poi_fused_abs_mul_pow_relu_sign_0_ynumel ,triton_poi_fused_abs_mul_pow_relu_sign_0_xnumel )](arg3_1 ,buf0 ,64 ,64 ,1921 ,27 ,XBLOCK =1 ,YBLOCK =512 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        buf1 =empty_strided_cuda ((7 ,),(1 ,),torch .float32 )

        (6 +((-9 )*s0 *s1 )+((-9 )*s0 *s2 )+9 *s0 *((3 +s1 *s2 )//2 ))//7 
        get_raw_stream (0 )
        triton_red_fused_smooth_l1_loss_zeros_like_1 [grid (7 )](buf0 ,buf1 ,3 ,64 ,64 ,7 ,7410 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del buf0 
        buf2 =empty_strided_cuda ((),(),torch .float32 )
        buf3 =buf2 ;del buf2 

        get_raw_stream (0 )
        triton_per_fused_smooth_l1_loss_zeros_like_2 [grid (1 )](buf3 ,buf1 ,3 ,64 ,64 ,1 ,7 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        del buf1 
    return (buf3 ,)

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
