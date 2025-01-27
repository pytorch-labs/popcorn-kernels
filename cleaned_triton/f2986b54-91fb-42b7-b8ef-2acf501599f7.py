
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
def triton_poi_fused__to_copy_0 (out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =256 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =0.0 
    tl .store (out_ptr0 +(x0 ),tmp0 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__to_copy_1 (out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =512 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =0.0 
    tl .store (out_ptr0 +(x0 ),tmp0 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_add_addmm_tanh_2 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =256 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_out_ptr0 +(x0 ),xmask )
    tmp1 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp3 =tl .load (in_ptr1 +(x0 ),xmask )
    tmp4 =tl .load (in_ptr2 +(x0 ),xmask )
    tmp2 =tmp0 +tmp1 
    tmp5 =tmp3 +tmp4 
    tmp6 =tmp2 +tmp5 
    tmp7 =libdevice .tanh (tmp6 )
    tl .store (in_out_ptr0 +(x0 ),tmp7 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_add_addmm_tanh_3 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =512 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_out_ptr0 +(x0 ),xmask )
    tmp1 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp3 =tl .load (in_ptr1 +(x0 ),xmask )
    tmp4 =tl .load (in_ptr2 +(x0 ),xmask )
    tmp2 =tmp0 +tmp1 
    tmp5 =tmp3 +tmp4 
    tmp6 =tmp2 +tmp5 
    tmp7 =libdevice .tanh (tmp6 )
    tl .store (in_out_ptr0 +(x0 ),tmp7 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_4 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,in_ptr3 ,in_ptr4 ,out_ptr1 ,out_ptr4 ,out_ptr5 ,load_seed_offset ,xnumel ,r0_numel ):
    XBLOCK :tl .constexpr =1 
    R0_BLOCK :tl .constexpr =512 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    tl .full ([1 ],xoffset ,tl .int32 )
    tl .full ([R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[:]
    tl .full ([R0_BLOCK ],True ,tl .int1 )
    r0_0 =r0_index 
    tmp5 =tl .load (in_ptr1 +(r0_0 ),None )
    tmp7 =tl .load (in_out_ptr0 +(r0_0 ),None )
    tmp8 =tl .load (in_ptr2 +(r0_0 ),None )
    tmp34 =tl .load (in_ptr3 +(r0_0 ),None )
    tmp36 =tl .load (in_ptr4 +(r0_0 ),None )
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =r0_0 
    tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tmp3 =0.1 
    tmp4 =tmp2 >tmp3 
    tmp6 =tmp4 .to (tl .float32 )
    tmp9 =tmp7 +tmp8 
    tmp10 =tmp6 *tmp9 
    tmp11 =1.1111111111111112 
    tmp12 =tmp10 *tmp11 
    tmp13 =tmp5 +tmp12 
    tmp14 =tl .broadcast_to (tmp13 ,[R0_BLOCK ])
    tmp16 =tl .broadcast_to (tmp14 ,[R0_BLOCK ])
    tmp18 =triton_helpers .promote_to_tensor (tl .sum (tmp16 ,0 ))
    tmp19 =tl .full ([1 ],512 ,tl .int32 )
    tmp20 =tmp19 .to (tl .float32 )
    tmp21 =tmp18 /tmp20 
    tmp22 =tmp14 -tmp21 
    tmp23 =tmp22 *tmp22 
    tmp24 =tl .broadcast_to (tmp23 ,[R0_BLOCK ])
    tmp26 =triton_helpers .promote_to_tensor (tl .sum (tmp24 ,0 ))
    tmp27 =tmp13 -tmp21 
    tmp28 =512.0 
    tmp29 =tmp26 /tmp28 
    tmp30 =1e-05 
    tmp31 =tmp29 +tmp30 
    tmp32 =libdevice .rsqrt (tmp31 )
    tmp33 =tmp27 *tmp32 
    tmp35 =tmp33 *tmp34 
    tmp37 =tmp35 +tmp36 
    tmp38 =0.001953125 
    tmp39 =tmp32 *tmp38 
    tl .store (out_ptr1 +(tl .broadcast_to (r0_0 ,[R0_BLOCK ])),tmp4 ,None )
    tl .store (in_out_ptr0 +(tl .broadcast_to (r0_0 ,[R0_BLOCK ])),tmp33 ,None )
    tl .store (out_ptr4 +(tl .broadcast_to (r0_0 ,[R0_BLOCK ])),tmp37 ,None )
    tl .store (out_ptr5 +(tl .full ([1 ],0 ,tl .int32 )),tmp39 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_native_dropout_relu_threshold_backward_5 (in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr1 ,out_ptr2 ,out_ptr3 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =2048 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp6 =tl .load (in_ptr1 +(x0 ),xmask )
    tmp7 =tl .load (in_ptr2 +(x0 ),xmask )
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =x0 
    tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tmp3 =0.1 
    tmp4 =tmp2 >tmp3 
    tmp5 =tmp4 .to (tl .float32 )
    tmp8 =tmp6 +tmp7 
    tmp9 =tl .full ([1 ],0 ,tl .int32 )
    tmp10 =triton_helpers .maximum (tmp9 ,tmp8 )
    tmp11 =tmp5 *tmp10 
    tmp12 =1.1111111111111112 
    tmp13 =tmp11 *tmp12 
    tmp14 =0.0 
    tmp15 =tmp10 <=tmp14 
    tl .store (out_ptr1 +(x0 ),tmp4 ,xmask )
    tl .store (out_ptr2 +(x0 ),tmp13 ,xmask )
    tl .store (out_ptr3 +(x0 ),tmp15 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_native_dropout_relu_threshold_backward_6 (in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr1 ,out_ptr2 ,out_ptr3 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =2048 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp6 =tl .load (in_ptr1 +(x0 ),xmask )
    tmp7 =tl .load (in_ptr2 +(x0 ),xmask )
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =x0 
    tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tmp3 =0.1 
    tmp4 =tmp2 >tmp3 
    tmp5 =tmp4 .to (tl .float32 )
    tmp8 =tmp6 +tmp7 
    tmp9 =tl .full ([1 ],0 ,tl .int32 )
    tmp10 =triton_helpers .maximum (tmp9 ,tmp8 )
    tmp11 =tmp5 *tmp10 
    tmp12 =1.1111111111111112 
    tmp13 =tmp11 *tmp12 
    tmp14 =0.0 
    tmp15 =tmp10 <=tmp14 
    tl .store (out_ptr1 +(x0 ),tmp4 ,xmask )
    tl .store (out_ptr2 +(x0 ),tmp13 ,xmask )
    tl .store (out_ptr3 +(x0 ),tmp15 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_7 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr1 ,out_ptr4 ,load_seed_offset ,xnumel ,r0_numel ):
    XBLOCK :tl .constexpr =1 
    R0_BLOCK :tl .constexpr =512 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    tl .full ([1 ],xoffset ,tl .int32 )
    tl .full ([R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[:]
    tl .full ([R0_BLOCK ],True ,tl .int1 )
    r0_0 =r0_index 
    tmp5 =tl .load (in_ptr1 +(r0_0 ),None )
    tmp7 =tl .load (in_out_ptr0 +(r0_0 ),None )
    tmp8 =tl .load (in_ptr2 +(r0_0 ),None )
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =r0_0 
    tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tmp3 =0.1 
    tmp4 =tmp2 >tmp3 
    tmp6 =tmp4 .to (tl .float32 )
    tmp9 =tmp7 +tmp8 
    tmp10 =tmp6 *tmp9 
    tmp11 =1.1111111111111112 
    tmp12 =tmp10 *tmp11 
    tmp13 =tmp5 +tmp12 
    tmp14 =tl .broadcast_to (tmp13 ,[R0_BLOCK ])
    tmp16 =tl .broadcast_to (tmp14 ,[R0_BLOCK ])
    tmp18 =triton_helpers .promote_to_tensor (tl .sum (tmp16 ,0 ))
    tmp19 =tl .full ([1 ],512 ,tl .int32 )
    tmp20 =tmp19 .to (tl .float32 )
    tmp21 =tmp18 /tmp20 
    tmp22 =tmp14 -tmp21 
    tmp23 =tmp22 *tmp22 
    tmp24 =tl .broadcast_to (tmp23 ,[R0_BLOCK ])
    tmp26 =triton_helpers .promote_to_tensor (tl .sum (tmp24 ,0 ))
    tmp27 =tmp13 -tmp21 
    tmp28 =512.0 
    tmp29 =tmp26 /tmp28 
    tmp30 =1e-05 
    tmp31 =tmp29 +tmp30 
    tmp32 =libdevice .rsqrt (tmp31 )
    tmp33 =tmp27 *tmp32 
    tmp34 =0.001953125 
    tmp35 =tmp32 *tmp34 
    tl .store (out_ptr1 +(tl .broadcast_to (r0_0 ,[R0_BLOCK ])),tmp4 ,None )
    tl .store (in_out_ptr0 +(tl .broadcast_to (r0_0 ,[R0_BLOCK ])),tmp33 ,None )
    tl .store (out_ptr4 +(tl .full ([1 ],0 ,tl .int32 )),tmp35 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_constant_pad_nd_8 (in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =4626 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x2 =xindex //1542 
    x1 =((xindex //514 )%3 )
    x0 =(xindex %514 )
    x5 =xindex 
    tmp0 =(-1 )+x2 
    tmp1 =tl .full ([1 ],0 ,tl .int64 )
    tmp2 =tmp0 >=tmp1 
    tmp3 =tl .full ([1 ],1 ,tl .int64 )
    tmp4 =tmp0 <tmp3 
    tmp5 =(-1 )+x1 
    tmp6 =tmp5 >=tmp1 
    tmp7 =tmp5 <tmp3 
    tmp8 =(-1 )+x0 
    tmp9 =tmp8 >=tmp1 
    tmp10 =tl .full ([1 ],512 ,tl .int64 )
    tmp11 =tmp8 <tmp10 
    tmp12 =tmp2 &tmp4 
    tmp13 =tmp12 &tmp6 
    tmp14 =tmp13 &tmp7 
    tmp15 =tmp14 &tmp9 
    tmp16 =tmp15 &tmp11 
    tmp17 =tl .load (in_ptr0 +((-1 )+x0 ),tmp16 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp18 =tl .load (in_ptr1 +((-1 )+x0 ),tmp16 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp19 =tmp17 *tmp18 
    tmp20 =tl .load (in_ptr2 +((-1 )+x0 ),tmp16 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp21 =tmp19 +tmp20 
    tmp22 =tl .full (tmp21 .shape ,0.0 ,tmp21 .dtype )
    tmp23 =tl .where (tmp16 ,tmp21 ,tmp22 )
    tl .store (out_ptr0 +(x5 ),tmp23 ,xmask )

def call (args ):
    primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ,primals_8 ,primals_9 ,primals_10 ,primals_11 ,primals_12 ,primals_13 ,primals_14 ,primals_15 ,primals_16 ,primals_17 ,primals_18 ,primals_19 ,primals_20 ,primals_21 ,primals_22 ,primals_23 ,primals_24 ,primals_25 ,primals_26 ,primals_27 ,primals_28 ,primals_29 ,primals_30 ,primals_31 ,primals_32 ,primals_33 ,primals_34 ,primals_35 ,primals_36 ,primals_37 ,primals_38 ,primals_39 ,primals_40 ,primals_41 ,primals_42 ,primals_43 ,primals_44 ,primals_45 =args 
    args .clear ()
    assert_size_stride (primals_1 ,(1 ,10 ,128 ),(1280 ,128 ,1 ))
    assert_size_stride (primals_2 ,(256 ,128 ),(128 ,1 ))
    assert_size_stride (primals_3 ,(256 ,256 ),(256 ,1 ))
    assert_size_stride (primals_4 ,(256 ,),(1 ,))
    assert_size_stride (primals_5 ,(256 ,),(1 ,))
    assert_size_stride (primals_6 ,(512 ,256 ),(256 ,1 ))
    assert_size_stride (primals_7 ,(512 ,512 ),(512 ,1 ))
    assert_size_stride (primals_8 ,(512 ,),(1 ,))
    assert_size_stride (primals_9 ,(512 ,),(1 ,))
    assert_size_stride (primals_10 ,(1536 ,),(1 ,))
    assert_size_stride (primals_11 ,(1536 ,512 ),(512 ,1 ))
    assert_size_stride (primals_12 ,(512 ,512 ),(512 ,1 ))
    assert_size_stride (primals_13 ,(512 ,),(1 ,))
    assert_size_stride (primals_14 ,(512 ,),(1 ,))
    assert_size_stride (primals_15 ,(512 ,),(1 ,))
    assert_size_stride (primals_16 ,(2048 ,512 ),(512 ,1 ))
    assert_size_stride (primals_17 ,(2048 ,),(1 ,))
    assert_size_stride (primals_18 ,(512 ,2048 ),(2048 ,1 ))
    assert_size_stride (primals_19 ,(512 ,),(1 ,))
    assert_size_stride (primals_20 ,(512 ,),(1 ,))
    assert_size_stride (primals_21 ,(512 ,),(1 ,))
    assert_size_stride (primals_22 ,(1536 ,),(1 ,))
    assert_size_stride (primals_23 ,(1536 ,512 ),(512 ,1 ))
    assert_size_stride (primals_24 ,(512 ,512 ),(512 ,1 ))
    assert_size_stride (primals_25 ,(512 ,),(1 ,))
    assert_size_stride (primals_26 ,(512 ,),(1 ,))
    assert_size_stride (primals_27 ,(512 ,),(1 ,))
    assert_size_stride (primals_28 ,(2048 ,512 ),(512 ,1 ))
    assert_size_stride (primals_29 ,(2048 ,),(1 ,))
    assert_size_stride (primals_30 ,(512 ,2048 ),(2048 ,1 ))
    assert_size_stride (primals_31 ,(512 ,),(1 ,))
    assert_size_stride (primals_32 ,(512 ,),(1 ,))
    assert_size_stride (primals_33 ,(512 ,),(1 ,))
    assert_size_stride (primals_34 ,(1536 ,),(1 ,))
    assert_size_stride (primals_35 ,(1536 ,512 ),(512 ,1 ))
    assert_size_stride (primals_36 ,(512 ,512 ),(512 ,1 ))
    assert_size_stride (primals_37 ,(512 ,),(1 ,))
    assert_size_stride (primals_38 ,(512 ,),(1 ,))
    assert_size_stride (primals_39 ,(512 ,),(1 ,))
    assert_size_stride (primals_40 ,(2048 ,512 ),(512 ,1 ))
    assert_size_stride (primals_41 ,(2048 ,),(1 ,))
    assert_size_stride (primals_42 ,(512 ,2048 ),(2048 ,1 ))
    assert_size_stride (primals_43 ,(512 ,),(1 ,))
    assert_size_stride (primals_44 ,(512 ,),(1 ,))
    assert_size_stride (primals_45 ,(512 ,),(1 ,))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,256 ),(256 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused__to_copy_0 [grid (256 )](buf0 ,256 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        buf1 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused__to_copy_1 [grid (512 )](buf1 ,512 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        buf2 =empty_strided_cuda ((1 ,256 ),(256 ,1 ),torch .float32 )

        extern_kernels .mm (buf0 ,reinterpret_tensor (primals_3 ,(256 ,256 ),(1 ,256 ),0 ),out =buf2 )
        buf3 =empty_strided_cuda ((1 ,256 ),(256 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),0 ),reinterpret_tensor (primals_2 ,(128 ,256 ),(1 ,128 ),0 ),out =buf3 )
        buf4 =buf2 ;del buf2 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_2 [grid (256 )](buf4 ,primals_5 ,buf3 ,primals_4 ,256 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        buf5 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (buf1 ,reinterpret_tensor (primals_7 ,(512 ,512 ),(1 ,512 ),0 ),out =buf5 )
        buf6 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (buf4 ,reinterpret_tensor (primals_6 ,(256 ,512 ),(1 ,256 ),0 ),out =buf6 )
        buf7 =buf5 ;del buf5 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_3 [grid (512 )](buf7 ,primals_9 ,buf6 ,primals_8 ,512 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        buf8 =buf3 ;del buf3 

        extern_kernels .mm (buf4 ,reinterpret_tensor (primals_3 ,(256 ,256 ),(1 ,256 ),0 ),out =buf8 )
        buf9 =empty_strided_cuda ((1 ,256 ),(256 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),128 ),reinterpret_tensor (primals_2 ,(128 ,256 ),(1 ,128 ),0 ),out =buf9 )
        buf10 =buf8 ;del buf8 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_2 [grid (256 )](buf10 ,primals_5 ,buf9 ,primals_4 ,256 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        buf11 =buf6 ;del buf6 

        extern_kernels .mm (buf7 ,reinterpret_tensor (primals_7 ,(512 ,512 ),(1 ,512 ),0 ),out =buf11 )
        buf12 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (buf10 ,reinterpret_tensor (primals_6 ,(256 ,512 ),(1 ,256 ),0 ),out =buf12 )
        buf13 =buf11 ;del buf11 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_3 [grid (512 )](buf13 ,primals_9 ,buf12 ,primals_8 ,512 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        buf14 =buf9 ;del buf9 

        extern_kernels .mm (buf10 ,reinterpret_tensor (primals_3 ,(256 ,256 ),(1 ,256 ),0 ),out =buf14 )
        buf15 =empty_strided_cuda ((1 ,256 ),(256 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),256 ),reinterpret_tensor (primals_2 ,(128 ,256 ),(1 ,128 ),0 ),out =buf15 )
        buf16 =buf14 ;del buf14 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_2 [grid (256 )](buf16 ,primals_5 ,buf15 ,primals_4 ,256 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        buf17 =buf12 ;del buf12 

        extern_kernels .mm (buf13 ,reinterpret_tensor (primals_7 ,(512 ,512 ),(1 ,512 ),0 ),out =buf17 )
        buf18 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (buf16 ,reinterpret_tensor (primals_6 ,(256 ,512 ),(1 ,256 ),0 ),out =buf18 )
        buf19 =buf17 ;del buf17 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_3 [grid (512 )](buf19 ,primals_9 ,buf18 ,primals_8 ,512 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        buf20 =buf15 ;del buf15 

        extern_kernels .mm (buf16 ,reinterpret_tensor (primals_3 ,(256 ,256 ),(1 ,256 ),0 ),out =buf20 )
        buf21 =empty_strided_cuda ((1 ,256 ),(256 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),384 ),reinterpret_tensor (primals_2 ,(128 ,256 ),(1 ,128 ),0 ),out =buf21 )
        buf22 =buf20 ;del buf20 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_2 [grid (256 )](buf22 ,primals_5 ,buf21 ,primals_4 ,256 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        buf23 =buf18 ;del buf18 

        extern_kernels .mm (buf19 ,reinterpret_tensor (primals_7 ,(512 ,512 ),(1 ,512 ),0 ),out =buf23 )
        buf24 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (buf22 ,reinterpret_tensor (primals_6 ,(256 ,512 ),(1 ,256 ),0 ),out =buf24 )
        buf25 =buf23 ;del buf23 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_3 [grid (512 )](buf25 ,primals_9 ,buf24 ,primals_8 ,512 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        buf26 =buf21 ;del buf21 

        extern_kernels .mm (buf22 ,reinterpret_tensor (primals_3 ,(256 ,256 ),(1 ,256 ),0 ),out =buf26 )
        buf27 =empty_strided_cuda ((1 ,256 ),(256 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),512 ),reinterpret_tensor (primals_2 ,(128 ,256 ),(1 ,128 ),0 ),out =buf27 )
        buf28 =buf26 ;del buf26 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_2 [grid (256 )](buf28 ,primals_5 ,buf27 ,primals_4 ,256 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        buf29 =buf24 ;del buf24 

        extern_kernels .mm (buf25 ,reinterpret_tensor (primals_7 ,(512 ,512 ),(1 ,512 ),0 ),out =buf29 )
        buf30 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (buf28 ,reinterpret_tensor (primals_6 ,(256 ,512 ),(1 ,256 ),0 ),out =buf30 )
        buf31 =buf29 ;del buf29 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_3 [grid (512 )](buf31 ,primals_9 ,buf30 ,primals_8 ,512 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        buf32 =buf27 ;del buf27 

        extern_kernels .mm (buf28 ,reinterpret_tensor (primals_3 ,(256 ,256 ),(1 ,256 ),0 ),out =buf32 )
        buf33 =empty_strided_cuda ((1 ,256 ),(256 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),640 ),reinterpret_tensor (primals_2 ,(128 ,256 ),(1 ,128 ),0 ),out =buf33 )
        buf34 =buf32 ;del buf32 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_2 [grid (256 )](buf34 ,primals_5 ,buf33 ,primals_4 ,256 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        buf35 =buf30 ;del buf30 

        extern_kernels .mm (buf31 ,reinterpret_tensor (primals_7 ,(512 ,512 ),(1 ,512 ),0 ),out =buf35 )
        buf36 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (buf34 ,reinterpret_tensor (primals_6 ,(256 ,512 ),(1 ,256 ),0 ),out =buf36 )
        buf37 =buf35 ;del buf35 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_3 [grid (512 )](buf37 ,primals_9 ,buf36 ,primals_8 ,512 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        buf38 =buf33 ;del buf33 

        extern_kernels .mm (buf34 ,reinterpret_tensor (primals_3 ,(256 ,256 ),(1 ,256 ),0 ),out =buf38 )
        buf39 =empty_strided_cuda ((1 ,256 ),(256 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),768 ),reinterpret_tensor (primals_2 ,(128 ,256 ),(1 ,128 ),0 ),out =buf39 )
        buf40 =buf38 ;del buf38 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_2 [grid (256 )](buf40 ,primals_5 ,buf39 ,primals_4 ,256 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        buf41 =buf36 ;del buf36 

        extern_kernels .mm (buf37 ,reinterpret_tensor (primals_7 ,(512 ,512 ),(1 ,512 ),0 ),out =buf41 )
        buf42 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (buf40 ,reinterpret_tensor (primals_6 ,(256 ,512 ),(1 ,256 ),0 ),out =buf42 )
        buf43 =buf41 ;del buf41 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_3 [grid (512 )](buf43 ,primals_9 ,buf42 ,primals_8 ,512 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        buf44 =buf39 ;del buf39 

        extern_kernels .mm (buf40 ,reinterpret_tensor (primals_3 ,(256 ,256 ),(1 ,256 ),0 ),out =buf44 )
        buf45 =empty_strided_cuda ((1 ,256 ),(256 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),896 ),reinterpret_tensor (primals_2 ,(128 ,256 ),(1 ,128 ),0 ),out =buf45 )
        buf46 =buf44 ;del buf44 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_2 [grid (256 )](buf46 ,primals_5 ,buf45 ,primals_4 ,256 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        buf47 =buf42 ;del buf42 

        extern_kernels .mm (buf43 ,reinterpret_tensor (primals_7 ,(512 ,512 ),(1 ,512 ),0 ),out =buf47 )
        buf48 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (buf46 ,reinterpret_tensor (primals_6 ,(256 ,512 ),(1 ,256 ),0 ),out =buf48 )
        buf49 =buf47 ;del buf47 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_3 [grid (512 )](buf49 ,primals_9 ,buf48 ,primals_8 ,512 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        buf50 =buf45 ;del buf45 

        extern_kernels .mm (buf46 ,reinterpret_tensor (primals_3 ,(256 ,256 ),(1 ,256 ),0 ),out =buf50 )
        buf51 =empty_strided_cuda ((1 ,256 ),(256 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),1024 ),reinterpret_tensor (primals_2 ,(128 ,256 ),(1 ,128 ),0 ),out =buf51 )
        buf52 =buf50 ;del buf50 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_2 [grid (256 )](buf52 ,primals_5 ,buf51 ,primals_4 ,256 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        buf53 =buf48 ;del buf48 

        extern_kernels .mm (buf49 ,reinterpret_tensor (primals_7 ,(512 ,512 ),(1 ,512 ),0 ),out =buf53 )
        buf54 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (buf52 ,reinterpret_tensor (primals_6 ,(256 ,512 ),(1 ,256 ),0 ),out =buf54 )
        buf55 =buf53 ;del buf53 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_3 [grid (512 )](buf55 ,primals_9 ,buf54 ,primals_8 ,512 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        buf56 =buf51 ;del buf51 

        extern_kernels .mm (buf52 ,reinterpret_tensor (primals_3 ,(256 ,256 ),(1 ,256 ),0 ),out =buf56 )
        buf57 =empty_strided_cuda ((1 ,256 ),(256 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),1152 ),reinterpret_tensor (primals_2 ,(128 ,256 ),(1 ,128 ),0 ),out =buf57 )
        del primals_2 
        buf58 =buf56 ;del buf56 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_2 [grid (256 )](buf58 ,primals_5 ,buf57 ,primals_4 ,256 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf57 
        del primals_4 
        del primals_5 
        buf59 =buf54 ;del buf54 

        extern_kernels .mm (buf55 ,reinterpret_tensor (primals_7 ,(512 ,512 ),(1 ,512 ),0 ),out =buf59 )
        buf60 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (buf58 ,reinterpret_tensor (primals_6 ,(256 ,512 ),(1 ,256 ),0 ),out =buf60 )
        buf61 =buf59 ;del buf59 

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_3 [grid (512 )](buf61 ,primals_9 ,buf60 ,primals_8 ,512 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del primals_8 
        del primals_9 
        buf62 =empty_strided_cuda ((1 ,1536 ),(1536 ,1 ),torch .float32 )

        extern_kernels .addmm (primals_10 ,buf61 ,reinterpret_tensor (primals_11 ,(512 ,1536 ),(1 ,512 ),0 ),alpha =1 ,beta =1 ,out =buf62 )
        del primals_10 

        buf63 =torch .ops .aten ._scaled_dot_product_efficient_attention .default (reinterpret_tensor (buf62 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),0 ),reinterpret_tensor (buf62 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),512 ),reinterpret_tensor (buf62 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),1024 ),None ,True ,0.1 )
        buf64 =buf63 [0 ]
        buf65 =buf63 [1 ]
        buf66 =buf63 [2 ]
        buf67 =buf63 [3 ]
        del buf63 
        buf68 =buf60 ;del buf60 

        extern_kernels .mm (reinterpret_tensor (buf64 ,(1 ,512 ),(512 ,1 ),0 ),reinterpret_tensor (primals_12 ,(512 ,512 ),(1 ,512 ),0 ),out =buf68 )
        buf69 =empty_strided_cuda ((9 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[9 ],out =buf69 )
        buf71 =empty_strided_cuda ((1 ,1 ,512 ),(512 ,512 ,1 ),torch .bool )
        buf75 =reinterpret_tensor (buf68 ,(1 ,1 ,512 ),(512 ,512 ,1 ),0 );del buf68 
        buf76 =empty_strided_cuda ((1 ,1 ,512 ),(512 ,512 ,1 ),torch .float32 )
        buf149 =empty_strided_cuda ((1 ,1 ,1 ),(1 ,1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_4 [grid (1 )](buf75 ,buf69 ,buf61 ,primals_13 ,primals_14 ,primals_15 ,buf71 ,buf76 ,buf149 ,6 ,1 ,512 ,num_warps =4 ,num_stages =1 )
        del primals_13 
        del primals_15 
        buf77 =empty_strided_cuda ((1 ,2048 ),(2048 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf76 ,(1 ,512 ),(0 ,1 ),0 ),reinterpret_tensor (primals_16 ,(512 ,2048 ),(1 ,512 ),0 ),out =buf77 )
        buf79 =empty_strided_cuda ((1 ,1 ,2048 ),(2048 ,2048 ,1 ),torch .bool )
        buf80 =empty_strided_cuda ((1 ,1 ,2048 ),(2048 ,2048 ,1 ),torch .float32 )
        buf143 =empty_strided_cuda ((1 ,1 ,2048 ),(2048 ,2048 ,1 ),torch .bool )

        get_raw_stream (0 )
        triton_poi_fused_native_dropout_relu_threshold_backward_5 [grid (2048 )](buf69 ,buf77 ,primals_17 ,buf79 ,buf80 ,buf143 ,1 ,2048 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del primals_17 
        buf81 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf80 ,(1 ,2048 ),(0 ,1 ),0 ),reinterpret_tensor (primals_18 ,(2048 ,512 ),(1 ,2048 ),0 ),out =buf81 )
        buf83 =empty_strided_cuda ((1 ,1 ,512 ),(512 ,512 ,1 ),torch .bool )
        buf87 =reinterpret_tensor (buf81 ,(1 ,1 ,512 ),(512 ,512 ,1 ),0 );del buf81 
        buf88 =empty_strided_cuda ((1 ,1 ,512 ),(512 ,512 ,1 ),torch .float32 )
        buf148 =empty_strided_cuda ((1 ,1 ,1 ),(1 ,1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_4 [grid (1 )](buf87 ,buf69 ,buf76 ,primals_19 ,primals_20 ,primals_21 ,buf83 ,buf88 ,buf148 ,6 ,1 ,512 ,num_warps =4 ,num_stages =1 )
        del primals_19 
        del primals_21 
        buf89 =empty_strided_cuda ((1 ,1536 ),(1536 ,1 ),torch .float32 )

        extern_kernels .addmm (primals_22 ,reinterpret_tensor (buf88 ,(1 ,512 ),(0 ,1 ),0 ),reinterpret_tensor (primals_23 ,(512 ,1536 ),(1 ,512 ),0 ),alpha =1 ,beta =1 ,out =buf89 )
        del primals_22 

        buf90 =torch .ops .aten ._scaled_dot_product_efficient_attention .default (reinterpret_tensor (buf89 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),0 ),reinterpret_tensor (buf89 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),512 ),reinterpret_tensor (buf89 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),1024 ),None ,True ,0.1 )
        buf91 =buf90 [0 ]
        buf92 =buf90 [1 ]
        buf93 =buf90 [2 ]
        buf94 =buf90 [3 ]
        del buf90 
        buf95 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf91 ,(1 ,512 ),(512 ,1 ),0 ),reinterpret_tensor (primals_24 ,(512 ,512 ),(1 ,512 ),0 ),out =buf95 )
        buf97 =empty_strided_cuda ((1 ,1 ,512 ),(512 ,512 ,1 ),torch .bool )
        buf101 =reinterpret_tensor (buf95 ,(1 ,1 ,512 ),(512 ,512 ,1 ),0 );del buf95 
        buf102 =empty_strided_cuda ((1 ,1 ,512 ),(512 ,512 ,1 ),torch .float32 )
        buf147 =empty_strided_cuda ((1 ,1 ,1 ),(1 ,1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_4 [grid (1 )](buf101 ,buf69 ,buf88 ,primals_25 ,primals_26 ,primals_27 ,buf97 ,buf102 ,buf147 ,6 ,1 ,512 ,num_warps =4 ,num_stages =1 )
        del primals_25 
        del primals_27 
        buf103 =buf77 ;del buf77 

        extern_kernels .mm (reinterpret_tensor (buf102 ,(1 ,512 ),(0 ,1 ),0 ),reinterpret_tensor (primals_28 ,(512 ,2048 ),(1 ,512 ),0 ),out =buf103 )
        buf105 =empty_strided_cuda ((1 ,1 ,2048 ),(2048 ,2048 ,1 ),torch .bool )
        buf106 =empty_strided_cuda ((1 ,1 ,2048 ),(2048 ,2048 ,1 ),torch .float32 )
        buf142 =empty_strided_cuda ((1 ,1 ,2048 ),(2048 ,2048 ,1 ),torch .bool )

        get_raw_stream (0 )
        triton_poi_fused_native_dropout_relu_threshold_backward_6 [grid (2048 )](buf69 ,buf103 ,primals_29 ,buf105 ,buf106 ,buf142 ,7 ,2048 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del primals_29 
        buf107 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf106 ,(1 ,2048 ),(0 ,1 ),0 ),reinterpret_tensor (primals_30 ,(2048 ,512 ),(1 ,2048 ),0 ),out =buf107 )
        buf109 =empty_strided_cuda ((1 ,1 ,512 ),(512 ,512 ,1 ),torch .bool )
        buf113 =reinterpret_tensor (buf107 ,(1 ,1 ,512 ),(512 ,512 ,1 ),0 );del buf107 
        buf114 =empty_strided_cuda ((1 ,1 ,512 ),(512 ,512 ,1 ),torch .float32 )
        buf146 =empty_strided_cuda ((1 ,1 ,1 ),(1 ,1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_4 [grid (1 )](buf113 ,buf69 ,buf102 ,primals_31 ,primals_32 ,primals_33 ,buf109 ,buf114 ,buf146 ,6 ,1 ,512 ,num_warps =4 ,num_stages =1 )
        del primals_31 
        del primals_33 
        buf115 =empty_strided_cuda ((1 ,1536 ),(1536 ,1 ),torch .float32 )

        extern_kernels .addmm (primals_34 ,reinterpret_tensor (buf114 ,(1 ,512 ),(0 ,1 ),0 ),reinterpret_tensor (primals_35 ,(512 ,1536 ),(1 ,512 ),0 ),alpha =1 ,beta =1 ,out =buf115 )
        del primals_34 

        buf116 =torch .ops .aten ._scaled_dot_product_efficient_attention .default (reinterpret_tensor (buf115 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),0 ),reinterpret_tensor (buf115 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),512 ),reinterpret_tensor (buf115 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),1024 ),None ,True ,0.1 )
        buf117 =buf116 [0 ]
        buf118 =buf116 [1 ]
        buf119 =buf116 [2 ]
        buf120 =buf116 [3 ]
        del buf116 
        buf121 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf117 ,(1 ,512 ),(512 ,1 ),0 ),reinterpret_tensor (primals_36 ,(512 ,512 ),(1 ,512 ),0 ),out =buf121 )
        buf123 =empty_strided_cuda ((1 ,1 ,512 ),(512 ,512 ,1 ),torch .bool )
        buf127 =reinterpret_tensor (buf121 ,(1 ,1 ,512 ),(512 ,512 ,1 ),0 );del buf121 
        buf128 =empty_strided_cuda ((1 ,1 ,512 ),(512 ,512 ,1 ),torch .float32 )
        buf145 =empty_strided_cuda ((1 ,1 ,1 ),(1 ,1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_4 [grid (1 )](buf127 ,buf69 ,buf114 ,primals_37 ,primals_38 ,primals_39 ,buf123 ,buf128 ,buf145 ,6 ,1 ,512 ,num_warps =4 ,num_stages =1 )
        del primals_37 
        del primals_39 
        buf129 =buf103 ;del buf103 

        extern_kernels .mm (reinterpret_tensor (buf128 ,(1 ,512 ),(0 ,1 ),0 ),reinterpret_tensor (primals_40 ,(512 ,2048 ),(1 ,512 ),0 ),out =buf129 )
        buf131 =empty_strided_cuda ((1 ,1 ,2048 ),(2048 ,2048 ,1 ),torch .bool )
        buf132 =empty_strided_cuda ((1 ,1 ,2048 ),(2048 ,2048 ,1 ),torch .float32 )
        buf141 =empty_strided_cuda ((1 ,1 ,2048 ),(2048 ,2048 ,1 ),torch .bool )

        get_raw_stream (0 )
        triton_poi_fused_native_dropout_relu_threshold_backward_6 [grid (2048 )](buf69 ,buf129 ,primals_41 ,buf131 ,buf132 ,buf141 ,7 ,2048 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del buf129 
        del primals_41 
        buf133 =empty_strided_cuda ((1 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf132 ,(1 ,2048 ),(0 ,1 ),0 ),reinterpret_tensor (primals_42 ,(2048 ,512 ),(1 ,2048 ),0 ),out =buf133 )
        buf135 =empty_strided_cuda ((1 ,1 ,512 ),(512 ,512 ,1 ),torch .bool )
        buf139 =reinterpret_tensor (buf133 ,(1 ,1 ,512 ),(512 ,512 ,1 ),0 );del buf133 
        buf144 =empty_strided_cuda ((1 ,1 ,1 ),(1 ,1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_7 [grid (1 )](buf139 ,buf69 ,buf128 ,primals_43 ,buf135 ,buf144 ,8 ,1 ,512 ,num_warps =4 ,num_stages =1 )
        del buf69 
        del primals_43 
        buf140 =empty_strided_cuda ((1 ,1 ,3 ,3 ,514 ),(4626 ,1 ,1542 ,514 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_constant_pad_nd_8 [grid (4626 )](buf139 ,primals_44 ,primals_45 ,buf140 ,4626 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del primals_45 
    return (reinterpret_tensor (buf140 ,(1 ,4626 ),(4626 ,1 ),0 ),primals_14 ,primals_20 ,primals_26 ,primals_32 ,primals_38 ,primals_44 ,buf0 ,buf1 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),0 ),buf4 ,buf7 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),128 ),buf10 ,buf13 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),256 ),buf16 ,buf19 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),384 ),buf22 ,buf25 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),512 ),buf28 ,buf31 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),640 ),buf34 ,buf37 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),768 ),buf40 ,buf43 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),896 ),buf46 ,buf49 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),1024 ),buf52 ,buf55 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),1152 ),buf58 ,buf61 ,reinterpret_tensor (buf62 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),0 ),reinterpret_tensor (buf62 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),512 ),reinterpret_tensor (buf62 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),1024 ),buf64 ,buf65 ,buf66 ,buf67 ,buf71 ,buf75 ,reinterpret_tensor (buf76 ,(1 ,512 ),(512 ,1 ),0 ),buf79 ,reinterpret_tensor (buf80 ,(1 ,2048 ),(2048 ,1 ),0 ),buf83 ,buf87 ,reinterpret_tensor (buf88 ,(1 ,512 ),(512 ,1 ),0 ),reinterpret_tensor (buf89 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),0 ),reinterpret_tensor (buf89 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),512 ),reinterpret_tensor (buf89 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),1024 ),buf91 ,buf92 ,buf93 ,buf94 ,buf97 ,buf101 ,reinterpret_tensor (buf102 ,(1 ,512 ),(512 ,1 ),0 ),buf105 ,reinterpret_tensor (buf106 ,(1 ,2048 ),(2048 ,1 ),0 ),buf109 ,buf113 ,reinterpret_tensor (buf114 ,(1 ,512 ),(512 ,1 ),0 ),reinterpret_tensor (buf115 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),0 ),reinterpret_tensor (buf115 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),512 ),reinterpret_tensor (buf115 ,(1 ,8 ,1 ,64 ),(512 ,64 ,1536 ,1 ),1024 ),buf117 ,buf118 ,buf119 ,buf120 ,buf123 ,buf127 ,reinterpret_tensor (buf128 ,(1 ,512 ),(512 ,1 ),0 ),buf131 ,reinterpret_tensor (buf132 ,(1 ,2048 ),(2048 ,1 ),0 ),buf135 ,buf139 ,buf144 ,primals_42 ,buf141 ,primals_40 ,buf145 ,primals_36 ,primals_35 ,buf146 ,primals_30 ,buf142 ,primals_28 ,buf147 ,primals_24 ,primals_23 ,buf148 ,primals_18 ,buf143 ,primals_16 ,buf149 ,primals_12 ,primals_11 ,primals_6 ,primals_7 ,primals_3 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    primals_1 =rand_strided ((1 ,10 ,128 ),(1280 ,128 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_2 =rand_strided ((256 ,128 ),(128 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_3 =rand_strided ((256 ,256 ),(256 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_4 =rand_strided ((256 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_5 =rand_strided ((256 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_6 =rand_strided ((512 ,256 ),(256 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_7 =rand_strided ((512 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_8 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_9 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_10 =rand_strided ((1536 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_11 =rand_strided ((1536 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_12 =rand_strided ((512 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_13 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_14 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_15 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_16 =rand_strided ((2048 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_17 =rand_strided ((2048 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_18 =rand_strided ((512 ,2048 ),(2048 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_19 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_20 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_21 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_22 =rand_strided ((1536 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_23 =rand_strided ((1536 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_24 =rand_strided ((512 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_25 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_26 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_27 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_28 =rand_strided ((2048 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_29 =rand_strided ((2048 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_30 =rand_strided ((512 ,2048 ),(2048 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_31 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_32 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_33 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_34 =rand_strided ((1536 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_35 =rand_strided ((1536 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_36 =rand_strided ((512 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_37 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_38 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_39 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_40 =rand_strided ((2048 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_41 =rand_strided ((2048 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_42 =rand_strided ((512 ,2048 ),(2048 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_43 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_44 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_45 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ,primals_8 ,primals_9 ,primals_10 ,primals_11 ,primals_12 ,primals_13 ,primals_14 ,primals_15 ,primals_16 ,primals_17 ,primals_18 ,primals_19 ,primals_20 ,primals_21 ,primals_22 ,primals_23 ,primals_24 ,primals_25 ,primals_26 ,primals_27 ,primals_28 ,primals_29 ,primals_30 ,primals_31 ,primals_32 ,primals_33 ,primals_34 ,primals_35 ,primals_36 ,primals_37 ,primals_38 ,primals_39 ,primals_40 ,primals_41 ,primals_42 ,primals_43 ,primals_44 ,primals_45 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
