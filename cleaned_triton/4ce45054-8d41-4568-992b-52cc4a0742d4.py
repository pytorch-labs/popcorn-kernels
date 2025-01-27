
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
def triton_poi_fused_clone_0 (in_ptr0 ,in_ptr1 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =15360 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %512 )
    x1 =((xindex //512 )%10 )
    x2 =xindex //5120 
    x3 =xindex 
    tmp0 =tl .load (in_ptr0 +(x0 +512 *x2 +1536 *x1 ),xmask )
    tmp1 =tl .load (in_ptr1 +(x0 +512 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp2 =tmp0 +tmp1 
    tl .store (out_ptr0 +(x3 ),tmp2 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_1 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,in_ptr3 ,in_ptr4 ,out_ptr1 ,out_ptr4 ,out_ptr5 ,load_seed_offset ,xnumel ,r0_numel ):
    XBLOCK :tl .constexpr =1 
    R0_BLOCK :tl .constexpr =512 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =tl .full ([1 ],xoffset ,tl .int32 )
    tl .full ([R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[:]
    tl .full ([R0_BLOCK ],True ,tl .int1 )
    r0_1 =r0_index 
    x0 =xindex 
    tmp5 =tl .load (in_ptr1 +(r0_1 +512 *x0 ),None )
    tmp7 =tl .load (in_out_ptr0 +(r0_1 +512 *x0 ),None )
    tmp8 =tl .load (in_ptr2 +(r0_1 ),None ,eviction_policy ='evict_last')
    tmp34 =tl .load (in_ptr3 +(r0_1 ),None ,eviction_policy ='evict_last')
    tmp36 =tl .load (in_ptr4 +(r0_1 ),None ,eviction_policy ='evict_last')
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =r0_1 +512 *x0 
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
    tl .store (out_ptr1 +(r0_1 +512 *x0 ),tmp4 ,None )
    tl .store (in_out_ptr0 +(r0_1 +512 *x0 ),tmp33 ,None )
    tl .store (out_ptr4 +(r0_1 +512 *x0 ),tmp37 ,None )
    tl .store (out_ptr5 +(x0 ),tmp39 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_clone_2 (in_ptr0 ,in_ptr1 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =10240 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %512 )
    x1 =((xindex //512 )%10 )
    x2 =xindex //5120 
    x3 =xindex 
    tmp0 =tl .load (in_ptr0 +(x0 +512 *x2 +1024 *x1 ),xmask )
    tmp1 =tl .load (in_ptr1 +(512 +x0 +512 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp2 =tmp0 +tmp1 
    tl .store (out_ptr0 +(x3 ),tmp2 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_3 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,in_ptr3 ,in_ptr4 ,out_ptr1 ,out_ptr4 ,out_ptr5 ,load_seed_offset ,xnumel ,r0_numel ):
    XBLOCK :tl .constexpr =1 
    R0_BLOCK :tl .constexpr =512 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =tl .full ([1 ],xoffset ,tl .int32 )
    tl .full ([R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[:]
    tl .full ([R0_BLOCK ],True ,tl .int1 )
    r0_1 =r0_index 
    x0 =xindex 
    tmp5 =tl .load (in_ptr1 +(r0_1 +512 *x0 ),None )
    tmp7 =tl .load (in_out_ptr0 +(r0_1 +512 *x0 ),None )
    tmp8 =tl .load (in_ptr2 +(r0_1 ),None ,eviction_policy ='evict_last')
    tmp34 =tl .load (in_ptr3 +(r0_1 ),None ,eviction_policy ='evict_last')
    tmp36 =tl .load (in_ptr4 +(r0_1 ),None ,eviction_policy ='evict_last')
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =r0_1 +512 *x0 
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
    tl .store (out_ptr1 +(r0_1 +512 *x0 ),tmp4 ,None )
    tl .store (in_out_ptr0 +(r0_1 +512 *x0 ),tmp33 ,None )
    tl .store (out_ptr4 +(r0_1 +512 *x0 ),tmp37 ,None )
    tl .store (out_ptr5 +(x0 ),tmp39 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_native_dropout_relu_threshold_backward_4 (in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr1 ,out_ptr2 ,out_ptr3 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    tl .full ([XBLOCK ],True ,tl .int1 )
    x0 =xindex 
    x1 =(xindex %2048 )
    tmp6 =tl .load (in_ptr1 +(x0 ),None )
    tmp7 =tl .load (in_ptr2 +(x1 ),None ,eviction_policy ='evict_last')
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
    tl .store (out_ptr1 +(x0 ),tmp4 ,None )
    tl .store (out_ptr2 +(x0 ),tmp13 ,None )
    tl .store (out_ptr3 +(x0 ),tmp15 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused_mean_5 (in_out_ptr0 ,in_ptr0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    r0_numel =5120 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    _tmp2 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_0 =r0_index 
        tmp0 =tl .load (in_ptr0 +(r0_0 ),r0_mask ,eviction_policy ='evict_first',other =0.0 )
        tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
        tmp3 =_tmp2 +tmp1 
        _tmp2 =tl .where (r0_mask ,tmp3 ,_tmp2 )
    tmp2 =tl .sum (_tmp2 ,1 )[:,None ]
    tmp4 =5120.0 
    tmp5 =tmp2 /tmp4 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp5 ,None )

def call (args ):
    primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ,primals_8 ,primals_9 ,primals_10 ,primals_11 ,primals_12 ,primals_13 ,primals_14 ,primals_15 ,primals_16 ,primals_17 ,primals_18 ,primals_19 ,primals_20 ,primals_21 ,primals_22 =args 
    args .clear ()
    assert_size_stride (primals_1 ,(1 ,10 ,512 ),(5120 ,512 ,1 ))
    assert_size_stride (primals_2 ,(1536 ,),(1 ,))
    assert_size_stride (primals_3 ,(1536 ,512 ),(512 ,1 ))
    assert_size_stride (primals_4 ,(512 ,512 ),(512 ,1 ))
    assert_size_stride (primals_5 ,(512 ,),(1 ,))
    assert_size_stride (primals_6 ,(512 ,),(1 ,))
    assert_size_stride (primals_7 ,(512 ,),(1 ,))
    assert_size_stride (primals_8 ,(1536 ,512 ),(512 ,1 ))
    assert_size_stride (primals_9 ,(1536 ,),(1 ,))
    assert_size_stride (primals_10 ,(512 ,512 ),(512 ,1 ))
    assert_size_stride (primals_11 ,(512 ,),(1 ,))
    assert_size_stride (primals_12 ,(512 ,),(1 ,))
    assert_size_stride (primals_13 ,(512 ,),(1 ,))
    assert_size_stride (primals_14 ,(2048 ,512 ),(512 ,1 ))
    assert_size_stride (primals_15 ,(2048 ,),(1 ,))
    assert_size_stride (primals_16 ,(512 ,2048 ),(2048 ,1 ))
    assert_size_stride (primals_17 ,(512 ,),(1 ,))
    assert_size_stride (primals_18 ,(512 ,),(1 ,))
    assert_size_stride (primals_19 ,(512 ,),(1 ,))
    assert_size_stride (primals_20 ,(512 ,512 ),(512 ,1 ))
    assert_size_stride (primals_21 ,(512 ,512 ),(512 ,1 ))
    assert_size_stride (primals_22 ,(512 ,512 ),(512 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((10 ,1536 ),(1536 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(10 ,512 ),(512 ,1 ),0 ),reinterpret_tensor (primals_3 ,(512 ,1536 ),(1 ,512 ),0 ),out =buf0 )
        del primals_3 
        buf1 =empty_strided_cuda ((3 ,10 ,1 ,512 ),(5120 ,512 ,512 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_clone_0 [grid (15360 )](buf0 ,primals_2 ,buf1 ,15360 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del buf0 
        del primals_2 

        buf2 =torch .ops .aten ._scaled_dot_product_efficient_attention .default (reinterpret_tensor (buf1 ,(1 ,8 ,10 ,64 ),(512 ,64 ,512 ,1 ),0 ),reinterpret_tensor (buf1 ,(1 ,8 ,10 ,64 ),(512 ,64 ,512 ,1 ),5120 ),reinterpret_tensor (buf1 ,(1 ,8 ,10 ,64 ),(512 ,64 ,512 ,1 ),10240 ),None ,True ,0.1 )
        buf3 =buf2 [0 ]
        buf4 =buf2 [1 ]
        buf5 =buf2 [2 ]
        buf6 =buf2 [3 ]
        del buf2 
        buf7 =empty_strided_cuda ((10 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf3 ,(10 ,512 ),(512 ,1 ),0 ),reinterpret_tensor (primals_4 ,(512 ,512 ),(1 ,512 ),0 ),out =buf7 )
        buf8 =empty_strided_cuda ((4 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[4 ],out =buf8 )
        buf10 =empty_strided_cuda ((10 ,1 ,512 ),(512 ,512 ,1 ),torch .bool )
        buf14 =reinterpret_tensor (buf7 ,(10 ,1 ,512 ),(512 ,512 ,1 ),0 );del buf7 
        buf15 =empty_strided_cuda ((10 ,1 ,512 ),(512 ,512 ,1 ),torch .float32 )
        buf52 =empty_strided_cuda ((10 ,1 ,1 ),(1 ,1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_1 [grid (10 )](buf14 ,buf8 ,primals_1 ,primals_5 ,primals_6 ,primals_7 ,buf10 ,buf15 ,buf52 ,3 ,10 ,512 ,num_warps =4 ,num_stages =1 )
        del primals_5 
        del primals_7 
        buf16 =empty_strided_cuda ((10 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .addmm (reinterpret_tensor (primals_9 ,(512 ,),(1 ,),0 ),reinterpret_tensor (buf15 ,(10 ,512 ),(512 ,1 ),0 ),reinterpret_tensor (primals_8 ,(512 ,512 ),(1 ,512 ),0 ),alpha =1 ,beta =1 ,out =buf16 )
        buf17 =empty_strided_cuda ((10 ,1024 ),(1024 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(10 ,512 ),(512 ,1 ),0 ),reinterpret_tensor (primals_8 ,(512 ,1024 ),(1 ,512 ),262144 ),out =buf17 )
        buf18 =empty_strided_cuda ((2 ,10 ,1 ,512 ),(5120 ,512 ,512 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_clone_2 [grid (10240 )](buf17 ,primals_9 ,buf18 ,10240 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf17 
        del primals_9 

        buf19 =torch .ops .aten ._scaled_dot_product_efficient_attention .default (reinterpret_tensor (buf16 ,(1 ,8 ,10 ,64 ),(512 ,64 ,512 ,1 ),0 ),reinterpret_tensor (buf18 ,(1 ,8 ,10 ,64 ),(512 ,64 ,512 ,1 ),0 ),reinterpret_tensor (buf18 ,(1 ,8 ,10 ,64 ),(512 ,64 ,512 ,1 ),5120 ),None ,True ,0.1 )
        buf20 =buf19 [0 ]
        buf21 =buf19 [1 ]
        buf22 =buf19 [2 ]
        buf23 =buf19 [3 ]
        del buf19 
        buf24 =empty_strided_cuda ((10 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf20 ,(10 ,512 ),(512 ,1 ),0 ),reinterpret_tensor (primals_10 ,(512 ,512 ),(1 ,512 ),0 ),out =buf24 )
        buf26 =empty_strided_cuda ((10 ,1 ,512 ),(512 ,512 ,1 ),torch .bool )
        buf30 =reinterpret_tensor (buf24 ,(10 ,1 ,512 ),(512 ,512 ,1 ),0 );del buf24 
        buf31 =empty_strided_cuda ((10 ,1 ,512 ),(512 ,512 ,1 ),torch .float32 )
        buf51 =empty_strided_cuda ((10 ,1 ,1 ),(1 ,1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_3 [grid (10 )](buf30 ,buf8 ,buf15 ,primals_11 ,primals_12 ,primals_13 ,buf26 ,buf31 ,buf51 ,1 ,10 ,512 ,num_warps =4 ,num_stages =1 )
        del primals_11 
        del primals_13 
        buf32 =empty_strided_cuda ((10 ,2048 ),(2048 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf31 ,(10 ,512 ),(512 ,1 ),0 ),reinterpret_tensor (primals_14 ,(512 ,2048 ),(1 ,512 ),0 ),out =buf32 )
        buf34 =empty_strided_cuda ((10 ,1 ,2048 ),(2048 ,2048 ,1 ),torch .bool )
        buf35 =empty_strided_cuda ((10 ,1 ,2048 ),(2048 ,2048 ,1 ),torch .float32 )
        buf50 =empty_strided_cuda ((10 ,1 ,2048 ),(2048 ,2048 ,1 ),torch .bool )

        get_raw_stream (0 )
        triton_poi_fused_native_dropout_relu_threshold_backward_4 [grid (20480 )](buf8 ,buf32 ,primals_15 ,buf34 ,buf35 ,buf50 ,2 ,20480 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del buf32 
        del primals_15 
        buf36 =empty_strided_cuda ((10 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf35 ,(10 ,2048 ),(2048 ,1 ),0 ),reinterpret_tensor (primals_16 ,(2048 ,512 ),(1 ,2048 ),0 ),out =buf36 )
        buf38 =empty_strided_cuda ((10 ,1 ,512 ),(512 ,512 ,1 ),torch .bool )
        buf42 =reinterpret_tensor (buf36 ,(10 ,1 ,512 ),(512 ,512 ,1 ),0 );del buf36 
        buf43 =empty_strided_cuda ((10 ,1 ,512 ),(512 ,512 ,1 ),torch .float32 )
        buf49 =empty_strided_cuda ((10 ,1 ,1 ),(1 ,1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_1 [grid (10 )](buf42 ,buf8 ,buf31 ,primals_17 ,primals_18 ,primals_19 ,buf38 ,buf43 ,buf49 ,3 ,10 ,512 ,num_warps =4 ,num_stages =1 )
        del buf8 
        del primals_17 
        del primals_19 
        buf44 =empty_strided_cuda ((10 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf43 ,(10 ,512 ),(512 ,1 ),0 ),primals_20 ,out =buf44 )
        buf45 =empty_strided_cuda ((10 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (buf44 ,primals_21 ,out =buf45 )
        buf46 =empty_strided_cuda ((10 ,512 ),(512 ,1 ),torch .float32 )

        extern_kernels .mm (buf45 ,primals_22 ,out =buf46 )
        buf47 =empty_strided_cuda ((1 ,1 ,1 ,1 ,1 ),(1 ,1 ,1 ,1 ,1 ),torch .float32 )
        buf48 =buf47 ;del buf47 

        get_raw_stream (0 )
        triton_red_fused_mean_5 [grid (1 )](buf48 ,buf46 ,1 ,5120 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del buf46 
    return (reinterpret_tensor (buf48 ,(1 ,1 ),(1 ,1 ),0 ),primals_6 ,primals_12 ,primals_18 ,reinterpret_tensor (primals_1 ,(10 ,512 ),(512 ,1 ),0 ),reinterpret_tensor (buf1 ,(1 ,8 ,10 ,64 ),(512 ,64 ,512 ,1 ),0 ),reinterpret_tensor (buf1 ,(1 ,8 ,10 ,64 ),(512 ,64 ,512 ,1 ),5120 ),reinterpret_tensor (buf1 ,(1 ,8 ,10 ,64 ),(512 ,64 ,512 ,1 ),10240 ),buf3 ,buf4 ,buf5 ,buf6 ,buf10 ,buf14 ,reinterpret_tensor (buf15 ,(10 ,512 ),(512 ,1 ),0 ),reinterpret_tensor (buf16 ,(1 ,8 ,10 ,64 ),(512 ,64 ,512 ,1 ),0 ),reinterpret_tensor (buf18 ,(1 ,8 ,10 ,64 ),(512 ,64 ,512 ,1 ),0 ),reinterpret_tensor (buf18 ,(1 ,8 ,10 ,64 ),(512 ,64 ,512 ,1 ),5120 ),buf20 ,buf21 ,buf22 ,buf23 ,buf26 ,buf30 ,reinterpret_tensor (buf31 ,(10 ,512 ),(512 ,1 ),0 ),buf34 ,reinterpret_tensor (buf35 ,(10 ,2048 ),(2048 ,1 ),0 ),buf38 ,buf42 ,reinterpret_tensor (buf45 ,(512 ,10 ),(1 ,512 ),0 ),reinterpret_tensor (primals_22 ,(512 ,512 ),(1 ,512 ),0 ),reinterpret_tensor (buf44 ,(512 ,10 ),(1 ,512 ),0 ),reinterpret_tensor (primals_21 ,(512 ,512 ),(1 ,512 ),0 ),reinterpret_tensor (buf43 ,(512 ,10 ),(1 ,512 ),0 ),reinterpret_tensor (primals_20 ,(512 ,512 ),(1 ,512 ),0 ),buf49 ,primals_16 ,buf50 ,primals_14 ,buf51 ,primals_10 ,reinterpret_tensor (primals_8 ,(512 ,512 ),(512 ,1 ),0 ),buf52 ,primals_4 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    primals_1 =rand_strided ((1 ,10 ,512 ),(5120 ,512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_2 =rand_strided ((1536 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_3 =rand_strided ((1536 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_4 =rand_strided ((512 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_5 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_6 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_7 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_8 =rand_strided ((1536 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_9 =rand_strided ((1536 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_10 =rand_strided ((512 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_11 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_12 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_13 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_14 =rand_strided ((2048 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_15 =rand_strided ((2048 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_16 =rand_strided ((512 ,2048 ),(2048 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_17 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_18 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_19 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_20 =rand_strided ((512 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_21 =rand_strided ((512 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_22 =rand_strided ((512 ,512 ),(512 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ,primals_8 ,primals_9 ,primals_10 ,primals_11 ,primals_12 ,primals_13 ,primals_14 ,primals_15 ,primals_16 ,primals_17 ,primals_18 ,primals_19 ,primals_20 ,primals_21 ,primals_22 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
