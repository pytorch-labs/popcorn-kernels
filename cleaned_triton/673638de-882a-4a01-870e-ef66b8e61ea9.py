
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
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__to_copy_0 (out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =20 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =0.0 
    tl .store (out_ptr0 +(x0 ),tmp0 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__adaptive_avg_pool2d_1 (in_ptr0 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =100 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %10 )
    x1 =xindex //10 
    x2 =xindex 
    tmp0 =tl .load (in_ptr0 +(x0 +20 *x1 ),xmask )
    tmp13 =tl .load (in_ptr0 +(10 +x0 +20 *x1 ),xmask )
    tmp1 =3.0 
    tmp2 =tmp0 +tmp1 
    tmp3 =0.0 
    tmp4 =triton_helpers .maximum (tmp2 ,tmp3 )
    tmp5 =6.0 
    tmp6 =triton_helpers .minimum (tmp4 ,tmp5 )
    tmp7 =0.16666666666666666 
    tmp8 =tmp6 *tmp7 
    tmp9 =tmp8 >tmp3 
    tmp10 =0.01 
    tmp11 =tmp8 *tmp10 
    tmp12 =tl .where (tmp9 ,tmp8 ,tmp11 )
    tmp14 =tmp13 +tmp1 
    tmp15 =triton_helpers .maximum (tmp14 ,tmp3 )
    tmp16 =triton_helpers .minimum (tmp15 ,tmp5 )
    tmp17 =tmp16 *tmp7 
    tmp18 =tmp17 >tmp3 
    tmp19 =tmp17 *tmp10 
    tmp20 =tl .where (tmp18 ,tmp17 ,tmp19 )
    tmp21 =tmp20 +tmp12 
    tmp22 =0.5 
    tmp23 =tmp21 *tmp22 
    tl .store (out_ptr0 +(x2 ),tmp23 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_add_addmm_stack_tanh_2 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =20 
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
    tl .store (out_ptr0 +(x0 ),tmp7 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_add_addmm_stack_tanh_3 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =20 
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
    tl .store (out_ptr0 +(x0 ),tmp7 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_add_addmm_tanh_tanh_backward_4 (in_ptr0 ,in_ptr1 ,in_ptr2 ,in_ptr3 ,out_ptr0 ,out_ptr1 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =20 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp1 =tl .load (in_ptr1 +(x0 ),xmask )
    tmp3 =tl .load (in_ptr2 +(x0 ),xmask )
    tmp4 =tl .load (in_ptr3 +(x0 ),xmask )
    tmp2 =tmp0 +tmp1 
    tmp5 =tmp3 +tmp4 
    tmp6 =tmp2 +tmp5 
    tmp7 =libdevice .tanh (tmp6 )
    tmp8 =tmp7 *tmp7 
    tmp9 =1.0 
    tmp10 =tmp9 -tmp8 
    tl .store (out_ptr0 +(x0 ),tmp7 ,xmask )
    tl .store (out_ptr1 +(x0 ),tmp10 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused__softmax_bernoulli_5 (in_ptr0 ,in_ptr1 ,out_ptr1 ,out_ptr2 ,out_ptr3 ,load_seed_offset ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    xnumel =20 
    r0_numel =10 
    R0_BLOCK :tl .constexpr =16 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    r0_mask =r0_index <r0_numel 
    x0 =xindex 
    r0_1 =r0_index 
    tmp5 =tl .load (in_ptr1 +(x0 +20 *r0_1 ),r0_mask &xmask ,other =0.0 )
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =x0 
    tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tmp3 =0.5 
    tmp4 =tmp2 <tmp3 
    tmp6 =tmp4 .to (tl .float32 )
    tmp7 =2.0 
    tmp8 =tmp6 *tmp7 
    tmp9 =tmp5 *tmp8 
    tmp10 =tl .broadcast_to (tmp9 ,[XBLOCK ,R0_BLOCK ])
    tmp12 =tl .where (r0_mask &xmask ,tmp10 ,float ("-inf"))
    tmp13 =triton_helpers .max2 (tmp12 ,1 )[:,None ]
    tmp14 =tmp9 -tmp13 
    tmp15 =tl_math .exp (tmp14 )
    tmp16 =tl .broadcast_to (tmp15 ,[XBLOCK ,R0_BLOCK ])
    tmp18 =tl .where (r0_mask &xmask ,tmp16 ,0 )
    tmp19 =tl .sum (tmp18 ,1 )[:,None ]
    tl .store (out_ptr1 +(x0 ),tmp4 ,xmask )
    tl .store (out_ptr2 +(x0 ),tmp13 ,xmask )
    tl .store (out_ptr3 +(x0 ),tmp19 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused__softmax_abs_mean_sub_6 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,in_ptr3 ,out_ptr0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    r0_numel =200 
    R0_BLOCK :tl .constexpr =256 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    r0_mask =r0_index <r0_numel 
    r0_2 =r0_index 
    r0_0 =(r0_index %20 )
    tmp0 =tl .load (in_ptr0 +(r0_2 ),r0_mask ,other =0.0 )
    tmp1 =tl .load (in_ptr1 +(r0_0 ),r0_mask ,eviction_policy ='evict_last',other =0.0 ).to (tl .int1 )
    tmp6 =tl .load (in_ptr2 +(r0_0 ),r0_mask ,eviction_policy ='evict_last',other =0.0 )
    tmp9 =tl .load (in_ptr3 +(r0_0 ),r0_mask ,eviction_policy ='evict_last',other =0.0 )
    tmp2 =tmp1 .to (tl .float32 )
    tmp3 =2.0 
    tmp4 =tmp2 *tmp3 
    tmp5 =tmp0 *tmp4 
    tmp7 =tmp5 -tmp6 
    tmp8 =tl_math .exp (tmp7 )
    tmp10 =tmp8 /tmp9 
    tmp11 =tl_math .abs (tmp10 )
    tmp12 =tl .broadcast_to (tmp11 ,[XBLOCK ,R0_BLOCK ])
    tmp14 =tl .where (r0_mask ,tmp12 ,0 )
    tmp15 =tl .sum (tmp14 ,1 )[:,None ]
    tmp16 =200.0 
    tmp17 =tmp15 /tmp16 
    tl .store (out_ptr0 +(tl .broadcast_to (r0_2 ,[XBLOCK ,R0_BLOCK ])),tmp10 ,r0_mask )
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp17 ,None )

def call (args ):
    primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 =args 
    args .clear ()
    assert_size_stride (primals_2 ,(1 ,20 ,10 ),(200 ,10 ,1 ))
    assert_size_stride (primals_3 ,(20 ,10 ),(10 ,1 ))
    assert_size_stride (primals_4 ,(20 ,20 ),(20 ,1 ))
    assert_size_stride (primals_5 ,(20 ,),(1 ,))
    assert_size_stride (primals_6 ,(20 ,),(1 ,))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,20 ),(20 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused__to_copy_0 [grid (20 )](buf0 ,20 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        buf1 =empty_strided_cuda ((1 ,10 ,1 ,10 ),(100 ,1 ,100 ,10 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused__adaptive_avg_pool2d_1 [grid (100 )](primals_2 ,buf1 ,100 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del primals_2 
        buf2 =empty_strided_cuda ((1 ,20 ),(20 ,1 ),torch .float32 )

        extern_kernels .mm (buf0 ,reinterpret_tensor (primals_4 ,(20 ,20 ),(1 ,20 ),0 ),out =buf2 )
        buf3 =empty_strided_cuda ((1 ,20 ),(20 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf1 ,(1 ,10 ),(0 ,1 ),0 ),reinterpret_tensor (primals_3 ,(10 ,20 ),(1 ,10 ),0 ),out =buf3 )
        buf4 =buf2 ;del buf2 
        buf41 =empty_strided_cuda ((1 ,200 ),(200 ,1 ),torch .float32 )
        buf31 =reinterpret_tensor (buf41 ,(1 ,20 ),(200 ,1 ),0 )

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_stack_tanh_2 [grid (20 )](buf4 ,primals_6 ,buf3 ,primals_5 ,buf31 ,20 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        buf5 =buf3 ;del buf3 

        extern_kernels .mm (buf4 ,reinterpret_tensor (primals_4 ,(20 ,20 ),(1 ,20 ),0 ),out =buf5 )
        buf6 =empty_strided_cuda ((1 ,20 ),(20 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf1 ,(1 ,10 ),(0 ,1 ),10 ),reinterpret_tensor (primals_3 ,(10 ,20 ),(1 ,10 ),0 ),out =buf6 )
        buf7 =buf5 ;del buf5 
        buf32 =reinterpret_tensor (buf41 ,(1 ,20 ),(200 ,1 ),20 )

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_stack_tanh_3 [grid (20 )](buf7 ,primals_6 ,buf6 ,primals_5 ,buf32 ,20 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        buf8 =buf6 ;del buf6 

        extern_kernels .mm (buf7 ,reinterpret_tensor (primals_4 ,(20 ,20 ),(1 ,20 ),0 ),out =buf8 )
        buf9 =empty_strided_cuda ((1 ,20 ),(20 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf1 ,(1 ,10 ),(0 ,1 ),20 ),reinterpret_tensor (primals_3 ,(10 ,20 ),(1 ,10 ),0 ),out =buf9 )
        buf10 =buf8 ;del buf8 
        buf33 =reinterpret_tensor (buf41 ,(1 ,20 ),(200 ,1 ),40 )

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_stack_tanh_3 [grid (20 )](buf10 ,primals_6 ,buf9 ,primals_5 ,buf33 ,20 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        buf11 =buf9 ;del buf9 

        extern_kernels .mm (buf10 ,reinterpret_tensor (primals_4 ,(20 ,20 ),(1 ,20 ),0 ),out =buf11 )
        buf12 =empty_strided_cuda ((1 ,20 ),(20 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf1 ,(1 ,10 ),(0 ,1 ),30 ),reinterpret_tensor (primals_3 ,(10 ,20 ),(1 ,10 ),0 ),out =buf12 )
        buf13 =buf11 ;del buf11 
        buf34 =reinterpret_tensor (buf41 ,(1 ,20 ),(200 ,1 ),60 )

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_stack_tanh_3 [grid (20 )](buf13 ,primals_6 ,buf12 ,primals_5 ,buf34 ,20 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        buf14 =buf12 ;del buf12 

        extern_kernels .mm (buf13 ,reinterpret_tensor (primals_4 ,(20 ,20 ),(1 ,20 ),0 ),out =buf14 )
        buf15 =empty_strided_cuda ((1 ,20 ),(20 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf1 ,(1 ,10 ),(0 ,1 ),40 ),reinterpret_tensor (primals_3 ,(10 ,20 ),(1 ,10 ),0 ),out =buf15 )
        buf16 =buf14 ;del buf14 
        buf35 =reinterpret_tensor (buf41 ,(1 ,20 ),(200 ,1 ),80 )

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_stack_tanh_2 [grid (20 )](buf16 ,primals_6 ,buf15 ,primals_5 ,buf35 ,20 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        buf17 =buf15 ;del buf15 

        extern_kernels .mm (buf16 ,reinterpret_tensor (primals_4 ,(20 ,20 ),(1 ,20 ),0 ),out =buf17 )
        buf18 =empty_strided_cuda ((1 ,20 ),(20 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf1 ,(1 ,10 ),(0 ,1 ),50 ),reinterpret_tensor (primals_3 ,(10 ,20 ),(1 ,10 ),0 ),out =buf18 )
        buf19 =buf17 ;del buf17 
        buf36 =reinterpret_tensor (buf41 ,(1 ,20 ),(200 ,1 ),100 )

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_stack_tanh_3 [grid (20 )](buf19 ,primals_6 ,buf18 ,primals_5 ,buf36 ,20 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        buf20 =buf18 ;del buf18 

        extern_kernels .mm (buf19 ,reinterpret_tensor (primals_4 ,(20 ,20 ),(1 ,20 ),0 ),out =buf20 )
        buf21 =empty_strided_cuda ((1 ,20 ),(20 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf1 ,(1 ,10 ),(0 ,1 ),60 ),reinterpret_tensor (primals_3 ,(10 ,20 ),(1 ,10 ),0 ),out =buf21 )
        buf22 =buf20 ;del buf20 
        buf37 =reinterpret_tensor (buf41 ,(1 ,20 ),(200 ,1 ),120 )

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_stack_tanh_3 [grid (20 )](buf22 ,primals_6 ,buf21 ,primals_5 ,buf37 ,20 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        buf23 =buf21 ;del buf21 

        extern_kernels .mm (buf22 ,reinterpret_tensor (primals_4 ,(20 ,20 ),(1 ,20 ),0 ),out =buf23 )
        buf24 =empty_strided_cuda ((1 ,20 ),(20 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf1 ,(1 ,10 ),(0 ,1 ),70 ),reinterpret_tensor (primals_3 ,(10 ,20 ),(1 ,10 ),0 ),out =buf24 )
        buf25 =buf23 ;del buf23 
        buf38 =reinterpret_tensor (buf41 ,(1 ,20 ),(200 ,1 ),140 )

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_stack_tanh_3 [grid (20 )](buf25 ,primals_6 ,buf24 ,primals_5 ,buf38 ,20 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        buf26 =buf24 ;del buf24 

        extern_kernels .mm (buf25 ,reinterpret_tensor (primals_4 ,(20 ,20 ),(1 ,20 ),0 ),out =buf26 )
        buf27 =empty_strided_cuda ((1 ,20 ),(20 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf1 ,(1 ,10 ),(0 ,1 ),80 ),reinterpret_tensor (primals_3 ,(10 ,20 ),(1 ,10 ),0 ),out =buf27 )
        buf28 =buf26 ;del buf26 
        buf39 =reinterpret_tensor (buf41 ,(1 ,20 ),(200 ,1 ),160 )

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_stack_tanh_2 [grid (20 )](buf28 ,primals_6 ,buf27 ,primals_5 ,buf39 ,20 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        buf29 =buf27 ;del buf27 

        extern_kernels .mm (buf28 ,reinterpret_tensor (primals_4 ,(20 ,20 ),(1 ,20 ),0 ),out =buf29 )
        buf30 =empty_strided_cuda ((1 ,20 ),(20 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf1 ,(1 ,10 ),(0 ,1 ),90 ),reinterpret_tensor (primals_3 ,(10 ,20 ),(1 ,10 ),0 ),out =buf30 )
        del primals_3 
        buf40 =reinterpret_tensor (buf41 ,(1 ,20 ),(200 ,1 ),180 )
        buf49 =empty_strided_cuda ((1 ,20 ),(20 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_add_addmm_tanh_tanh_backward_4 [grid (20 )](buf29 ,primals_6 ,buf30 ,primals_5 ,buf40 ,buf49 ,20 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        del primals_5 
        del primals_6 
        del buf31 
        del buf32 
        del buf33 
        del buf34 
        del buf35 
        del buf36 
        del buf37 
        del buf38 
        del buf39 
        del buf40 
        buf42 =empty_strided_cuda ((1 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[1 ],out =buf42 )
        buf44 =empty_strided_cuda ((1 ,20 ,1 ),(20 ,1 ,1 ),torch .bool )
        buf45 =reinterpret_tensor (buf30 ,(1 ,1 ,20 ),(20 ,20 ,1 ),0 );del buf30 
        buf46 =reinterpret_tensor (buf29 ,(1 ,1 ,20 ),(20 ,20 ,1 ),0 );del buf29 

        get_raw_stream (0 )
        triton_per_fused__softmax_bernoulli_5 [grid (20 )](buf42 ,buf41 ,buf44 ,buf45 ,buf46 ,0 ,20 ,10 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        del buf42 
        buf47 =empty_strided_cuda ((1 ,10 ,20 ),(200 ,20 ,1 ),torch .float32 )
        buf48 =empty_strided_cuda ((),(),torch .float32 )
        buf50 =buf48 ;del buf48 

        get_raw_stream (0 )
        triton_per_fused__softmax_abs_mean_sub_6 [grid (1 )](buf50 ,buf41 ,buf44 ,buf45 ,buf46 ,buf47 ,1 ,200 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        del buf41 
        del buf45 
        del buf46 
    return (buf47 ,buf50 ,buf0 ,reinterpret_tensor (buf1 ,(1 ,10 ),(100 ,1 ),0 ),buf4 ,reinterpret_tensor (buf1 ,(1 ,10 ),(100 ,1 ),10 ),buf7 ,reinterpret_tensor (buf1 ,(1 ,10 ),(100 ,1 ),20 ),buf10 ,reinterpret_tensor (buf1 ,(1 ,10 ),(100 ,1 ),30 ),buf13 ,reinterpret_tensor (buf1 ,(1 ,10 ),(100 ,1 ),40 ),buf16 ,reinterpret_tensor (buf1 ,(1 ,10 ),(100 ,1 ),50 ),buf19 ,reinterpret_tensor (buf1 ,(1 ,10 ),(100 ,1 ),60 ),buf22 ,reinterpret_tensor (buf1 ,(1 ,10 ),(100 ,1 ),70 ),buf25 ,reinterpret_tensor (buf1 ,(1 ,10 ),(100 ,1 ),80 ),buf28 ,reinterpret_tensor (buf1 ,(1 ,10 ),(100 ,1 ),90 ),buf44 ,buf47 ,buf49 ,primals_4 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    primals_1 =20 
    primals_2 =rand_strided ((1 ,20 ,10 ),(200 ,10 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_3 =rand_strided ((20 ,10 ),(10 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_4 =rand_strided ((20 ,20 ),(20 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_5 =rand_strided ((20 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_6 =rand_strided ((20 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
