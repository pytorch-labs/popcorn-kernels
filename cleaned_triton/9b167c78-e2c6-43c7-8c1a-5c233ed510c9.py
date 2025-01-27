
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
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_max_unpool2d_0 (out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
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
def triton_poi_fused_max_pool2d_with_indices_max_unpool2d_1 (in_ptr0 ,out_ptr1 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%ks1 )
    x2 =xindex //ks2 
    x3 =xindex 
    tmp0 =tl .load (in_ptr0 +(2 *x0 +2 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp1 =tl .load (in_ptr0 +(1 +2 *x0 +2 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp7 =tl .load (in_ptr0 +(ks4 +2 *x0 +2 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp12 =tl .load (in_ptr0 +(1 +ks4 +2 *x0 +2 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp35 =tl .load (in_ptr0 +(2 *((x3 %ks0 ))+2 *ks4 *(((x3 //ks0 )%ks1 ))+ks3 *ks4 *(x3 //ks2 )),xmask ,eviction_policy ='evict_last')
    tmp36 =tl .load (in_ptr0 +(1 +2 *((x3 %ks0 ))+2 *ks4 *(((x3 //ks0 )%ks1 ))+ks3 *ks4 *(x3 //ks2 )),xmask ,eviction_policy ='evict_last')
    tmp38 =tl .load (in_ptr0 +(ks4 +2 *((x3 %ks0 ))+2 *ks4 *(((x3 //ks0 )%ks1 ))+ks3 *ks4 *(x3 //ks2 )),xmask ,eviction_policy ='evict_last')
    tmp40 =tl .load (in_ptr0 +(1 +ks4 +2 *((x3 %ks0 ))+2 *ks4 *(((x3 //ks0 )%ks1 ))+ks3 *ks4 *(x3 //ks2 )),xmask ,eviction_policy ='evict_last')
    tmp2 =tmp1 >tmp0 
    tmp3 =tl .full ([1 ],1 ,tl .int8 )
    tmp4 =tl .full ([1 ],0 ,tl .int8 )
    tmp5 =tl .where (tmp2 ,tmp3 ,tmp4 )
    tmp6 =triton_helpers .maximum (tmp1 ,tmp0 )
    tmp8 =tmp7 >tmp6 
    tmp9 =tl .full ([1 ],2 ,tl .int8 )
    tmp10 =tl .where (tmp8 ,tmp9 ,tmp5 )
    tmp11 =triton_helpers .maximum (tmp7 ,tmp6 )
    tmp13 =tmp12 >tmp11 
    tmp14 =tl .full ([1 ],3 ,tl .int8 )
    tmp15 =tl .where (tmp13 ,tmp14 ,tmp10 )
    triton_helpers .maximum (tmp12 ,tmp11 )
    tmp17 =tl .full ([1 ],2 ,tl .int32 )
    tmp18 =tl .where ((tmp15 <0 )!=(tmp17 <0 ),tl .where (tmp15 %tmp17 !=0 ,tmp15 //tmp17 -1 ,tmp15 //tmp17 ),tmp15 //tmp17 )
    tmp19 =tmp18 *tmp17 
    tmp20 =tmp15 -tmp19 
    tmp21 =2 *x1 
    tmp22 =tmp21 +tmp18 
    tmp23 =2 *x0 
    tmp24 =tmp23 +tmp20 
    tmp25 =ks4 
    tmp26 =tmp22 *tmp25 
    tmp27 =tmp26 +tmp24 
    tmp28 =4 *ks0 *ks1 *x2 
    tmp29 =tmp27 +tmp28 
    tmp30 =40 *ks0 *ks1 *ks5 
    tmp31 =tmp29 +tmp30 
    tmp32 =tmp29 <0 
    tmp33 =tl .where (tmp32 ,tmp31 ,tmp29 )
    tl .device_assert (((0 <=tmp33 )&(tmp33 <40 *ks5 *(ks3 //2 )*(ks4 //2 )))|~(xmask ),"index out of bounds: 0 <= tmp33 < 40*ks5*(ks3 // 2)*(ks4 // 2)")
    tmp37 =triton_helpers .maximum (tmp36 ,tmp35 )
    tmp39 =triton_helpers .maximum (tmp38 ,tmp37 )
    tmp41 =triton_helpers .maximum (tmp40 ,tmp39 )
    tl .store (out_ptr1 +(tl .broadcast_to ((tmp33 %(40 *ks0 *ks1 *ks5 )),[XBLOCK ])),tmp41 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused__native_batch_norm_legit_2 (in_ptr0 ,out_ptr0 ,out_ptr1 ,ks0 ,ks1 ,ks2 ,ks3 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    tmp2_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp2_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp2_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp10_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp10_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp10_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        roffset =r0_offset 
        r0_1 =(r0_index %ks0 )
        r0_2 =r0_index //ks0 
        tmp0 =tl .load (in_ptr0 +(r0_1 +2 *ks1 *((((r0_1 +2 *ks1 *r0_2 )//(2 *ks1 ))%(2 *ks2 )))+4 *ks1 *ks2 *((((r0_1 +2 *ks1 *r0_2 +4 *ks1 *ks2 *((x0 %10 )))//(4 *ks1 *ks2 ))%10 ))+40 *ks1 *ks2 *((((r0_1 +2 *ks1 *r0_2 +4 *ks1 *ks2 *((x0 %10 ))+40 *ks1 *ks2 *(x0 //10 ))//(40 *ks1 *ks2 ))%ks3 ))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp8 =tl .load (in_ptr0 +(r0_1 +2 *ks1 *((((r0_1 +2 *ks1 *r0_2 )//ks0 )%(2 *ks2 )))+4 *ks1 *ks2 *((((r0_1 +2 *ks1 *r0_2 +4 *ks1 *ks2 *((x0 %10 )))//(4 *ks1 *ks2 ))%10 ))+40 *ks1 *ks2 *((((r0_1 +2 *ks1 *r0_2 +4 *ks1 *ks2 *((x0 %10 ))+40 *ks1 *ks2 *(x0 //10 ))//(40 *ks1 *ks2 ))%ks3 ))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
        tmp2_mean_next ,tmp2_m2_next ,tmp2_weight_next =triton_helpers .welford_reduce (
        tmp1 ,tmp2_mean ,tmp2_m2 ,tmp2_weight ,roffset ==0 
        )
        tmp2_mean =tl .where (r0_mask &xmask ,tmp2_mean_next ,tmp2_mean )
        tmp2_m2 =tl .where (r0_mask &xmask ,tmp2_m2_next ,tmp2_m2 )
        tmp2_weight =tl .where (r0_mask &xmask ,tmp2_weight_next ,tmp2_weight )
        tmp9 =tl .broadcast_to (tmp8 ,[XBLOCK ,R0_BLOCK ])
        tmp10_mean_next ,tmp10_m2_next ,tmp10_weight_next =triton_helpers .welford_reduce (
        tmp9 ,tmp10_mean ,tmp10_m2 ,tmp10_weight ,roffset ==0 
        )
        tmp10_mean =tl .where (r0_mask &xmask ,tmp10_mean_next ,tmp10_mean )
        tmp10_m2 =tl .where (r0_mask &xmask ,tmp10_m2_next ,tmp10_m2 )
        tmp10_weight =tl .where (r0_mask &xmask ,tmp10_weight_next ,tmp10_weight )
    tmp5 ,tmp6 ,tmp7 =triton_helpers .welford (tmp2_mean ,tmp2_m2 ,tmp2_weight ,1 )
    tmp2 =tmp5 [:,None ]
    tmp3 =tmp6 [:,None ]
    tmp4 =tmp7 [:,None ]
    tmp13 ,tmp14 ,tmp15 =triton_helpers .welford (tmp10_mean ,tmp10_m2 ,tmp10_weight ,1 )
    tmp10 =tmp13 [:,None ]
    tmp11 =tmp14 [:,None ]
    tmp12 =tmp15 [:,None ]
    tl .store (out_ptr0 +(x0 ),tmp2 ,xmask )
    tl .store (out_ptr1 +(x0 ),tmp11 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused_add_norm_sub_3 (in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr0 ,out_ptr1 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%10 )
    x2 =xindex //ks1 
    x4 =xindex //ks0 
    tmp1 =tl .load (in_ptr1 +(x4 ),xmask ,eviction_policy ='evict_last')
    tmp3 =tl .load (in_ptr2 +(x4 ),xmask ,eviction_policy ='evict_last')
    tmp12 =tl .load (in_ptr1 +(x4 +10 *(ks5 //3 )),xmask ,eviction_policy ='evict_last')
    tmp14 =tl .load (in_ptr2 +(x4 +10 *(ks5 //3 )),xmask ,eviction_policy ='evict_last')
    _tmp24 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    x5 =xindex 
    tmp30 =tl .load (in_ptr1 +(x4 +10 *((2 *ks5 )//3 )),xmask ,eviction_policy ='evict_last')
    tmp32 =tl .load (in_ptr2 +(x4 +10 *((2 *ks5 )//3 )),xmask ,eviction_policy ='evict_last')
    _tmp41 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_3 =r0_index 
        tmp0 =tl .load (in_ptr0 +(r0_3 +2 *ks2 *((((r0_3 +2 *ks2 *x0 )//ks4 )%(2 *ks3 )))+4 *ks2 *ks3 *((((r0_3 +2 *ks2 *x0 +4 *ks2 *ks3 *x1 )//(4 *ks2 *ks3 ))%10 ))+40 *ks2 *ks3 *((((r0_3 +2 *ks2 *x0 +4 *ks2 *ks3 *x1 +40 *ks2 *ks3 *((x2 %ks5 )))//(40 *ks2 *ks3 ))%ks5 ))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp11 =tl .load (in_ptr0 +(r0_3 +2 *ks2 *((((r0_3 +2 *ks2 *x0 )//ks4 )%ks0 ))+4 *ks2 *ks3 *((((r0_3 +2 *ks2 *x0 +4 *ks2 *ks3 *x1 )//(4 *ks2 *ks3 ))%10 ))+40 *ks2 *ks3 *((((r0_3 +2 *ks2 *x0 +4 *ks2 *ks3 *x1 +40 *ks2 *ks3 *(((x2 +(ks5 //3 ))%ks5 )))//(40 *ks2 *ks3 ))%ks5 ))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp26 =tl .load (in_ptr0 +(r0_3 +2 *ks2 *((((r0_3 +2 *ks2 *x0 )//ks4 )%ks0 ))+4 *ks2 *ks3 *((((r0_3 +2 *ks2 *x0 +4 *ks2 *ks3 *x1 )//(4 *ks2 *ks3 ))%10 ))+40 *ks2 *ks3 *((((r0_3 +2 *ks2 *x0 +4 *ks2 *ks3 *x1 +40 *ks2 *ks3 *((x2 %ks5 )))//(40 *ks2 *ks3 ))%ks5 ))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp29 =tl .load (in_ptr0 +(r0_3 +2 *ks2 *((((r0_3 +2 *ks2 *x0 )//ks4 )%ks0 ))+4 *ks2 *ks3 *((((r0_3 +2 *ks2 *x0 +4 *ks2 *ks3 *x1 )//(4 *ks2 *ks3 ))%10 ))+40 *ks2 *ks3 *((((r0_3 +2 *ks2 *x0 +4 *ks2 *ks3 *x1 +40 *ks2 *ks3 *(((x2 +((2 *ks5 )//3 ))%ks5 )))//(40 *ks2 *ks3 ))%ks5 ))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp2 =tmp0 -tmp1 
        tmp4 =4 *ks2 *ks3 
        tmp5 =tmp4 .to (tl .float32 )
        tmp6 =tmp3 /tmp5 
        tmp7 =1e-05 
        tmp8 =tmp6 +tmp7 
        tmp9 =libdevice .rsqrt (tmp8 )
        tmp10 =tmp2 *tmp9 
        tmp13 =tmp11 -tmp12 
        tmp15 =tmp14 /tmp5 
        tmp16 =tmp15 +tmp7 
        tmp17 =libdevice .rsqrt (tmp16 )
        tmp18 =tmp13 *tmp17 
        tmp19 =tmp10 -tmp18 
        tmp20 =1e-06 
        tmp21 =tmp19 +tmp20 
        tmp22 =tmp21 *tmp21 
        tmp23 =tl .broadcast_to (tmp22 ,[XBLOCK ,R0_BLOCK ])
        tmp25 =_tmp24 +tmp23 
        _tmp24 =tl .where (r0_mask &xmask ,tmp25 ,_tmp24 )
        tmp27 =tmp26 -tmp1 
        tmp28 =tmp27 *tmp9 
        tmp31 =tmp29 -tmp30 
        tmp33 =tmp32 /tmp5 
        tmp34 =tmp33 +tmp7 
        tmp35 =libdevice .rsqrt (tmp34 )
        tmp36 =tmp31 *tmp35 
        tmp37 =tmp28 -tmp36 
        tmp38 =tmp37 +tmp20 
        tmp39 =tmp38 *tmp38 
        tmp40 =tl .broadcast_to (tmp39 ,[XBLOCK ,R0_BLOCK ])
        tmp42 =_tmp41 +tmp40 
        _tmp41 =tl .where (r0_mask &xmask ,tmp42 ,_tmp41 )
    tmp24 =tl .sum (_tmp24 ,1 )[:,None ]
    tmp41 =tl .sum (_tmp41 ,1 )[:,None ]
    tl .store (out_ptr0 +(x5 ),tmp24 ,xmask )
    tl .store (out_ptr1 +(x5 ),tmp41 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused_add_clamp_min_mean_norm_sub_4 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,ks0 ,ks1 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    _tmp10 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_0 =r0_index 
        tmp0 =tl .load (in_ptr0 +(r0_0 ),r0_mask ,eviction_policy ='evict_first',other =0.0 )
        tmp4 =tl .load (in_ptr1 +(r0_0 ),r0_mask ,eviction_policy ='evict_first',other =0.0 )
        tmp1 =libdevice .sqrt (tmp0 )
        tmp2 =1.0 
        tmp3 =tmp1 +tmp2 
        tmp5 =libdevice .sqrt (tmp4 )
        tmp6 =tmp3 -tmp5 
        tmp7 =0.0 
        tmp8 =triton_helpers .maximum (tmp6 ,tmp7 )
        tmp9 =tl .broadcast_to (tmp8 ,[XBLOCK ,R0_BLOCK ])
        tmp11 =_tmp10 +tmp9 
        _tmp10 =tl .where (r0_mask ,tmp11 ,_tmp10 )
    tmp10 =tl .sum (_tmp10 ,1 )[:,None ]
    tmp12 =20 *ks0 *(ks1 //3 )
    tmp13 =tmp12 .to (tl .float32 )
    tmp14 =tmp10 /tmp13 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp14 ,None )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s2 =arg1_1 
    s3 =arg2_1 
    assert_size_stride (arg3_1 ,(s0 ,10 ,s2 ,s3 ),(10 *s2 *s3 ,s2 *s3 ,s3 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf1 =empty_strided_cuda ((s0 ,10 ,2 *(s2 //2 ),2 *(s3 //2 )),(40 *(s2 //2 )*(s3 //2 ),4 *(s2 //2 )*(s3 //2 ),2 *(s3 //2 ),1 ),torch .float32 )

        triton_poi_fused_max_unpool2d_0_xnumel =40 *s0 *(s2 //2 )*(s3 //2 )
        get_raw_stream (0 )
        triton_poi_fused_max_unpool2d_0 [grid (triton_poi_fused_max_unpool2d_0_xnumel )](buf1 ,122880 ,XBLOCK =512 ,num_warps =8 ,num_stages =1 )
        s3 //2 
        s2 //2 
        (s2 //2 )*(s3 //2 )

        triton_poi_fused_max_pool2d_with_indices_max_unpool2d_1_xnumel =10 *s0 *(s2 //2 )*(s3 //2 )
        get_raw_stream (0 )
        triton_poi_fused_max_pool2d_with_indices_max_unpool2d_1 [grid (triton_poi_fused_max_pool2d_with_indices_max_unpool2d_1_xnumel )](arg3_1 ,buf1 ,16 ,16 ,256 ,32 ,32 ,12 ,30720 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        2 *(s3 //2 )
        buf3 =empty_strided_cuda ((1 ,10 *s0 ,1 ,1 ,1 ),(10 *s0 ,1 ,10 *s0 ,10 *s0 ,10 *s0 ),torch .float32 )
        buf4 =empty_strided_cuda ((1 ,10 *s0 ,1 ,1 ,1 ),(10 *s0 ,1 ,10 *s0 ,10 *s0 ,10 *s0 ),torch .float32 )

        triton_red_fused__native_batch_norm_legit_2_xnumel =10 *s0 
        4 *(s2 //2 )*(s3 //2 )
        get_raw_stream (0 )
        triton_red_fused__native_batch_norm_legit_2 [grid (triton_red_fused__native_batch_norm_legit_2_xnumel )](buf1 ,buf3 ,buf4 ,32 ,16 ,16 ,12 ,120 ,1024 ,XBLOCK =1 ,R0_BLOCK =1024 ,num_warps =8 ,num_stages =1 )
        2 *(s2 //2 )
        20 *(s2 //2 )
        buf6 =empty_strided_cuda ((s0 //3 ,10 ,1 ,2 *(s2 //2 )),(20 *(s2 //2 ),2 *(s2 //2 ),20 *(s0 //3 )*(s2 //2 ),1 ),torch .float32 )
        buf7 =empty_strided_cuda ((s0 //3 ,10 ,1 ,2 *(s2 //2 )),(20 *(s2 //2 ),2 *(s2 //2 ),20 *(s0 //3 )*(s2 //2 ),1 ),torch .float32 )

        triton_red_fused_add_norm_sub_3_xnumel =20 *(s0 //3 )*(s2 //2 )
        2 *(s3 //2 )
        get_raw_stream (0 )
        triton_red_fused_add_norm_sub_3 [grid (triton_red_fused_add_norm_sub_3_xnumel )](buf1 ,buf3 ,buf4 ,buf6 ,buf7 ,32 ,320 ,16 ,16 ,32 ,12 ,1280 ,32 ,XBLOCK =8 ,R0_BLOCK =32 ,num_warps =2 ,num_stages =1 )
        del buf1 
        del buf3 
        del buf4 
        buf8 =empty_strided_cuda ((),(),torch .float32 )
        buf9 =buf8 ;del buf8 

        20 *(s0 //3 )*(s2 //2 )
        get_raw_stream (0 )
        triton_red_fused_add_clamp_min_mean_norm_sub_4 [grid (1 )](buf9 ,buf6 ,buf7 ,16 ,12 ,1 ,1280 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del buf6 
        del buf7 
    return (buf9 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =12 
    arg1_1 =32 
    arg2_1 =32 
    arg3_1 =rand_strided ((12 ,10 ,32 ,32 ),(10240 ,1024 ,32 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
