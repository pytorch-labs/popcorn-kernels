
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
def triton_poi_fused_native_dropout_0 (in_ptr0 ,out_ptr0 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
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
def triton_poi_fused_native_dropout_1 (in_out_ptr0 ,in_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x3 =xindex 
    x1 =((xindex //ks1 )%ks2 )
    x0 =(xindex %ks1 )
    x7 =xindex //ks4 
    tmp0 =tl .load (in_out_ptr0 +(x3 ),xmask ,eviction_policy ='evict_last')
    tmp1 =0.5 
    tmp2 =tmp0 >tmp1 
    tmp3 =tmp2 .to (tl .float32 )
    tmp4 =tl .full ([1 ],2.0 ,tl .float64 )
    tmp5 =ks0 
    tmp6 =tmp5 .to (tl .float64 )
    tmp7 =tmp4 *tmp6 
    tmp8 =tmp6 /tmp7 
    tmp9 =tmp8 .to (tl .float32 )
    tmp10 =tl .where ((-1 )+((-1 )*tl_math .abs (1 +((-2 )*ks0 )+tl_math .abs ((-1 )+x1 )))+2 *ks0 <0 ,(-1 )+((-1 )*tl_math .abs (1 +((-2 )*ks0 )+tl_math .abs ((-1 )+x1 )))+4 *ks0 ,(-1 )+((-1 )*tl_math .abs (1 +((-2 )*ks0 )+tl_math .abs ((-1 )+x1 )))+2 *ks0 )
    tmp11 =tmp10 .to (tl .float32 )
    tmp12 =tmp11 *tmp9 
    tmp13 =tmp12 .to (tl .int64 )
    tmp14 =tmp13 +tmp5 
    tmp15 =tmp13 <0 
    tmp16 =tl .where (tmp15 ,tmp14 ,tmp13 )
    tmp17 =ks3 
    tmp18 =tmp17 .to (tl .float64 )
    tmp19 =tmp4 *tmp18 
    tmp20 =tmp18 /tmp19 
    tmp21 =tmp20 .to (tl .float32 )
    tmp22 =tl .where ((-1 )+((-1 )*tl_math .abs (1 +((-2 )*ks3 )+tl_math .abs ((-1 )+x0 )))+2 *ks3 <0 ,(-1 )+((-1 )*tl_math .abs (1 +((-2 )*ks3 )+tl_math .abs ((-1 )+x0 )))+4 *ks3 ,(-1 )+((-1 )*tl_math .abs (1 +((-2 )*ks3 )+tl_math .abs ((-1 )+x0 )))+2 *ks3 )
    tmp23 =tmp22 .to (tl .float32 )
    tmp24 =tmp23 *tmp21 
    tmp25 =tmp24 .to (tl .int64 )
    tmp26 =tmp25 +tmp17 
    tmp27 =tmp25 <0 
    tmp28 =tl .where (tmp27 ,tmp26 ,tmp25 )
    tmp29 =tl .load (in_ptr0 +(tmp28 +ks3 *tmp16 +ks0 *ks3 *(tl .where ((-1 )+ks5 +((-1 )*tl_math .abs (1 +((-1 )*ks5 )+tl_math .abs ((-1 )+x7 )))<0 ,(-1 )+((-1 )*tl_math .abs (1 +((-1 )*ks5 )+tl_math .abs ((-1 )+x7 )))+2 *ks5 ,(-1 )+ks5 +((-1 )*tl_math .abs (1 +((-1 )*ks5 )+tl_math .abs ((-1 )+x7 )))))),xmask ,eviction_policy ='evict_last')
    tmp30 =tmp3 *tmp29 
    tl .store (in_out_ptr0 +(x3 ),tmp30 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_bernoulli_2 (in_ptr0 ,out_ptr0 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
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
def triton_red_fused__to_copy_add_bernoulli_clamp_min_fill_hardtanh_mean_mul_native_dropout_ne_soft_margin_loss_sub_where_zeros_like_3 (in_ptr0 ,in_ptr1 ,out_ptr0 ,out_ptr1 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =3 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    _tmp30 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    _tmp43 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_1 =r0_index 
        tmp0 =r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 )
        tmp1 =8 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 
        tmp2 =tmp0 <tmp1 
        tmp3 =tl .load (in_ptr0 +(2 *((((r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 ))//ks3 )%ks4 ))+4 *((((r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 ))//ks5 )%(2 +ks0 )))+2 *ks2 *((((r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 ))//ks3 )%ks4 ))+4 *ks1 *((((r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 ))//ks5 )%(2 +ks0 )))+4 *ks2 *((((r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 ))//ks5 )%(2 +ks0 )))+4 *ks1 *ks2 *((((r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 ))//ks5 )%(2 +ks0 )))+(((r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 ))%ks3 ))),r0_mask &tmp2 &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp4 =2.0 
        tmp5 =tmp3 *tmp4 
        tmp6 =0.0 
        tmp7 =triton_helpers .maximum (tmp5 ,tmp6 )
        tmp8 =6.0 
        tmp9 =triton_helpers .minimum (tmp7 ,tmp8 )
        tmp10 =tl .load (in_ptr1 +((((r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 ))//ks5 )%(2 +ks0 ))),r0_mask &tmp2 &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp11 =0.5 
        tmp12 =tmp10 <tmp11 
        tmp13 =tmp12 .to (tl .float32 )
        tmp14 =0.8864048946659319 
        tmp15 =tmp13 *tmp14 
        tmp16 =tmp9 *tmp15 
        tmp17 =-1.0 
        tmp18 =tmp13 +tmp17 
        tmp19 =1.558387861036063 
        tmp20 =tmp18 *tmp19 
        tmp21 =0.7791939305180315 
        tmp22 =tmp20 +tmp21 
        tmp23 =tmp16 +tmp22 
        tmp24 =-tmp23 
        tmp25 =tl_math .exp (tmp24 )
        tmp26 =libdevice .log1p (tmp25 )
        tmp27 =tl .full (tmp26 .shape ,0 ,tmp26 .dtype )
        tmp28 =tl .where (tmp2 ,tmp26 ,tmp27 )
        tmp29 =tl .broadcast_to (tmp28 ,[XBLOCK ,R0_BLOCK ])
        tmp31 =_tmp30 +tmp29 
        _tmp30 =tl .where (r0_mask &xmask ,tmp31 ,_tmp30 )
        tmp32 =1.0 
        tmp33 =tmp32 -tmp23 
        tmp34 =triton_helpers .maximum (tmp33 ,tmp6 )
        tmp35 =tl .full ([1 ,1 ],False ,tl .int1 )
        tmp36 =tl .where (tmp35 ,tmp34 ,tmp6 )
        tmp37 =tl .full ([1 ,1 ],True ,tl .int1 )
        tmp38 =tl .where (tmp37 ,tmp23 ,tmp6 )
        tmp39 =tmp36 +tmp38 
        tmp40 =tl .full (tmp39 .shape ,0 ,tmp39 .dtype )
        tmp41 =tl .where (tmp2 ,tmp39 ,tmp40 )
        tmp42 =tl .broadcast_to (tmp41 ,[XBLOCK ,R0_BLOCK ])
        tmp44 =_tmp43 +tmp42 
        _tmp43 =tl .where (r0_mask &xmask ,tmp44 ,_tmp43 )
    tmp30 =tl .sum (_tmp30 ,1 )[:,None ]
    tmp43 =tl .sum (_tmp43 ,1 )[:,None ]
    tl .store (out_ptr0 +(x0 ),tmp30 ,xmask )
    tl .store (out_ptr1 +(x0 ),tmp43 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused__to_copy_add_bernoulli_hardtanh_mul_native_dropout_soft_margin_loss_4 (in_out_ptr0 ,in_ptr0 ,ks0 ,ks1 ,ks2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    r0_numel =3 
    R0_BLOCK :tl .constexpr =4 
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
    tmp5 =8 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 
    tmp6 =tmp5 .to (tl .float32 )
    tmp7 =tmp4 /tmp6 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp7 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__to_copy_add_bernoulli_hardtanh_huber_loss_mul_native_dropout_ones_like_5 (in_out_ptr0 ,in_ptr0 ,ks0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x2 =xindex 
    x1 =xindex //ks0 
    tmp0 =tl .load (in_out_ptr0 +(x2 ),xmask ,eviction_policy ='evict_last')
    tmp7 =tl .load (in_ptr0 +(x1 ),xmask ,eviction_policy ='evict_last')
    tmp1 =2.0 
    tmp2 =tmp0 *tmp1 
    tmp3 =0.0 
    tmp4 =triton_helpers .maximum (tmp2 ,tmp3 )
    tmp5 =6.0 
    tmp6 =triton_helpers .minimum (tmp4 ,tmp5 )
    tmp8 =0.5 
    tmp9 =tmp7 <tmp8 
    tmp10 =tmp9 .to (tl .float32 )
    tmp11 =0.8864048946659319 
    tmp12 =tmp10 *tmp11 
    tmp13 =tmp6 *tmp12 
    tmp14 =-1.0 
    tmp15 =tmp10 +tmp14 
    tmp16 =1.558387861036063 
    tmp17 =tmp15 *tmp16 
    tmp18 =0.7791939305180315 
    tmp19 =tmp17 +tmp18 
    tmp20 =tmp13 +tmp19 
    tmp21 =1.0 
    tmp22 =tmp20 -tmp21 
    tmp23 =tl_math .abs (tmp22 )
    tmp24 =tmp23 <tmp21 
    tmp25 =tmp23 *tmp8 
    tmp26 =tmp25 *tmp23 
    tmp27 =tmp23 -tmp8 
    tmp28 =tmp27 *tmp21 
    tmp29 =tl .where (tmp24 ,tmp26 ,tmp28 )
    tl .store (in_out_ptr0 +(x2 ),tmp29 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused_huber_loss_6 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =3 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    _tmp5 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_1 =r0_index 
        tmp0 =r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 )
        tmp1 =8 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 
        tmp2 =tmp0 <tmp1 
        tmp3 =tl .load (in_ptr0 +(2 *((((r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 ))//ks3 )%ks4 ))+4 *((((r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 ))//ks5 )%(2 +ks0 )))+2 *ks2 *((((r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 ))//ks3 )%ks4 ))+4 *ks1 *((((r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 ))//ks5 )%(2 +ks0 )))+4 *ks2 *((((r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 ))//ks5 )%(2 +ks0 )))+4 *ks1 *ks2 *((((r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 ))//ks5 )%(2 +ks0 )))+(((r0_1 +x0 *((10 +4 *ks0 +8 *ks1 +8 *ks2 +4 *ks0 *ks1 +4 *ks0 *ks2 +8 *ks1 *ks2 +4 *ks0 *ks1 *ks2 )//3 ))%ks3 ))),r0_mask &tmp2 &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp4 =tl .broadcast_to (tmp3 ,[XBLOCK ,R0_BLOCK ])
        tmp6 =_tmp5 +tmp4 
        _tmp5 =tl .where (r0_mask &xmask ,tmp6 ,_tmp5 )
    tmp5 =tl .sum (_tmp5 ,1 )[:,None ]
    tl .store (out_ptr0 +(x0 ),tmp5 ,xmask )

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
        buf1 =empty_strided_cuda ((1 ,2 +s0 ,2 +2 *s1 ,2 +2 *s2 ),(8 +4 *s0 +8 *s1 +8 *s2 +4 *s0 *s1 +4 *s0 *s2 +8 *s1 *s2 +4 *s0 *s1 *s2 ,4 +4 *s1 +4 *s2 +4 *s1 *s2 ,2 +2 *s2 ,1 ),torch .float32 )

        triton_poi_fused_native_dropout_0_xnumel =8 +4 *s0 +8 *s1 +8 *s2 +4 *s0 *s1 +4 *s0 *s2 +8 *s1 *s2 +4 *s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_native_dropout_0 [grid (triton_poi_fused_native_dropout_0_xnumel )](buf0 ,buf1 ,0 ,21780 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        2 +2 *s2 
        2 +2 *s1 
        4 +4 *s1 +4 *s2 +4 *s1 *s2 
        buf2 =buf1 ;del buf1 

        triton_poi_fused_native_dropout_1_xnumel =8 +4 *s0 +8 *s1 +8 *s2 +4 *s0 *s1 +4 *s0 *s2 +8 *s1 *s2 +4 *s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_native_dropout_1 [grid (triton_poi_fused_native_dropout_1_xnumel )](buf2 ,arg3_1 ,32 ,66 ,66 ,32 ,4356 ,3 ,21780 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        buf3 =empty_strided_cuda ((1 ,2 +s0 ,1 ,1 ),(2 +s0 ,1 ,2 +s0 ,2 +s0 ),torch .float32 )

        triton_poi_fused_bernoulli_2_xnumel =2 +s0 
        get_raw_stream (0 )
        triton_poi_fused_bernoulli_2 [grid (triton_poi_fused_bernoulli_2_xnumel )](buf0 ,buf3 ,1 ,5 ,XBLOCK =8 ,num_warps =1 ,num_stages =1 )
        del buf0 
        buf4 =empty_strided_cuda ((3 ,),(1 ,),torch .float32 )
        buf6 =empty_strided_cuda ((3 ,),(1 ,),torch .float32 )

        (10 +4 *s0 +8 *s1 +8 *s2 +4 *s0 *s1 +4 *s0 *s2 +8 *s1 *s2 +4 *s0 *s1 *s2 )//3 
        get_raw_stream (0 )
        triton_red_fused__to_copy_add_bernoulli_clamp_min_fill_hardtanh_mean_mul_native_dropout_ne_soft_margin_loss_sub_where_zeros_like_3 [grid (3 )](buf2 ,buf3 ,buf4 ,buf6 ,3 ,32 ,32 ,66 ,66 ,4356 ,3 ,7260 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        buf5 =empty_strided_cuda ((),(),torch .float32 )
        buf11 =buf5 ;del buf5 

        get_raw_stream (0 )
        triton_per_fused__to_copy_add_bernoulli_hardtanh_mul_native_dropout_soft_margin_loss_4 [grid (1 )](buf11 ,buf4 ,3 ,32 ,32 ,1 ,3 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        del buf4 
        buf7 =empty_strided_cuda ((),(),torch .float32 )
        buf12 =buf7 ;del buf7 

        get_raw_stream (0 )
        triton_per_fused__to_copy_add_bernoulli_hardtanh_mul_native_dropout_soft_margin_loss_4 [grid (1 )](buf12 ,buf6 ,3 ,32 ,32 ,1 ,3 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        4 +4 *s1 +4 *s2 +4 *s1 *s2 
        buf8 =buf2 ;del buf2 

        triton_poi_fused__to_copy_add_bernoulli_hardtanh_huber_loss_mul_native_dropout_ones_like_5_xnumel =8 +4 *s0 +8 *s1 +8 *s2 +4 *s0 *s1 +4 *s0 *s2 +8 *s1 *s2 +4 *s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused__to_copy_add_bernoulli_hardtanh_huber_loss_mul_native_dropout_ones_like_5 [grid (triton_poi_fused__to_copy_add_bernoulli_hardtanh_huber_loss_mul_native_dropout_ones_like_5_xnumel )](buf8 ,buf3 ,4356 ,21780 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf3 
        buf9 =buf6 ;del buf6 

        (10 +4 *s0 +8 *s1 +8 *s2 +4 *s0 *s1 +4 *s0 *s2 +8 *s1 *s2 +4 *s0 *s1 *s2 )//3 
        get_raw_stream (0 )
        triton_red_fused_huber_loss_6 [grid (3 )](buf8 ,buf9 ,3 ,32 ,32 ,66 ,66 ,4356 ,3 ,7260 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del buf8 
        buf10 =empty_strided_cuda ((),(),torch .float32 )
        buf13 =buf10 ;del buf10 

        get_raw_stream (0 )
        triton_per_fused__to_copy_add_bernoulli_hardtanh_mul_native_dropout_soft_margin_loss_4 [grid (1 )](buf13 ,buf9 ,3 ,32 ,32 ,1 ,3 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        del buf9 
    return (buf11 ,buf12 ,buf13 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =3 
    arg1_1 =32 
    arg2_1 =32 
    arg3_1 =rand_strided ((1 ,3 ,32 ,32 ),(3072 ,1024 ,32 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
