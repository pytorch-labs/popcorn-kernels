
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
def triton_poi_fused_rand_like_0 (in_ptr0 ,out_ptr0 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
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
def triton_red_fused__softmax_1 (in_ptr0 ,out_ptr0 ,out_ptr1 ,ks0 ,ks1 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    _tmp2 =tl .full ([XBLOCK ,R0_BLOCK ],float ("-inf"),tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_1 =r0_index 
        tmp0 =tl .load (in_ptr0 +(x0 +16 *r0_1 +4 *ks0 *r0_1 +4 *ks1 *r0_1 +ks0 *ks1 *r0_1 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
        tmp3 =triton_helpers .maximum (_tmp2 ,tmp1 )
        _tmp2 =tl .where (r0_mask &xmask ,tmp3 ,_tmp2 )
    tmp2 =triton_helpers .max2 (_tmp2 ,1 )[:,None ]
    tl .store (out_ptr0 +(x0 ),tmp2 ,xmask )
    _tmp8 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_1 =r0_index 
        tmp4 =tl .load (in_ptr0 +(x0 +16 *r0_1 +4 *ks0 *r0_1 +4 *ks1 *r0_1 +ks0 *ks1 *r0_1 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp5 =tmp4 -tmp2 
        tmp6 =tl_math .exp (tmp5 )
        tmp7 =tl .broadcast_to (tmp6 ,[XBLOCK ,R0_BLOCK ])
        tmp9 =_tmp8 +tmp7 
        _tmp8 =tl .where (r0_mask &xmask ,tmp9 ,_tmp8 )
    tmp8 =tl .sum (_tmp8 ,1 )[:,None ]
    tl .store (out_ptr1 +(x0 ),tmp8 ,xmask )

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
def triton_poi_fused__to_copy_add_bernoulli_mul_replication_pad2d_3 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%ks1 )
    x2 =xindex //ks2 
    x3 =xindex 
    tmp0 =tl .load (in_ptr0 +(ks4 *(((-1 )+ks3 )*(((-1 )+ks3 )<=(((0 )*((0 )>=((-2 )+x1 ))+((-2 )+x1 )*(((-2 )+x1 )>(0 )))))+(((0 )*((0 )>=((-2 )+x1 ))+((-2 )+x1 )*(((-2 )+x1 )>(0 ))))*((((0 )*((0 )>=((-2 )+x1 ))+((-2 )+x1 )*(((-2 )+x1 )>(0 ))))<((-1 )+ks3 )))+ks3 *ks4 *x2 +(((-1 )+ks4 )*(((-1 )+ks4 )<=(((0 )*((0 )>=((-2 )+x0 ))+((-2 )+x0 )*(((-2 )+x0 )>(0 )))))+(((0 )*((0 )>=((-2 )+x0 ))+((-2 )+x0 )*(((-2 )+x0 )>(0 ))))*((((0 )*((0 )>=((-2 )+x0 ))+((-2 )+x0 )*(((-2 )+x0 )>(0 ))))<((-1 )+ks4 )))),xmask ,eviction_policy ='evict_last')
    tmp1 =tl .load (in_out_ptr0 +(x3 ),xmask ,eviction_policy ='evict_last')
    tmp15 =tl .load (in_ptr1 +(x3 ),xmask ,eviction_policy ='evict_last')
    tmp2 =0.5 
    tmp3 =tmp1 <tmp2 
    tmp4 =tmp3 .to (tl .float32 )
    tmp5 =0.8864048946659319 
    tmp6 =tmp4 *tmp5 
    tmp7 =tmp0 *tmp6 
    tmp8 =-1.0 
    tmp9 =tmp4 +tmp8 
    tmp10 =1.558387861036063 
    tmp11 =tmp9 *tmp10 
    tmp12 =0.7791939305180315 
    tmp13 =tmp11 +tmp12 
    tmp14 =tmp7 +tmp13 
    tmp16 =tmp15 <tmp2 
    tmp17 =tmp16 .to (tl .float32 )
    tmp18 =tmp17 *tmp5 
    tmp19 =tmp14 *tmp18 
    tmp20 =tmp17 +tmp8 
    tmp21 =tmp20 *tmp10 
    tmp22 =tmp21 +tmp12 
    tmp23 =tmp19 +tmp22 
    tl .store (in_out_ptr0 +(x3 ),tmp23 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused__log_softmax__softmax_mul_sub_sum_xlogy_4 (in_ptr0 ,in_ptr1 ,in_ptr2 ,in_ptr3 ,in_ptr4 ,in_ptr5 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =(xindex %ks0 )
    x1 =xindex //ks0 
    _tmp23 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    x3 =xindex 
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_2 =r0_index 
        tmp0 =tl .load (in_ptr0 +(4 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//ks0 )%ks1 ))+16 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//(16 +4 *ks3 +4 *ks4 +ks3 *ks4 ))%ks2 ))+ks4 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//ks0 )%ks1 ))+4 *ks3 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//(16 +4 *ks3 +4 *ks4 +ks3 *ks4 ))%ks2 ))+4 *ks4 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//(16 +4 *ks3 +4 *ks4 +ks3 *ks4 ))%ks2 ))+ks3 *ks4 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//(16 +4 *ks3 +4 *ks4 +ks3 *ks4 ))%ks2 ))+(((r0_2 +ks2 *x0 )%ks0 ))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp1 =tl .load (in_ptr1 +(4 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//ks0 )%ks1 ))+ks4 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//ks0 )%ks1 ))+(((r0_2 +ks2 *x0 )%ks0 ))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp4 =tl .load (in_ptr2 +(4 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//ks0 )%ks1 ))+ks4 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//ks0 )%ks1 ))+(((r0_2 +ks2 *x0 )%ks0 ))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp14 =tl .load (in_ptr3 +(4 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//ks0 )%ks1 ))+16 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//(16 +4 *ks3 +4 *ks4 +ks3 *ks4 ))%ks2 ))+ks4 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//ks0 )%ks1 ))+4 *ks3 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//(16 +4 *ks3 +4 *ks4 +ks3 *ks4 ))%ks2 ))+4 *ks4 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//(16 +4 *ks3 +4 *ks4 +ks3 *ks4 ))%ks2 ))+ks3 *ks4 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//(16 +4 *ks3 +4 *ks4 +ks3 *ks4 ))%ks2 ))+(((r0_2 +ks2 *x0 )%ks0 ))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp15 =tl .load (in_ptr4 +(4 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//ks0 )%ks1 ))+ks4 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//ks0 )%ks1 ))+(((r0_2 +ks2 *x0 )%ks0 ))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp17 =tl .load (in_ptr5 +(4 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//ks0 )%ks1 ))+ks4 *((((r0_2 +ks2 *x0 +4 *ks2 *x1 +ks2 *ks4 *x1 )//ks0 )%ks1 ))+(((r0_2 +ks2 *x0 )%ks0 ))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp2 =tmp0 -tmp1 
        tmp3 =tl_math .exp (tmp2 )
        tmp5 =tmp3 /tmp4 
        tmp6 =libdevice .isnan (tmp5 ).to (tl .int1 )
        tmp7 =0.0 
        tmp8 =tmp5 ==tmp7 
        tmp9 =tl_math .log (tmp5 )
        tmp10 =tmp5 *tmp9 
        tmp11 =tl .where (tmp8 ,tmp7 ,tmp10 )
        tmp12 =float ("nan")
        tmp13 =tl .where (tmp6 ,tmp12 ,tmp11 )
        tmp16 =tmp14 -tmp15 
        tmp18 =tl_math .log (tmp17 )
        tmp19 =tmp16 -tmp18 
        tmp20 =tmp5 *tmp19 
        tmp21 =tmp13 -tmp20 
        tmp22 =tl .broadcast_to (tmp21 ,[XBLOCK ,R0_BLOCK ])
        tmp24 =_tmp23 +tmp22 
        _tmp23 =tl .where (r0_mask &xmask ,tmp24 ,_tmp23 )
    tmp23 =tl .sum (_tmp23 ,1 )[:,None ]
    tl .store (out_ptr0 +(x3 ),tmp23 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused__log_softmax__softmax_div_mul_sub_sum_xlogy_5 (in_out_ptr0 ,in_ptr0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
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
    tmp4 =1.0 
    tmp5 =tmp2 *tmp4 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp5 ,None )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((3 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[3 ],out =buf0 )
        buf1 =empty_strided_cuda ((1 ,s0 ,4 +s1 ,4 +s2 ),(16 *s0 +4 *s0 *s1 +4 *s0 *s2 +s0 *s1 *s2 ,16 +4 *s1 +4 *s2 +s1 *s2 ,4 +s2 ,1 ),torch .float32 )

        triton_poi_fused_rand_like_0_xnumel =16 *s0 +4 *s0 *s1 +4 *s0 *s2 +s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_rand_like_0 [grid (triton_poi_fused_rand_like_0_xnumel )](buf0 ,buf1 ,0 ,13872 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        buf2 =empty_strided_cuda ((1 ,1 ,4 +s1 ,4 +s2 ),(16 +4 *s1 +4 *s2 +s1 *s2 ,16 +4 *s1 +4 *s2 +s1 *s2 ,4 +s2 ,1 ),torch .float32 )
        buf3 =empty_strided_cuda ((1 ,1 ,4 +s1 ,4 +s2 ),(16 +4 *s1 +4 *s2 +s1 *s2 ,16 +4 *s1 +4 *s2 +s1 *s2 ,4 +s2 ,1 ),torch .float32 )

        triton_red_fused__softmax_1_xnumel =16 +4 *s1 +4 *s2 +s1 *s2 
        get_raw_stream (0 )
        triton_red_fused__softmax_1 [grid (triton_red_fused__softmax_1_xnumel )](buf1 ,buf2 ,buf3 ,64 ,64 ,4624 ,3 ,XBLOCK =64 ,R0_BLOCK =4 ,num_warps =2 ,num_stages =1 )
        buf4 =empty_strided_cuda ((1 ,s0 ,4 +s1 ,4 +s2 ),(16 *s0 +4 *s0 *s1 +4 *s0 *s2 +s0 *s1 *s2 ,16 +4 *s1 +4 *s2 +s1 *s2 ,4 +s2 ,1 ),torch .float32 )

        triton_poi_fused_rand_like_0_xnumel =16 *s0 +4 *s0 *s1 +4 *s0 *s2 +s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_rand_like_0 [grid (triton_poi_fused_rand_like_0_xnumel )](buf0 ,buf4 ,0 ,13872 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        buf5 =empty_strided_cuda ((1 ,s0 ,4 +s1 ,4 +s2 ),(16 *s0 +4 *s0 *s1 +4 *s0 *s2 +s0 *s1 *s2 ,16 +4 *s1 +4 *s2 +s1 *s2 ,4 +s2 ,1 ),torch .float32 )

        triton_poi_fused_bernoulli_2_xnumel =16 *s0 +4 *s0 *s1 +4 *s0 *s2 +s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_bernoulli_2 [grid (triton_poi_fused_bernoulli_2_xnumel )](buf0 ,buf5 ,1 ,13872 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf0 
        4 +s2 
        4 +s1 
        16 +4 *s1 +4 *s2 +s1 *s2 
        buf6 =buf4 ;del buf4 

        triton_poi_fused__to_copy_add_bernoulli_mul_replication_pad2d_3_xnumel =16 *s0 +4 *s0 *s1 +4 *s0 *s2 +s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused__to_copy_add_bernoulli_mul_replication_pad2d_3 [grid (triton_poi_fused__to_copy_add_bernoulli_mul_replication_pad2d_3_xnumel )](buf6 ,arg3_1 ,buf5 ,68 ,68 ,4624 ,64 ,64 ,13872 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        del buf5 
        buf7 =empty_strided_cuda ((1 ,1 ,4 +s1 ,4 +s2 ),(16 +4 *s1 +4 *s2 +s1 *s2 ,16 +4 *s1 +4 *s2 +s1 *s2 ,4 +s2 ,1 ),torch .float32 )
        buf8 =empty_strided_cuda ((1 ,1 ,4 +s1 ,4 +s2 ),(16 +4 *s1 +4 *s2 +s1 *s2 ,16 +4 *s1 +4 *s2 +s1 *s2 ,4 +s2 ,1 ),torch .float32 )

        triton_red_fused__softmax_1_xnumel =16 +4 *s1 +4 *s2 +s1 *s2 
        get_raw_stream (0 )
        triton_red_fused__softmax_1 [grid (triton_red_fused__softmax_1_xnumel )](buf6 ,buf7 ,buf8 ,64 ,64 ,4624 ,3 ,XBLOCK =64 ,R0_BLOCK =4 ,num_warps =2 ,num_stages =1 )
        buf9 =empty_strided_cuda ((1 ,1 ,4 +s1 ,4 +s2 ),(16 +4 *s1 +4 *s2 +s1 *s2 ,16 +4 *s1 +4 *s2 +s1 *s2 ,4 +s2 ,1 ),torch .float32 )

        triton_red_fused__log_softmax__softmax_mul_sub_sum_xlogy_4_xnumel =16 +4 *s1 +4 *s2 +s1 *s2 
        get_raw_stream (0 )
        triton_red_fused__log_softmax__softmax_mul_sub_sum_xlogy_4 [grid (triton_red_fused__log_softmax__softmax_mul_sub_sum_xlogy_4_xnumel )](buf1 ,buf2 ,buf3 ,buf6 ,buf7 ,buf8 ,buf9 ,68 ,68 ,3 ,64 ,64 ,4624 ,3 ,XBLOCK =1 ,R0_BLOCK =4 ,num_warps =2 ,num_stages =1 )
        del buf1 
        del buf2 
        del buf3 
        del buf7 
        del buf8 
        buf10 =empty_strided_cuda ((),(),torch .float32 )
        buf11 =buf10 ;del buf10 

        16 +4 *s1 +4 *s2 +s1 *s2 
        get_raw_stream (0 )
        triton_red_fused__log_softmax__softmax_div_mul_sub_sum_xlogy_5 [grid (1 )](buf11 ,buf9 ,1 ,4624 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del buf9 
    return (buf6 ,buf11 ,)

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
