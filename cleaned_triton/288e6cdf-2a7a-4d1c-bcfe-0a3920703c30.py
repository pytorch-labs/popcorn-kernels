
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
def triton_red_fused_add_bernoulli_norm_sub_0 (in_ptr0 ,in_ptr1 ,out_ptr0 ,out_ptr1 ,out_ptr2 ,load_seed_offset ,ks1 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =x0 
    tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tl .store (out_ptr0 +(x0 ),tmp2 ,xmask )
    _tmp17 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    _tmp25 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_1 =r0_index 
        tmp3 =tl .load (in_ptr1 +(ks1 *x0 +(tl .where ((-1 )+ks1 +((-1 )*tl_math .abs (1 +((-1 )*ks1 )+tl_math .abs ((-2 )+r0_1 )))<0 ,(-1 )+((-1 )*tl_math .abs (1 +((-1 )*ks1 )+tl_math .abs ((-2 )+r0_1 )))+2 *ks1 ,(-1 )+ks1 +((-1 )*tl_math .abs (1 +((-1 )*ks1 )+tl_math .abs ((-2 )+r0_1 )))))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp10 =tl .load (in_ptr1 +(ks1 *x0 +(tl .where ((-1 )+ks1 +((-1 )*tl_math .abs (1 +((-1 )*ks1 )+tl_math .abs (r0_1 +(ks1 //2 ))))<0 ,(-1 )+((-1 )*tl_math .abs (1 +((-1 )*ks1 )+tl_math .abs (r0_1 +(ks1 //2 ))))+2 *ks1 ,(-1 )+ks1 +((-1 )*tl_math .abs (1 +((-1 )*ks1 )+tl_math .abs (r0_1 +(ks1 //2 ))))))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp19 =tl .load (in_ptr1 +(ks1 *x0 +(tl .where ((-1 )+ks1 +((-1 )*tl_math .abs (1 +((-1 )*ks1 )+tl_math .abs (1 +ks1 +((-1 )*r0_1 ))))<0 ,(-1 )+((-1 )*tl_math .abs (1 +((-1 )*ks1 )+tl_math .abs (1 +ks1 +((-1 )*r0_1 ))))+2 *ks1 ,(-1 )+ks1 +((-1 )*tl_math .abs (1 +((-1 )*ks1 )+tl_math .abs (1 +ks1 +((-1 )*r0_1 ))))))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp4 =0.5 
        tmp5 =tmp2 <tmp4 
        tmp6 =tmp5 .to (tl .float32 )
        tmp7 =2.0 
        tmp8 =tmp6 *tmp7 
        tmp9 =tmp3 *tmp8 
        tmp11 =tmp10 *tmp8 
        tmp12 =tmp9 -tmp11 
        tmp13 =1e-06 
        tmp14 =tmp12 +tmp13 
        tmp15 =tmp14 *tmp14 
        tmp16 =tl .broadcast_to (tmp15 ,[XBLOCK ,R0_BLOCK ])
        tmp18 =_tmp17 +tmp16 
        _tmp17 =tl .where (r0_mask &xmask ,tmp18 ,_tmp17 )
        tmp20 =tmp19 *tmp8 
        tmp21 =tmp9 -tmp20 
        tmp22 =tmp21 +tmp13 
        tmp23 =tmp22 *tmp22 
        tmp24 =tl .broadcast_to (tmp23 ,[XBLOCK ,R0_BLOCK ])
        tmp26 =_tmp25 +tmp24 
        _tmp25 =tl .where (r0_mask &xmask ,tmp26 ,_tmp25 )
    tmp17 =tl .sum (_tmp17 ,1 )[:,None ]
    tmp25 =tl .sum (_tmp25 ,1 )[:,None ]
    tl .store (out_ptr1 +(x0 ),tmp17 ,xmask )
    tl .store (out_ptr2 +(x0 ),tmp25 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused_add_clamp_min_mean_norm_sub_1 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,ks0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
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
    tmp12 =ks0 
    tmp13 =tmp12 .to (tl .float32 )
    tmp14 =tmp10 /tmp13 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp14 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused__log_softmax__to_copy_bernoulli_div_mul_nll_loss2d_forward_reflection_pad1d_2 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,ks0 ,load_seed_offset ,ks2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    _tmp9 =tl .full ([XBLOCK ,R0_BLOCK ],float ("-inf"),tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_1 =r0_index 
        tmp0 =tl .load (in_ptr0 +(ks0 *r0_1 +(tl .where ((-1 )+ks0 +((-1 )*tl_math .abs (1 +((-1 )*ks0 )+tl_math .abs ((-2 )+x0 )))<0 ,(-1 )+((-1 )*tl_math .abs (1 +((-1 )*ks0 )+tl_math .abs ((-2 )+x0 )))+2 *ks0 ,(-1 )+ks0 +((-1 )*tl_math .abs (1 +((-1 )*ks0 )+tl_math .abs ((-2 )+x0 )))))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp1 =tl .load (in_ptr1 +(r0_1 ),r0_mask ,eviction_policy ='evict_last',other =0.0 )
        tmp2 =0.5 
        tmp3 =tmp1 <tmp2 
        tmp4 =tmp3 .to (tl .float32 )
        tmp5 =2.0 
        tmp6 =tmp4 *tmp5 
        tmp7 =tmp0 *tmp6 
        tmp8 =tl .broadcast_to (tmp7 ,[XBLOCK ,R0_BLOCK ])
        tmp10 =triton_helpers .maximum (_tmp9 ,tmp8 )
        _tmp9 =tl .where (r0_mask &xmask ,tmp10 ,_tmp9 )
    tmp9 =triton_helpers .max2 (_tmp9 ,1 )[:,None ]
    _tmp22 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_1 =r0_index 
        tmp11 =tl .load (in_ptr0 +(ks0 *r0_1 +(tl .where ((-1 )+ks0 +((-1 )*tl_math .abs (1 +((-1 )*ks0 )+tl_math .abs ((-2 )+x0 )))<0 ,(-1 )+((-1 )*tl_math .abs (1 +((-1 )*ks0 )+tl_math .abs ((-2 )+x0 )))+2 *ks0 ,(-1 )+ks0 +((-1 )*tl_math .abs (1 +((-1 )*ks0 )+tl_math .abs ((-2 )+x0 )))))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp12 =tl .load (in_ptr1 +(r0_1 ),r0_mask ,eviction_policy ='evict_last',other =0.0 )
        tmp13 =0.5 
        tmp14 =tmp12 <tmp13 
        tmp15 =tmp14 .to (tl .float32 )
        tmp16 =2.0 
        tmp17 =tmp15 *tmp16 
        tmp18 =tmp11 *tmp17 
        tmp19 =tmp18 -tmp9 
        tmp20 =tl_math .exp (tmp19 )
        tmp21 =tl .broadcast_to (tmp20 ,[XBLOCK ,R0_BLOCK ])
        tmp23 =_tmp22 +tmp21 
        _tmp22 =tl .where (r0_mask &xmask ,tmp23 ,_tmp22 )
    tmp22 =tl .sum (_tmp22 ,1 )[:,None ]
    tmp24 =tl .load (in_ptr2 +load_seed_offset )
    tmp25 =x0 
    tmp26 =tl .full ([1 ,1 ],0 ,tl .int64 )
    tmp27 =ks2 
    tmp28 =triton_helpers .randint64 (tmp24 ,(tmp25 ).to (tl .uint32 ),tmp26 ,tmp27 )
    tmp29 =tl .full ([1 ,1 ],-100 ,tl .int64 )
    tmp30 =tmp28 !=tmp29 
    tmp31 =tl .where (tmp30 ,tmp28 ,tmp26 )
    tmp32 =tmp31 +tmp27 
    tmp33 =tmp31 <0 
    tmp34 =tl .where (tmp33 ,tmp32 ,tmp31 )
    tl .device_assert (((0 <=tmp34 )&(tmp34 <ks2 ))|~(xmask ),"index out of bounds: 0 <= tmp34 < ks2")
    tmp36 =tl .load (in_ptr0 +(ks0 *tmp34 +(tl .where ((-1 )+ks0 +((-1 )*tl_math .abs (1 +((-1 )*ks0 )+tl_math .abs ((-2 )+x0 )))<0 ,(-1 )+((-1 )*tl_math .abs (1 +((-1 )*ks0 )+tl_math .abs ((-2 )+x0 )))+2 *ks0 ,(-1 )+ks0 +((-1 )*tl_math .abs (1 +((-1 )*ks0 )+tl_math .abs ((-2 )+x0 )))))),xmask ,eviction_policy ='evict_last')
    tmp37 =tl .load (in_ptr1 +(tmp34 ),xmask ,eviction_policy ='evict_last')
    tmp38 =0.5 
    tmp39 =tmp37 <tmp38 
    tmp40 =tmp39 .to (tl .float32 )
    tmp41 =2.0 
    tmp42 =tmp40 *tmp41 
    tmp43 =tmp36 *tmp42 
    tmp44 =tmp43 -tmp9 
    tmp45 =tl_math .log (tmp22 )
    tmp46 =tmp44 -tmp45 
    tmp47 =-tmp46 
    tmp48 =0.0 
    tmp49 =tl .where (tmp30 ,tmp47 ,tmp48 )
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(x0 ),tmp49 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused_nll_loss2d_forward_3 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,load_seed_offset ,ks1 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
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
    _tmp13 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .int64 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_0 =r0_index 
        tmp4 =tl .load (in_ptr1 +load_seed_offset )
        tmp5 =r0_0 
        tmp6 =tl .full ([1 ,1 ],0 ,tl .int64 )
        tmp7 =ks1 
        tmp8 =triton_helpers .randint64 (tmp4 ,(tmp5 ).to (tl .uint32 ),tmp6 ,tmp7 )
        tmp9 =tl .full ([1 ,1 ],-100 ,tl .int64 )
        tmp10 =tmp8 !=tmp9 
        tmp11 =tmp10 .to (tl .int64 )
        tmp12 =tl .broadcast_to (tmp11 ,[XBLOCK ,R0_BLOCK ])
        tmp14 =_tmp13 +tmp12 
        _tmp13 =tl .where (r0_mask ,tmp14 ,_tmp13 )
    tmp13 =tl .sum (_tmp13 ,1 )[:,None ]
    tmp15 =tmp13 .to (tl .float32 )
    tmp16 =tmp2 /tmp15 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp16 ,None )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    assert_size_stride (arg2_1 ,(1 ,s0 ,s1 ),(s0 *s1 ,s1 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((2 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[2 ],out =buf0 )
        buf1 =empty_strided_cuda ((1 ,s0 ,1 ),(s0 ,1 ,s0 ),torch .float32 )
        buf7 =empty_strided_cuda ((1 ,s0 ),(s0 ,1 ),torch .float32 )
        buf8 =empty_strided_cuda ((1 ,s0 ),(s0 ,1 ),torch .float32 )

        2 +(s1 //2 )
        get_raw_stream (0 )
        triton_red_fused_add_bernoulli_norm_sub_0 [grid (s0 )](buf0 ,arg2_1 ,buf1 ,buf7 ,buf8 ,0 ,32 ,10 ,18 ,XBLOCK =1 ,R0_BLOCK =32 ,num_warps =2 ,num_stages =1 )
        buf9 =empty_strided_cuda ((),(),torch .float32 )
        buf11 =buf9 ;del buf9 

        get_raw_stream (0 )
        triton_red_fused_add_clamp_min_mean_norm_sub_1 [grid (1 )](buf11 ,buf7 ,buf8 ,10 ,1 ,10 ,XBLOCK =1 ,R0_BLOCK =16 ,num_warps =2 ,num_stages =1 )
        del buf7 
        del buf8 
        buf2 =empty_strided_cuda ((1 ,1 ,4 +s1 ),(4 +s1 ,4 +s1 ,1 ),torch .float32 )
        buf4 =buf2 ;del buf2 

        triton_red_fused__log_softmax__to_copy_bernoulli_div_mul_nll_loss2d_forward_reflection_pad1d_2_xnumel =4 +s1 
        get_raw_stream (0 )
        triton_red_fused__log_softmax__to_copy_bernoulli_div_mul_nll_loss2d_forward_reflection_pad1d_2 [grid (triton_red_fused__log_softmax__to_copy_bernoulli_div_mul_nll_loss2d_forward_reflection_pad1d_2_xnumel )](buf4 ,arg2_1 ,buf1 ,buf0 ,32 ,1 ,10 ,36 ,10 ,XBLOCK =1 ,R0_BLOCK =16 ,num_warps =2 ,num_stages =1 )
        del arg2_1 
        del buf1 
        buf5 =empty_strided_cuda ((),(),torch .float32 )
        buf10 =buf5 ;del buf5 

        4 +s1 
        get_raw_stream (0 )
        triton_red_fused_nll_loss2d_forward_3 [grid (1 )](buf10 ,buf4 ,buf0 ,1 ,10 ,1 ,36 ,XBLOCK =1 ,R0_BLOCK =64 ,num_warps =2 ,num_stages =1 )
        del buf0 
        del buf4 
    return (buf10 ,buf11 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =10 
    arg1_1 =32 
    arg2_1 =rand_strided ((1 ,10 ,32 ),(320 ,32 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
