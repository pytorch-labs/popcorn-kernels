
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
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_view_0 (in_out_ptr1 ,in_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x1 =((xindex //ks1 )%ks2 )
    x0 =(xindex %ks1 )
    x2 =xindex //ks4 
    x5 =xindex 
    tmp0 =tl .full ([1 ],-1.0 ,tl .float64 )
    tmp1 =ks0 
    tmp2 =tmp1 .to (tl .float64 )
    tmp3 =tmp0 +tmp2 
    tmp4 =2.0 
    tmp5 =tmp1 .to (tl .float32 )
    tmp6 =tmp4 *tmp5 
    tmp7 =tmp6 .to (tl .float64 )
    tmp8 =tmp0 +tmp7 
    tmp9 =tmp3 /tmp8 
    tmp10 =tmp9 .to (tl .float32 )
    tmp11 =x1 
    tmp12 =tmp11 .to (tl .float32 )
    tmp13 =tmp12 *tmp10 
    tmp14 =0.0 
    tmp15 =triton_helpers .maximum (tmp13 ,tmp14 )
    tmp16 =tmp15 .to (tl .int64 )
    tmp17 =ks3 
    tmp18 =tmp17 .to (tl .float64 )
    tmp19 =tmp0 +tmp18 
    tmp20 =tmp17 .to (tl .float32 )
    tmp21 =tmp4 *tmp20 
    tmp22 =tmp21 .to (tl .float64 )
    tmp23 =tmp0 +tmp22 
    tmp24 =tmp19 /tmp23 
    tmp25 =tmp24 .to (tl .float32 )
    tmp26 =x0 
    tmp27 =tmp26 .to (tl .float32 )
    tmp28 =tmp27 *tmp25 
    tmp29 =triton_helpers .maximum (tmp28 ,tmp14 )
    tmp30 =tmp29 .to (tl .int64 )
    tmp31 =tl .load (in_ptr0 +(tmp30 +ks3 *tmp16 +ks0 *ks3 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp32 =tl .full ([1 ],1 ,tl .int64 )
    tmp33 =tmp16 +tmp32 
    tmp34 =(-1 )+ks0 
    tmp35 =triton_helpers .minimum (tmp33 ,tmp34 )
    tmp36 =tl .load (in_ptr0 +(tmp30 +ks3 *tmp35 +ks0 *ks3 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp37 =tmp30 +tmp32 
    tmp38 =(-1 )+ks3 
    tmp39 =triton_helpers .minimum (tmp37 ,tmp38 )
    tmp40 =tl .load (in_ptr0 +(tmp39 +ks3 *tmp35 +ks0 *ks3 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp41 =tmp40 -tmp36 
    tmp42 =tl .load (in_ptr0 +(tmp39 +ks3 *tmp16 +ks0 *ks3 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp43 =tmp42 -tmp31 
    tmp44 =tmp30 .to (tl .float32 )
    tmp45 =tmp29 -tmp44 
    tmp46 =triton_helpers .maximum (tmp45 ,tmp14 )
    tmp47 =1.0 
    tmp48 =triton_helpers .minimum (tmp46 ,tmp47 )
    tmp49 =tmp41 *tmp48 
    tmp50 =tmp36 +tmp49 
    tmp51 =tmp43 *tmp48 
    tmp52 =tmp31 +tmp51 
    tmp53 =tmp50 -tmp52 
    tmp54 =tmp16 .to (tl .float32 )
    tmp55 =tmp15 -tmp54 
    tmp56 =triton_helpers .maximum (tmp55 ,tmp14 )
    tmp57 =triton_helpers .minimum (tmp56 ,tmp47 )
    tmp58 =tmp53 *tmp57 
    tmp59 =tmp52 +tmp58 
    tl .store (in_out_ptr1 +(x5 ),tmp59 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused__softmax_relu_1 (in_ptr0 ,out_ptr0 ,out_ptr1 ,out_ptr2 ,out_ptr3 ,ks0 ,ks1 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    R0_BLOCK :tl .constexpr =128 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    r0_mask =r0_index <r0_numel 
    r0_1 =r0_index 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(x0 +4 *ks0 *ks1 *r0_1 ),r0_mask &xmask ,other =0.0 )
    tmp1 =tl .full ([1 ,1 ],0 ,tl .int32 )
    tmp2 =triton_helpers .maximum (tmp1 ,tmp0 )
    tmp3 =tl .broadcast_to (tmp2 ,[XBLOCK ,R0_BLOCK ])
    tmp5 =tl .where (r0_mask &xmask ,tmp3 ,float ("-inf"))
    tmp6 =triton_helpers .max2 (tmp5 ,1 )[:,None ]
    tmp7 =tmp2 -tmp6 
    tmp8 =tl_math .exp (tmp7 )
    tmp9 =tl .broadcast_to (tmp8 ,[XBLOCK ,R0_BLOCK ])
    tmp11 =tl .where (r0_mask &xmask ,tmp9 ,0 )
    tmp12 =tl .sum (tmp11 ,1 )[:,None ]
    tmp13 =tmp8 /tmp12 
    tmp14 =triton_helpers .maximum (tmp1 ,tmp13 )
    tmp15 =tl .broadcast_to (tmp14 ,[XBLOCK ,R0_BLOCK ])
    tmp17 =tl .where (r0_mask &xmask ,tmp15 ,float ("-inf"))
    tmp18 =triton_helpers .max2 (tmp17 ,1 )[:,None ]
    tmp19 =tmp14 -tmp18 
    tmp20 =tl_math .exp (tmp19 )
    tmp21 =tl .broadcast_to (tmp20 ,[XBLOCK ,R0_BLOCK ])
    tmp23 =tl .where (r0_mask &xmask ,tmp21 ,0 )
    tmp24 =tl .sum (tmp23 ,1 )[:,None ]
    tl .store (out_ptr0 +(x0 ),tmp6 ,xmask )
    tl .store (out_ptr1 +(x0 ),tmp12 ,xmask )
    tl .store (out_ptr2 +(x0 ),tmp18 ,xmask )
    tl .store (out_ptr3 +(x0 ),tmp24 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__softmax_relu_smooth_l1_loss_2 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,in_ptr3 ,ks0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x2 =xindex 
    x0 =(xindex %ks0 )
    tmp0 =tl .load (in_out_ptr0 +(x2 ),xmask ,eviction_policy ='evict_last')
    tmp3 =tl .load (in_ptr0 +(x0 ),xmask ,eviction_policy ='evict_last')
    tmp6 =tl .load (in_ptr1 +(x0 ),xmask ,eviction_policy ='evict_last')
    tmp9 =tl .load (in_ptr2 +(x0 ),xmask ,eviction_policy ='evict_last')
    tmp12 =tl .load (in_ptr3 +(x0 ),xmask ,eviction_policy ='evict_last')
    tmp1 =tl .full ([1 ],0 ,tl .int32 )
    tmp2 =triton_helpers .maximum (tmp1 ,tmp0 )
    tmp4 =tmp2 -tmp3 
    tmp5 =tl_math .exp (tmp4 )
    tmp7 =tmp5 /tmp6 
    tmp8 =triton_helpers .maximum (tmp1 ,tmp7 )
    tmp10 =tmp8 -tmp9 
    tmp11 =tl_math .exp (tmp10 )
    tmp13 =tmp11 /tmp12 
    tmp14 =triton_helpers .maximum (tmp1 ,tmp13 )
    tl .store (in_out_ptr0 +(x2 ),tmp14 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused_huber_loss_smooth_l1_loss_3 (in_ptr0 ,out_ptr0 ,out_ptr1 ,ks0 ,ks1 ,ks2 ,ks3 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    R0_BLOCK :tl .constexpr =128 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    r0_mask =r0_index <r0_numel 
    r0_2 =r0_index 
    x0 =(xindex %ks0 )
    x1 =xindex //ks0 
    x3 =xindex 
    tmp0 =tl .load (in_ptr0 +(2 *ks3 *((((r0_2 +ks1 *x0 +2 *ks1 *ks3 *x1 )//ks0 )%(2 *ks1 *ks2 )))+(((r0_2 +ks1 *x0 )%ks0 ))),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp1 =tl_math .abs (tmp0 )
    tmp2 =1.0 
    tmp3 =tmp1 <tmp2 
    tmp4 =0.5 
    tmp5 =tmp1 *tmp4 
    tmp6 =tmp5 *tmp1 
    tmp7 =tmp1 -tmp4 
    tmp8 =tmp7 *tmp2 
    tmp9 =tl .where (tmp3 ,tmp6 ,tmp8 )
    tmp10 =tl .broadcast_to (tmp9 ,[XBLOCK ,R0_BLOCK ])
    tmp12 =tl .where (r0_mask &xmask ,tmp10 ,0 )
    tmp13 =tl .sum (tmp12 ,1 )[:,None ]
    tmp14 =tmp1 *tmp1 
    tmp15 =tmp14 *tmp4 
    tmp16 =tmp15 *tmp2 
    tmp17 =tl .where (tmp3 ,tmp16 ,tmp7 )
    tmp18 =tl .broadcast_to (tmp17 ,[XBLOCK ,R0_BLOCK ])
    tmp20 =tl .where (r0_mask &xmask ,tmp18 ,0 )
    tmp21 =tl .sum (tmp20 ,1 )[:,None ]
    tl .store (out_ptr0 +(x3 ),tmp13 ,xmask )
    tl .store (out_ptr1 +(x3 ),tmp21 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused_add_huber_loss_smooth_l1_loss_4 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,ks0 ,ks1 ,ks2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
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
    _tmp6 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_0 =r0_index 
        tmp4 =tl .load (in_ptr1 +(r0_0 ),r0_mask ,eviction_policy ='evict_first',other =0.0 )
        tmp5 =tl .broadcast_to (tmp4 ,[XBLOCK ,R0_BLOCK ])
        tmp7 =_tmp6 +tmp5 
        _tmp6 =tl .where (r0_mask ,tmp7 ,_tmp6 )
    tmp6 =tl .sum (_tmp6 ,1 )[:,None ]
    tmp8 =4 *ks0 *ks1 *ks2 
    tmp9 =tmp8 .to (tl .float32 )
    tmp10 =tmp2 /tmp9 
    tmp11 =tmp6 /tmp9 
    tmp12 =tmp10 +tmp11 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp12 ,None )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        2 *s2 
        2 *s1 
        4 *s1 *s2 
        buf2 =empty_strided_cuda ((1 ,s0 ,2 *s1 ,2 *s2 ),(4 *s0 *s1 *s2 ,4 *s1 *s2 ,2 *s2 ,1 ),torch .float32 )
        buf5 =buf2 ;del buf2 

        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_view_0_xnumel =4 *s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_view_0 [grid (triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_view_0_xnumel )](buf5 ,arg3_1 ,32 ,64 ,64 ,32 ,4096 ,12288 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        buf6 =empty_strided_cuda ((1 ,1 ,2 *s1 ,2 *s2 ),(4 *s1 *s2 ,4 *s1 *s2 ,2 *s2 ,1 ),torch .float32 )
        buf7 =empty_strided_cuda ((1 ,1 ,2 *s1 ,2 *s2 ),(4 *s1 *s2 ,4 *s1 *s2 ,2 *s2 ,1 ),torch .float32 )
        buf8 =empty_strided_cuda ((1 ,1 ,2 *s1 ,2 *s2 ),(4 *s1 *s2 ,4 *s1 *s2 ,2 *s2 ,1 ),torch .float32 )
        buf9 =empty_strided_cuda ((1 ,1 ,2 *s1 ,2 *s2 ),(4 *s1 *s2 ,4 *s1 *s2 ,2 *s2 ,1 ),torch .float32 )

        triton_per_fused__softmax_relu_1_xnumel =4 *s1 *s2 
        get_raw_stream (0 )
        triton_per_fused__softmax_relu_1 [grid (triton_per_fused__softmax_relu_1_xnumel )](buf5 ,buf6 ,buf7 ,buf8 ,buf9 ,32 ,32 ,4096 ,3 ,XBLOCK =8 ,num_warps =2 ,num_stages =1 )
        buf10 =buf5 ;del buf5 

        triton_poi_fused__softmax_relu_smooth_l1_loss_2_xnumel =4 *s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused__softmax_relu_smooth_l1_loss_2 [grid (triton_poi_fused__softmax_relu_smooth_l1_loss_2_xnumel )](buf10 ,buf6 ,buf7 ,buf8 ,buf9 ,4096 ,12288 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del buf6 
        del buf7 
        buf11 =buf9 ;del buf9 
        buf13 =buf8 ;del buf8 

        triton_per_fused_huber_loss_smooth_l1_loss_3_xnumel =4 *s1 *s2 
        get_raw_stream (0 )
        triton_per_fused_huber_loss_smooth_l1_loss_3 [grid (triton_per_fused_huber_loss_smooth_l1_loss_3_xnumel )](buf10 ,buf11 ,buf13 ,64 ,3 ,32 ,32 ,4096 ,3 ,XBLOCK =8 ,num_warps =2 ,num_stages =1 )
        del buf10 
        buf12 =empty_strided_cuda ((),(),torch .float32 )
        buf15 =buf12 ;del buf12 

        4 *s1 *s2 
        get_raw_stream (0 )
        triton_red_fused_add_huber_loss_smooth_l1_loss_4 [grid (1 )](buf15 ,buf11 ,buf13 ,3 ,32 ,32 ,1 ,4096 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del buf11 
        del buf13 
    return (buf15 ,)

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
