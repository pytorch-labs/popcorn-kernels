
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
def triton_poi_fused_constant_pad_nd_hardtanh_max_pool2d_with_indices_tanh_0 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,ks6 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x5 =xindex //ks0 
    x1 =((xindex //ks2 )%ks3 )
    x0 =(xindex %ks2 )
    x2 =xindex //ks6 
    x6 =xindex 
    tmp0 =(-1 )+x5 
    tmp1 =tl .full ([1 ],0 ,tl .int64 )
    tmp2 =tmp0 >=tmp1 
    tmp3 =ks1 
    tmp4 =tmp0 <tmp3 
    tmp5 =(-1 )+x1 
    tmp6 =tmp5 >=tmp1 
    tmp7 =ks4 //2 
    tmp8 =tmp5 <tmp7 
    tmp9 =(-1 )+x0 
    tmp10 =tmp9 >=tmp1 
    tmp11 =ks5 //2 
    tmp12 =tmp9 <tmp11 
    tmp13 =tmp2 &tmp4 
    tmp14 =tmp13 &tmp6 
    tmp15 =tmp14 &tmp8 
    tmp16 =tmp15 &tmp10 
    tmp17 =tmp16 &tmp12 
    tmp18 =tl .load (in_ptr0 +((-2 )+((-2 )*ks5 )+2 *x0 +((-1 )*ks4 *ks5 )+2 *ks5 *x1 +ks4 *ks5 *x2 ),tmp17 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp19 =tl .load (in_ptr0 +((-1 )+((-2 )*ks5 )+2 *x0 +((-1 )*ks4 *ks5 )+2 *ks5 *x1 +ks4 *ks5 *x2 ),tmp17 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp20 =triton_helpers .maximum (tmp19 ,tmp18 )
    tmp21 =tl .load (in_ptr0 +((-2 )+((-1 )*ks5 )+2 *x0 +((-1 )*ks4 *ks5 )+2 *ks5 *x1 +ks4 *ks5 *x2 ),tmp17 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp22 =triton_helpers .maximum (tmp21 ,tmp20 )
    tmp23 =tl .load (in_ptr0 +((-1 )+((-1 )*ks5 )+2 *x0 +((-1 )*ks4 *ks5 )+2 *ks5 *x1 +ks4 *ks5 *x2 ),tmp17 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp24 =triton_helpers .maximum (tmp23 ,tmp22 )
    tmp25 =tl .full (tmp24 .shape ,0.0 ,tmp24 .dtype )
    tmp26 =tl .where (tmp17 ,tmp24 ,tmp25 )
    tmp27 =-1.0 
    tmp28 =triton_helpers .maximum (tmp26 ,tmp27 )
    tmp29 =1.0 
    tmp30 =triton_helpers .minimum (tmp28 ,tmp29 )
    tmp31 =libdevice .tanh (tmp30 )
    tl .store (out_ptr0 +(x6 ),tmp31 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_constant_pad_nd_hardtanh_max_pool2d_with_indices_tanh_1 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,ks6 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x5 =xindex //ks0 
    x1 =((xindex //ks2 )%ks3 )
    x0 =(xindex %ks2 )
    x2 =xindex //ks6 
    x6 =xindex 
    tmp0 =(-1 )+x5 
    tmp1 =tl .full ([1 ],0 ,tl .int64 )
    tmp2 =tmp0 >=tmp1 
    tmp3 =2 +ks1 
    tmp4 =tmp0 <tmp3 
    tmp5 =(-1 )+x1 
    tmp6 =tmp5 >=tmp1 
    tmp7 =1 +(ks4 //4 )
    tmp8 =tmp5 <tmp7 
    tmp9 =(-1 )+x0 
    tmp10 =tmp9 >=tmp1 
    tmp11 =1 +(ks5 //4 )
    tmp12 =tmp9 <tmp11 
    tmp13 =tmp2 &tmp4 
    tmp14 =tmp13 &tmp6 
    tmp15 =tmp14 &tmp8 
    tmp16 =tmp15 &tmp10 
    tmp17 =tmp16 &tmp12 
    tmp18 =tl .load (in_ptr0 +((-10 )+((-4 )*(ks5 //2 ))+((-2 )*(ks4 //2 ))+2 *x0 +4 *x1 +4 *x2 +((-1 )*(ks4 //2 )*(ks5 //2 ))+2 *x1 *(ks5 //2 )+2 *x2 *(ks4 //2 )+2 *x2 *(ks5 //2 )+x2 *(ks4 //2 )*(ks5 //2 )),tmp17 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp19 =tl .load (in_ptr0 +((-9 )+((-4 )*(ks5 //2 ))+((-2 )*(ks4 //2 ))+2 *x0 +4 *x1 +4 *x2 +((-1 )*(ks4 //2 )*(ks5 //2 ))+2 *x1 *(ks5 //2 )+2 *x2 *(ks4 //2 )+2 *x2 *(ks5 //2 )+x2 *(ks4 //2 )*(ks5 //2 )),tmp17 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp20 =triton_helpers .maximum (tmp19 ,tmp18 )
    tmp21 =tl .load (in_ptr0 +((-8 )+((-3 )*(ks5 //2 ))+((-2 )*(ks4 //2 ))+2 *x0 +4 *x1 +4 *x2 +((-1 )*(ks4 //2 )*(ks5 //2 ))+2 *x1 *(ks5 //2 )+2 *x2 *(ks4 //2 )+2 *x2 *(ks5 //2 )+x2 *(ks4 //2 )*(ks5 //2 )),tmp17 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp22 =triton_helpers .maximum (tmp21 ,tmp20 )
    tmp23 =tl .load (in_ptr0 +((-7 )+((-3 )*(ks5 //2 ))+((-2 )*(ks4 //2 ))+2 *x0 +4 *x1 +4 *x2 +((-1 )*(ks4 //2 )*(ks5 //2 ))+2 *x1 *(ks5 //2 )+2 *x2 *(ks4 //2 )+2 *x2 *(ks5 //2 )+x2 *(ks4 //2 )*(ks5 //2 )),tmp17 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp24 =triton_helpers .maximum (tmp23 ,tmp22 )
    tmp25 =tl .full (tmp24 .shape ,0.0 ,tmp24 .dtype )
    tmp26 =tl .where (tmp17 ,tmp24 ,tmp25 )
    tmp27 =-1.0 
    tmp28 =triton_helpers .maximum (tmp26 ,tmp27 )
    tmp29 =1.0 
    tmp30 =triton_helpers .minimum (tmp28 ,tmp29 )
    tmp31 =libdevice .tanh (tmp30 )
    tl .store (out_ptr0 +(x6 ),tmp31 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused_add_clamp_min_div_exp_huber_loss_mean_mul_norm_ones_like_soft_margin_loss_sub_zeros_like_2 (in_out_ptr0 ,in_ptr0 ,ks0 ,ks1 ,ks2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    _tmp6 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    _tmp19 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    _tmp26 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_0 =r0_index 
        tmp0 =tl .load (in_ptr0 +(r0_0 ),r0_mask ,eviction_policy ='evict_first',other =0.0 )
        tmp1 =tl_math .exp (tmp0 )
        tmp2 =1.0 
        tmp3 =tmp2 *tmp0 
        tmp4 =tmp1 -tmp3 
        tmp5 =tl .broadcast_to (tmp4 ,[XBLOCK ,R0_BLOCK ])
        tmp7 =_tmp6 +tmp5 
        _tmp6 =tl .where (r0_mask ,tmp7 ,_tmp6 )
        tmp8 =0.0 
        tmp9 =tmp0 -tmp8 
        tmp10 =tl_math .abs (tmp9 )
        tmp11 =tmp10 <tmp2 
        tmp12 =0.5 
        tmp13 =tmp10 *tmp12 
        tmp14 =tmp13 *tmp10 
        tmp15 =tmp10 -tmp12 
        tmp16 =tmp15 *tmp2 
        tmp17 =tl .where (tmp11 ,tmp14 ,tmp16 )
        tmp18 =tl .broadcast_to (tmp17 ,[XBLOCK ,R0_BLOCK ])
        tmp20 =_tmp19 +tmp18 
        _tmp19 =tl .where (r0_mask ,tmp20 ,_tmp19 )
        tmp21 =-tmp0 
        tmp22 =tmp21 *tmp2 
        tmp23 =tl_math .exp (tmp22 )
        tmp24 =libdevice .log1p (tmp23 )
        tmp25 =tl .broadcast_to (tmp24 ,[XBLOCK ,R0_BLOCK ])
        tmp27 =_tmp26 +tmp25 
        _tmp26 =tl .where (r0_mask ,tmp27 ,_tmp26 )
    tmp6 =tl .sum (_tmp6 ,1 )[:,None ]
    tmp19 =tl .sum (_tmp19 ,1 )[:,None ]
    tmp26 =tl .sum (_tmp26 ,1 )[:,None ]
    tmp28 =0.0 
    tmp29 =tmp28 /tmp28 
    tmp30 =36 +9 *ks0 +12 *(ks1 //4 )+12 *(ks2 //4 )+3 *ks0 *(ks1 //4 )+3 *ks0 *(ks2 //4 )+4 *(ks1 //4 )*(ks2 //4 )+ks0 *(ks1 //4 )*(ks2 //4 )
    tmp31 =tmp30 .to (tl .float32 )
    tmp32 =tmp6 /tmp31 
    tmp33 =tmp29 +tmp32 
    tmp34 =tmp19 /tmp31 
    tmp35 =tmp33 +tmp34 
    tmp36 =tmp26 /tmp31 
    tmp37 =tmp35 +tmp36 
    tmp38 =0.25 
    tmp39 =tmp37 *tmp38 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp39 ,None )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        4 +2 *(s1 //2 )+2 *(s2 //2 )+(s1 //2 )*(s2 //2 )
        2 +(s2 //2 )
        2 +(s1 //2 )
        4 +2 *(s1 //2 )+2 *(s2 //2 )+(s1 //2 )*(s2 //2 )
        buf0 =empty_strided_cuda ((1 ,2 +s0 ,2 +(s1 //2 ),2 +(s2 //2 )),(8 +4 *s0 +4 *(s1 //2 )+4 *(s2 //2 )+2 *s0 *(s1 //2 )+2 *s0 *(s2 //2 )+2 *(s1 //2 )*(s2 //2 )+s0 *(s1 //2 )*(s2 //2 ),4 +2 *(s1 //2 )+2 *(s2 //2 )+(s1 //2 )*(s2 //2 ),2 +(s2 //2 ),1 ),torch .float32 )

        triton_poi_fused_constant_pad_nd_hardtanh_max_pool2d_with_indices_tanh_0_xnumel =8 +4 *s0 +4 *(s1 //2 )+4 *(s2 //2 )+2 *s0 *(s1 //2 )+2 *s0 *(s2 //2 )+2 *(s1 //2 )*(s2 //2 )+s0 *(s1 //2 )*(s2 //2 )
        get_raw_stream (0 )
        triton_poi_fused_constant_pad_nd_hardtanh_max_pool2d_with_indices_tanh_0 [grid (triton_poi_fused_constant_pad_nd_hardtanh_max_pool2d_with_indices_tanh_0_xnumel )](arg3_1 ,buf0 ,1156 ,3 ,34 ,34 ,64 ,64 ,1156 ,5780 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        9 +3 *(s1 //4 )+3 *(s2 //4 )+(s1 //4 )*(s2 //4 )
        3 +(s2 //4 )
        3 +(s1 //4 )
        9 +3 *(s1 //4 )+3 *(s2 //4 )+(s1 //4 )*(s2 //4 )
        buf1 =empty_strided_cuda ((1 ,4 +s0 ,3 +(s1 //4 ),3 +(s2 //4 )),(36 +9 *s0 +12 *(s1 //4 )+12 *(s2 //4 )+3 *s0 *(s1 //4 )+3 *s0 *(s2 //4 )+4 *(s1 //4 )*(s2 //4 )+s0 *(s1 //4 )*(s2 //4 ),9 +3 *(s1 //4 )+3 *(s2 //4 )+(s1 //4 )*(s2 //4 ),3 +(s2 //4 ),1 ),torch .float32 )

        triton_poi_fused_constant_pad_nd_hardtanh_max_pool2d_with_indices_tanh_1_xnumel =36 +9 *s0 +12 *(s1 //4 )+12 *(s2 //4 )+3 *s0 *(s1 //4 )+3 *s0 *(s2 //4 )+4 *(s1 //4 )*(s2 //4 )+s0 *(s1 //4 )*(s2 //4 )
        get_raw_stream (0 )
        triton_poi_fused_constant_pad_nd_hardtanh_max_pool2d_with_indices_tanh_1 [grid (triton_poi_fused_constant_pad_nd_hardtanh_max_pool2d_with_indices_tanh_1_xnumel )](buf0 ,buf1 ,361 ,3 ,19 ,19 ,64 ,64 ,361 ,2527 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf0 
        buf4 =empty_strided_cuda ((),(),torch .float32 )
        buf7 =buf4 ;del buf4 

        36 +9 *s0 +12 *(s1 //4 )+12 *(s2 //4 )+3 *s0 *(s1 //4 )+3 *s0 *(s2 //4 )+4 *(s1 //4 )*(s2 //4 )+s0 *(s1 //4 )*(s2 //4 )
        get_raw_stream (0 )
        triton_red_fused_add_clamp_min_div_exp_huber_loss_mean_mul_norm_ones_like_soft_margin_loss_sub_zeros_like_2 [grid (1 )](buf7 ,buf1 ,3 ,64 ,64 ,1 ,2527 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del buf1 
    return (buf7 ,)

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
