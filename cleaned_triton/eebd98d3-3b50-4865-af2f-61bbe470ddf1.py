
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
def triton_poi_fused_add_avg_pool3d_constant_pad_nd_div_elu_mul_pow_0 (in_out_ptr0 ,in_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x1 =xindex //ks0 
    x2 =xindex 
    tmp48 =tl .load (in_ptr0 +(x2 ),xmask )
    tmp0 =(-2 )+x1 
    tmp1 =tl .full ([1 ],0 ,tl .int64 )
    tmp2 =tmp0 >=tmp1 
    tmp3 =ks1 
    tmp4 =tmp0 <tmp3 
    tmp5 =tmp2 &tmp4 
    tmp6 =tl .load (in_ptr0 +(x2 +((-2 )*ks2 *ks3 )),tmp5 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp7 =tmp6 *tmp6 
    tmp8 =tl .full (tmp7 .shape ,0.0 ,tmp7 .dtype )
    tmp9 =tl .where (tmp5 ,tmp7 ,tmp8 )
    tmp10 =(-1 )+x1 
    tmp11 =tmp10 >=tmp1 
    tmp12 =tmp10 <tmp3 
    tmp13 =tmp11 &tmp12 
    tmp14 =tl .load (in_ptr0 +(x2 +((-1 )*ks2 *ks3 )),tmp13 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp15 =tmp14 *tmp14 
    tmp16 =tl .full (tmp15 .shape ,0.0 ,tmp15 .dtype )
    tmp17 =tl .where (tmp13 ,tmp15 ,tmp16 )
    tmp18 =tmp17 +tmp9 
    tmp19 =x1 
    tmp20 =tmp19 >=tmp1 
    tmp21 =tmp19 <tmp3 
    tmp22 =tmp20 &tmp21 
    tmp23 =tl .load (in_ptr0 +(x2 ),tmp22 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp24 =tmp23 *tmp23 
    tmp25 =tl .full (tmp24 .shape ,0.0 ,tmp24 .dtype )
    tmp26 =tl .where (tmp22 ,tmp24 ,tmp25 )
    tmp27 =tmp26 +tmp18 
    tmp28 =1 +x1 
    tmp29 =tmp28 >=tmp1 
    tmp30 =tmp28 <tmp3 
    tmp31 =tmp29 &tmp30 
    tmp32 =tl .load (in_ptr0 +(ks0 +x2 ),tmp31 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp33 =tmp32 *tmp32 
    tmp34 =tl .full (tmp33 .shape ,0.0 ,tmp33 .dtype )
    tmp35 =tl .where (tmp31 ,tmp33 ,tmp34 )
    tmp36 =tmp35 +tmp27 
    tmp37 =2 +x1 
    tmp38 =tmp37 >=tmp1 
    tmp39 =tmp37 <tmp3 
    tmp40 =tmp38 &tmp39 
    tmp41 =tl .load (in_ptr0 +(x2 +2 *ks2 *ks3 ),tmp40 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp42 =tmp41 *tmp41 
    tmp43 =tl .full (tmp42 .shape ,0.0 ,tmp42 .dtype )
    tmp44 =tl .where (tmp40 ,tmp42 ,tmp43 )
    tmp45 =tmp44 +tmp36 
    tmp46 =0.2 
    tmp47 =tmp45 *tmp46 
    tmp49 =0.0001 
    tmp50 =tmp47 *tmp49 
    tmp51 =1.0 
    tmp52 =tmp50 +tmp51 
    tmp53 =0.75 
    tmp54 =libdevice .pow (tmp52 ,tmp53 )
    tmp55 =tmp48 /tmp54 
    tmp56 =0.0 
    tmp57 =tmp55 >tmp56 
    tmp58 =tmp55 *tmp51 
    tmp59 =libdevice .expm1 (tmp58 )
    tmp60 =tmp59 *tmp51 
    tmp61 =tl .where (tmp57 ,tmp58 ,tmp60 )
    tl .store (in_out_ptr0 +(x2 ),tmp61 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused_add_div_elu_max_pool2d_with_indices_mean_mul_pow_1 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =(xindex %8 )
    x1 =xindex //8 
    _tmp13 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    x3 =xindex 
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_2 =r0_index 
        tmp0 =r0_2 +x0 *(triton_helpers .div_floor_integer (7 +(ks0 //2 )*(ks1 //2 ),8 ))
        tmp1 =(ks0 //2 )*(ks1 //2 )
        tmp2 =tmp0 <tmp1 
        tmp3 =tl .load (in_ptr0 +(2 *(((r0_2 +x0 *(triton_helpers .div_floor_integer (7 +(ks0 //2 )*(ks1 //2 ),8 )))%(ks1 //2 )))+2 *ks1 *((((r0_2 +x0 *(triton_helpers .div_floor_integer (7 +(ks0 //2 )*(ks1 //2 ),8 )))//(ks1 //2 ))%(ks0 //2 )))+ks0 *ks1 *x1 ),r0_mask &tmp2 &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp4 =tl .load (in_ptr0 +(1 +2 *(((r0_2 +x0 *(triton_helpers .div_floor_integer (7 +(ks0 //2 )*(ks1 //2 ),8 )))%(ks1 //2 )))+2 *ks1 *((((r0_2 +x0 *(triton_helpers .div_floor_integer (7 +(ks0 //2 )*(ks1 //2 ),8 )))//(ks1 //2 ))%(ks0 //2 )))+ks0 *ks1 *x1 ),r0_mask &tmp2 &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp5 =triton_helpers .maximum (tmp4 ,tmp3 )
        tmp6 =tl .load (in_ptr0 +(ks1 +2 *(((r0_2 +x0 *(triton_helpers .div_floor_integer (7 +(ks0 //2 )*(ks1 //2 ),8 )))%(ks1 //2 )))+2 *ks1 *((((r0_2 +x0 *(triton_helpers .div_floor_integer (7 +(ks0 //2 )*(ks1 //2 ),8 )))//(ks1 //2 ))%(ks0 //2 )))+ks0 *ks1 *x1 ),r0_mask &tmp2 &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp7 =triton_helpers .maximum (tmp6 ,tmp5 )
        tmp8 =tl .load (in_ptr0 +(1 +ks1 +2 *(((r0_2 +x0 *(triton_helpers .div_floor_integer (7 +(ks0 //2 )*(ks1 //2 ),8 )))%(ks1 //2 )))+2 *ks1 *((((r0_2 +x0 *(triton_helpers .div_floor_integer (7 +(ks0 //2 )*(ks1 //2 ),8 )))//(ks1 //2 ))%(ks0 //2 )))+ks0 *ks1 *x1 ),r0_mask &tmp2 &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp9 =triton_helpers .maximum (tmp8 ,tmp7 )
        tmp10 =tl .full (tmp9 .shape ,0 ,tmp9 .dtype )
        tmp11 =tl .where (tmp2 ,tmp9 ,tmp10 )
        tmp12 =tl .broadcast_to (tmp11 ,[XBLOCK ,R0_BLOCK ])
        tmp14 =_tmp13 +tmp12 
        _tmp13 =tl .where (r0_mask &xmask ,tmp14 ,_tmp13 )
    tmp13 =tl .sum (_tmp13 ,1 )[:,None ]
    tl .store (out_ptr0 +(x3 ),tmp13 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused_add_div_elu_max_pool2d_with_indices_mean_mul_pow_2 (in_ptr0 ,out_ptr0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    R0_BLOCK :tl .constexpr =8 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_1 =r0_index 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(r0_1 +8 *x0 ),xmask ,other =0.0 )
    tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
    tmp3 =tl .where (xmask ,tmp1 ,0 )
    tmp4 =tl .sum (tmp3 ,1 )[:,None ]
    tl .store (out_ptr0 +(x0 ),tmp4 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused_smooth_l1_loss_3 (in_out_ptr0 ,in_ptr0 ,ks0 ,ks1 ,ks2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    _tmp14 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_0 =r0_index 
        tmp0 =tl .load (in_ptr0 +(r0_0 ),r0_mask ,eviction_policy ='evict_first',other =0.0 )
        tmp1 =(ks0 //2 )*(ks1 //2 )
        tmp2 =tmp1 .to (tl .float32 )
        tmp3 =tmp0 /tmp2 
        tmp4 =tl_math .abs (tmp3 )
        tmp5 =1.0 
        tmp6 =tmp4 <tmp5 
        tmp7 =tmp4 *tmp4 
        tmp8 =0.5 
        tmp9 =tmp7 *tmp8 
        tmp10 =tmp9 *tmp5 
        tmp11 =tmp4 -tmp8 
        tmp12 =tl .where (tmp6 ,tmp10 ,tmp11 )
        tmp13 =tl .broadcast_to (tmp12 ,[XBLOCK ,R0_BLOCK ])
        tmp15 =_tmp14 +tmp13 
        _tmp14 =tl .where (r0_mask ,tmp15 ,_tmp14 )
    tmp14 =tl .sum (_tmp14 ,1 )[:,None ]
    tmp16 =ks2 
    tmp17 =tmp16 .to (tl .float32 )
    tmp18 =tmp14 /tmp17 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp18 ,None )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        s1 *s2 
        buf0 =empty_strided_cuda ((1 ,1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ),torch .float32 )
        buf1 =reinterpret_tensor (buf0 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ),0 );del buf0 

        triton_poi_fused_add_avg_pool3d_constant_pad_nd_div_elu_mul_pow_0_xnumel =s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_add_avg_pool3d_constant_pad_nd_div_elu_mul_pow_0 [grid (triton_poi_fused_add_avg_pool3d_constant_pad_nd_div_elu_mul_pow_0_xnumel )](buf1 ,arg3_1 ,4096 ,3 ,64 ,64 ,12288 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        buf2 =empty_strided_cuda ((1 ,s0 ,1 ,1 ,8 ),(8 *s0 ,8 ,8 *s0 ,8 *s0 ,1 ),torch .float32 )

        triton_red_fused_add_div_elu_max_pool2d_with_indices_mean_mul_pow_1_xnumel =8 *s0 
        (7 +(s1 //2 )*(s2 //2 ))//8 
        get_raw_stream (0 )
        triton_red_fused_add_div_elu_max_pool2d_with_indices_mean_mul_pow_1 [grid (triton_red_fused_add_div_elu_max_pool2d_with_indices_mean_mul_pow_1_xnumel )](buf1 ,buf2 ,64 ,64 ,24 ,128 ,XBLOCK =32 ,R0_BLOCK =8 ,num_warps =2 ,num_stages =1 )
        del buf1 
        buf3 =empty_strided_cuda ((1 ,s0 ,1 ,1 ),(s0 ,1 ,s0 ,s0 ),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused_add_div_elu_max_pool2d_with_indices_mean_mul_pow_2 [grid (s0 )](buf2 ,buf3 ,3 ,8 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        del buf2 
        buf4 =empty_strided_cuda ((),(),torch .float32 )
        buf5 =buf4 ;del buf4 

        get_raw_stream (0 )
        triton_red_fused_smooth_l1_loss_3 [grid (1 )](buf5 ,buf3 ,64 ,64 ,3 ,1 ,3 ,XBLOCK =1 ,R0_BLOCK =4 ,num_warps =2 ,num_stages =1 )
        del buf3 
    return (buf5 ,)

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
