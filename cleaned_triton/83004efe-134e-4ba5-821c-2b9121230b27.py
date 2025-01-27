
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
def triton_red_fused__native_batch_norm_legit_0 (in_ptr0 ,out_ptr2 ,ks0 ,ks1 ,ks2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =64 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    tmp2_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp2_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp2_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        roffset =r0_offset 
        r0_1 =r0_index 
        tmp0 =tl .load (in_ptr0 +(r0_1 +ks0 *ks1 *ks2 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
        tmp2_mean_next ,tmp2_m2_next ,tmp2_weight_next =triton_helpers .welford_reduce (
        tmp1 ,tmp2_mean ,tmp2_m2 ,tmp2_weight ,roffset ==0 
        )
        tmp2_mean =tl .where (r0_mask &xmask ,tmp2_mean_next ,tmp2_mean )
        tmp2_m2 =tl .where (r0_mask &xmask ,tmp2_m2_next ,tmp2_m2 )
        tmp2_weight =tl .where (r0_mask &xmask ,tmp2_weight_next ,tmp2_weight )
    tmp5 ,tmp6 ,tmp7 =triton_helpers .welford (tmp2_mean ,tmp2_m2 ,tmp2_weight ,1 )
    tmp2 =tmp5 [:,None ]
    tmp3 =tmp6 [:,None ]
    tmp4 =tmp7 [:,None ]
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        roffset =r0_offset 
        r0_1 =r0_index 
        tmp8 =tl .load (in_ptr0 +(r0_1 +ks0 *ks1 *ks2 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp9 =tmp8 -tmp2 
        tmp10 =ks0 *ks1 *ks2 
        tmp11 =tmp10 .to (tl .float32 )
        tmp12 =tmp3 /tmp11 
        tmp13 =1e-05 
        tmp14 =tmp12 +tmp13 
        tmp15 =libdevice .rsqrt (tmp14 )
        tmp16 =tmp9 *tmp15 
        tl .store (out_ptr2 +(r0_1 +ks0 *ks1 *ks2 *x0 ),tmp16 ,r0_mask &xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_avg_pool3d_1 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    tl .full ([XBLOCK ],True ,tl .int1 )
    x0 =(xindex %4 )
    x1 =((xindex //4 )%4 )
    x2 =((xindex //16 )%4 )
    x3 =xindex //64 
    x4 =xindex 
    tmp0 =tl .load (in_ptr0 +(2 *x0 +16 *x1 +128 *x2 +ks0 *ks1 *ks2 *x3 ),None ,eviction_policy ='evict_last')
    tmp1 =tl .load (in_ptr0 +(1 +2 *x0 +16 *x1 +128 *x2 +ks0 *ks1 *ks2 *x3 ),None ,eviction_policy ='evict_last')
    tmp3 =tl .load (in_ptr0 +(8 +2 *x0 +16 *x1 +128 *x2 +ks0 *ks1 *ks2 *x3 ),None ,eviction_policy ='evict_last')
    tmp5 =tl .load (in_ptr0 +(9 +2 *x0 +16 *x1 +128 *x2 +ks0 *ks1 *ks2 *x3 ),None ,eviction_policy ='evict_last')
    tmp7 =tl .load (in_ptr0 +(64 +2 *x0 +16 *x1 +128 *x2 +ks0 *ks1 *ks2 *x3 ),None ,eviction_policy ='evict_last')
    tmp9 =tl .load (in_ptr0 +(65 +2 *x0 +16 *x1 +128 *x2 +ks0 *ks1 *ks2 *x3 ),None ,eviction_policy ='evict_last')
    tmp11 =tl .load (in_ptr0 +(72 +2 *x0 +16 *x1 +128 *x2 +ks0 *ks1 *ks2 *x3 ),None ,eviction_policy ='evict_last')
    tmp13 =tl .load (in_ptr0 +(73 +2 *x0 +16 *x1 +128 *x2 +ks0 *ks1 *ks2 *x3 ),None ,eviction_policy ='evict_last')
    tmp2 =tmp1 +tmp0 
    tmp4 =tmp3 +tmp2 
    tmp6 =tmp5 +tmp4 
    tmp8 =tmp7 +tmp6 
    tmp10 =tmp9 +tmp8 
    tmp12 =tmp11 +tmp10 
    tmp14 =tmp13 +tmp12 
    tmp15 =0.125 
    tmp16 =tmp14 *tmp15 
    tl .store (out_ptr0 +(x4 ),tmp16 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__to_copy_add_arange_clamp_view_2 (out_ptr0 ,ks0 ,ks1 ,ks2 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =ks0 *ks1 *ks2 
    tmp1 =tmp0 .to (tl .float32 )
    tmp2 =128.0 
    tmp3 =tmp1 /tmp2 
    tmp4 =libdevice .floor (tmp3 )
    tmp5 =4.0 
    tmp6 =tmp5 *tmp4 
    tmp7 =tmp6 .to (tl .float64 )
    tmp8 =tl .full ([1 ],-1.0 ,tl .float64 )
    tmp9 =tmp8 +tmp7 
    tmp10 =8.0 
    tmp11 =tmp10 *tmp4 
    tmp12 =tmp11 .to (tl .float64 )
    tmp13 =tmp8 +tmp12 
    tmp14 =tmp9 /tmp13 
    tmp15 =tmp14 .to (tl .float32 )
    tmp16 =x0 
    tmp17 =tmp16 .to (tl .float32 )
    tmp18 =tmp17 *tmp15 
    tmp19 =0.0 
    tmp20 =triton_helpers .maximum (tmp18 ,tmp19 )
    tmp21 =tmp20 .to (tl .int64 )
    tmp22 =tl .full ([1 ],1 ,tl .int64 )
    tmp23 =tmp21 +tmp22 
    tmp24 =(-1 )+4 *((ks0 *ks1 *ks2 )//128 )
    tmp25 =triton_helpers .minimum (tmp23 ,tmp24 )
    tl .store (out_ptr0 +(x0 ),tmp25 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__to_copy_arange_clamp_sub_view_3 (out_ptr0 ,ks0 ,ks1 ,ks2 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =ks0 *ks1 *ks2 
    tmp1 =tmp0 .to (tl .float32 )
    tmp2 =128.0 
    tmp3 =tmp1 /tmp2 
    tmp4 =libdevice .floor (tmp3 )
    tmp5 =4.0 
    tmp6 =tmp5 *tmp4 
    tmp7 =tmp6 .to (tl .float64 )
    tmp8 =tl .full ([1 ],-1.0 ,tl .float64 )
    tmp9 =tmp8 +tmp7 
    tmp10 =8.0 
    tmp11 =tmp10 *tmp4 
    tmp12 =tmp11 .to (tl .float64 )
    tmp13 =tmp8 +tmp12 
    tmp14 =tmp9 /tmp13 
    tmp15 =tmp14 .to (tl .float32 )
    tmp16 =x0 
    tmp17 =tmp16 .to (tl .float32 )
    tmp18 =tmp17 *tmp15 
    tmp19 =0.0 
    tmp20 =triton_helpers .maximum (tmp18 ,tmp19 )
    tmp21 =tmp20 .to (tl .int64 )
    tmp22 =tmp21 .to (tl .float32 )
    tmp23 =tmp20 -tmp22 
    tmp24 =triton_helpers .maximum (tmp23 ,tmp19 )
    tmp25 =1.0 
    tmp26 =triton_helpers .minimum (tmp24 ,tmp25 )
    tl .store (out_ptr0 +(x0 ),tmp26 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__to_copy__unsafe_index_add_clamp_gelu_mul_sub_4 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    tl .full ([XBLOCK ],True ,tl .int1 )
    x1 =((xindex //ks0 )%8 )
    x0 =(xindex %ks0 )
    x2 =xindex //ks4 
    x3 =xindex 
    tmp34 =tl .load (in_ptr1 +(x0 ),None ,eviction_policy ='evict_last')
    tmp41 =tl .load (in_ptr2 +(x0 ),None ,eviction_policy ='evict_last')
    tmp0 =x1 
    tmp1 =tmp0 .to (tl .float32 )
    tmp2 =0.42857142857142855 
    tmp3 =tmp1 *tmp2 
    tmp4 =0.0 
    tmp5 =triton_helpers .maximum (tmp3 ,tmp4 )
    tmp6 =tmp5 .to (tl .int32 )
    tmp7 =tl .full ([1 ],1 ,tl .int64 )
    tmp8 =tmp6 +tmp7 
    tmp9 =tl .full ([1 ],3 ,tl .int64 )
    tmp10 =triton_helpers .minimum (tmp8 ,tmp9 )
    tmp11 =ks1 *ks2 *ks3 
    tmp12 =tmp11 .to (tl .float32 )
    tmp13 =128.0 
    tmp14 =tmp12 /tmp13 
    tmp15 =libdevice .floor (tmp14 )
    tmp16 =4.0 
    tmp17 =tmp16 *tmp15 
    tmp18 =tmp17 .to (tl .float64 )
    tmp19 =tl .full ([1 ],-1.0 ,tl .float64 )
    tmp20 =tmp19 +tmp18 
    tmp21 =8.0 
    tmp22 =tmp21 *tmp15 
    tmp23 =tmp22 .to (tl .float64 )
    tmp24 =tmp19 +tmp23 
    tmp25 =tmp20 /tmp24 
    tmp26 =tmp25 .to (tl .float32 )
    tmp27 =x0 
    tmp28 =tmp27 .to (tl .float32 )
    tmp29 =tmp28 *tmp26 
    tmp30 =triton_helpers .maximum (tmp29 ,tmp4 )
    tmp31 =tmp30 .to (tl .int64 )
    tmp32 =tl .load (in_ptr0 +(16 *tmp10 +64 *x2 +((tmp31 %16 ))),None ,eviction_policy ='evict_last')
    tmp33 =tl .load (in_ptr0 +(16 *tmp6 +64 *x2 +((tmp31 %16 ))),None ,eviction_policy ='evict_last')
    tmp35 =tl .full ([XBLOCK ],16 ,tl .int32 )
    tmp36 =tmp34 +tmp35 
    tmp37 =tmp34 <0 
    tmp38 =tl .where (tmp37 ,tmp36 ,tmp34 )
    tmp39 =tl .load (in_ptr0 +(16 *tmp6 +64 *x2 +((tmp38 %16 ))),None ,eviction_policy ='evict_last')
    tmp40 =tmp39 -tmp33 
    tmp42 =tmp40 *tmp41 
    tmp43 =tmp33 +tmp42 
    tmp44 =tl .load (in_ptr0 +(16 *tmp10 +64 *x2 +((tmp38 %16 ))),None ,eviction_policy ='evict_last')
    tmp45 =tmp44 -tmp32 
    tmp46 =tmp45 *tmp41 
    tmp47 =tmp32 +tmp46 
    tmp48 =tmp47 -tmp43 
    tmp49 =tmp6 .to (tl .float32 )
    tmp50 =tmp5 -tmp49 
    tmp51 =triton_helpers .maximum (tmp50 ,tmp4 )
    tmp52 =1.0 
    tmp53 =triton_helpers .minimum (tmp51 ,tmp52 )
    tmp54 =tmp48 *tmp53 
    tmp55 =tmp43 +tmp54 
    tmp56 =0.5 
    tmp57 =tmp55 *tmp56 
    tmp58 =0.7071067811865476 
    tmp59 =tmp55 *tmp58 
    tmp60 =libdevice .erf (tmp59 )
    tmp61 =tmp60 +tmp52 
    tmp62 =tmp57 *tmp61 
    tl .store (in_out_ptr0 +(x3 ),tmp62 ,None )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s1 =arg0_1 
    s2 =arg1_1 
    s3 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,64 ,s1 ,s2 ,s3 ),(64 *s1 *s2 *s3 ,s1 *s2 *s3 ,s2 *s3 ,s3 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf3 =empty_strided_cuda ((1 ,64 ,s1 *s2 *s3 ),(64 *s1 *s2 *s3 ,s1 *s2 *s3 ,1 ),torch .float32 )

        s1 *s2 *s3 
        get_raw_stream (0 )
        triton_red_fused__native_batch_norm_legit_0 [grid (64 )](arg3_1 ,buf3 ,8 ,8 ,8 ,64 ,512 ,XBLOCK =1 ,R0_BLOCK =512 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        buf4 =empty_strided_cuda ((1 ,64 ,4 ,4 ,4 ),(4096 ,64 ,16 ,4 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_avg_pool3d_1 [grid (4096 )](buf3 ,buf4 ,8 ,8 ,8 ,4096 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del buf3 
        buf5 =empty_strided_cuda ((8 *((s1 *s2 *s3 )//128 ),),(1 ,),torch .int64 )

        triton_poi_fused__to_copy_add_arange_clamp_view_2_xnumel =8 *((s1 *s2 *s3 )//128 )
        get_raw_stream (0 )
        triton_poi_fused__to_copy_add_arange_clamp_view_2 [grid (triton_poi_fused__to_copy_add_arange_clamp_view_2_xnumel )](buf5 ,8 ,8 ,8 ,32 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        buf7 =empty_strided_cuda ((8 *((s1 *s2 *s3 )//128 ),),(1 ,),torch .float32 )

        triton_poi_fused__to_copy_arange_clamp_sub_view_3_xnumel =8 *((s1 *s2 *s3 )//128 )
        get_raw_stream (0 )
        triton_poi_fused__to_copy_arange_clamp_sub_view_3 [grid (triton_poi_fused__to_copy_arange_clamp_sub_view_3_xnumel )](buf7 ,8 ,8 ,8 ,32 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        8 *((s1 *s2 *s3 )//128 )
        64 *((s1 *s2 *s3 )//128 )
        buf8 =empty_strided_cuda ((1 ,64 ,8 ,8 *((s1 *s2 *s3 )//128 )),(4096 *((s1 *s2 *s3 )//128 ),64 *((s1 *s2 *s3 )//128 ),8 *((s1 *s2 *s3 )//128 ),1 ),torch .float32 )
        buf9 =buf8 ;del buf8 
        buf10 =buf9 ;del buf9 

        triton_poi_fused__to_copy__unsafe_index_add_clamp_gelu_mul_sub_4_xnumel =4096 *((s1 *s2 *s3 )//128 )
        get_raw_stream (0 )
        triton_poi_fused__to_copy__unsafe_index_add_clamp_gelu_mul_sub_4 [grid (triton_poi_fused__to_copy__unsafe_index_add_clamp_gelu_mul_sub_4_xnumel )](buf10 ,buf4 ,buf5 ,buf7 ,32 ,8 ,8 ,8 ,256 ,16384 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf4 
        del buf5 
        del buf7 
    return (reinterpret_tensor (buf10 ,(1 ,64 ,64 *((s1 *s2 *s3 )//128 )),(4096 *((s1 *s2 *s3 )//128 ),64 *((s1 *s2 *s3 )//128 ),1 ),0 ),)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =8 
    arg1_1 =8 
    arg2_1 =8 
    arg3_1 =rand_strided ((1 ,64 ,8 ,8 ,8 ),(32768 ,512 ,64 ,8 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
