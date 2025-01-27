
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
def triton_red_fused__adaptive_avg_pool2d__native_batch_norm_legit_0 (in_ptr0 ,out_ptr0 ,out_ptr1 ,out_ptr2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =3 
    r0_numel =256 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    tmp34_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp34_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp34_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        roffset =r0_offset 
        r0_1 =(r0_index %16 )
        r0_2 =r0_index //16 
        r0_3 =r0_index 
        tmp0 =tl .load (in_ptr0 +(4 *r0_1 +256 *r0_2 +4096 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp1 =tl .load (in_ptr0 +(1 +4 *r0_1 +256 *r0_2 +4096 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp3 =tl .load (in_ptr0 +(2 +4 *r0_1 +256 *r0_2 +4096 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp5 =tl .load (in_ptr0 +(3 +4 *r0_1 +256 *r0_2 +4096 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp7 =tl .load (in_ptr0 +(64 +4 *r0_1 +256 *r0_2 +4096 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp9 =tl .load (in_ptr0 +(65 +4 *r0_1 +256 *r0_2 +4096 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp11 =tl .load (in_ptr0 +(66 +4 *r0_1 +256 *r0_2 +4096 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp13 =tl .load (in_ptr0 +(67 +4 *r0_1 +256 *r0_2 +4096 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp15 =tl .load (in_ptr0 +(128 +4 *r0_1 +256 *r0_2 +4096 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp17 =tl .load (in_ptr0 +(129 +4 *r0_1 +256 *r0_2 +4096 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp19 =tl .load (in_ptr0 +(130 +4 *r0_1 +256 *r0_2 +4096 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp21 =tl .load (in_ptr0 +(131 +4 *r0_1 +256 *r0_2 +4096 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp23 =tl .load (in_ptr0 +(192 +4 *r0_1 +256 *r0_2 +4096 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp25 =tl .load (in_ptr0 +(193 +4 *r0_1 +256 *r0_2 +4096 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp27 =tl .load (in_ptr0 +(194 +4 *r0_1 +256 *r0_2 +4096 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp29 =tl .load (in_ptr0 +(195 +4 *r0_1 +256 *r0_2 +4096 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last',other =0.0 )
        tmp2 =tmp1 +tmp0 
        tmp4 =tmp3 +tmp2 
        tmp6 =tmp5 +tmp4 
        tmp8 =tmp7 +tmp6 
        tmp10 =tmp9 +tmp8 
        tmp12 =tmp11 +tmp10 
        tmp14 =tmp13 +tmp12 
        tmp16 =tmp15 +tmp14 
        tmp18 =tmp17 +tmp16 
        tmp20 =tmp19 +tmp18 
        tmp22 =tmp21 +tmp20 
        tmp24 =tmp23 +tmp22 
        tmp26 =tmp25 +tmp24 
        tmp28 =tmp27 +tmp26 
        tmp30 =tmp29 +tmp28 
        tmp31 =0.0625 
        tmp32 =tmp30 *tmp31 
        tmp33 =tl .broadcast_to (tmp32 ,[XBLOCK ,R0_BLOCK ])
        tmp34_mean_next ,tmp34_m2_next ,tmp34_weight_next =triton_helpers .welford_reduce (
        tmp33 ,tmp34_mean ,tmp34_m2 ,tmp34_weight ,roffset ==0 
        )
        tmp34_mean =tl .where (r0_mask &xmask ,tmp34_mean_next ,tmp34_mean )
        tmp34_m2 =tl .where (r0_mask &xmask ,tmp34_m2_next ,tmp34_m2 )
        tmp34_weight =tl .where (r0_mask &xmask ,tmp34_weight_next ,tmp34_weight )
        tl .store (out_ptr0 +(r0_3 +256 *x0 ),tmp32 ,r0_mask &xmask )
    tmp37 ,tmp38 ,tmp39 =triton_helpers .welford (tmp34_mean ,tmp34_m2 ,tmp34_weight ,1 )
    tmp34 =tmp37 [:,None ]
    tmp35 =tmp38 [:,None ]
    tmp36 =tmp39 [:,None ]
    tl .store (out_ptr1 +(x0 ),tmp34 ,xmask )
    tl .store (out_ptr2 +(x0 ),tmp35 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused__adaptive_avg_pool2d__native_batch_norm_legit__to_copy__unsafe_index_add_arange_clamp_mul_sub_1 (in_out_ptr0 ,in_out_ptr1 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr2 ,out_ptr3 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =3 
    r0_numel =1024 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    tmp19 =tl .load (in_ptr1 +(x0 ),xmask ,eviction_policy ='evict_last')
    tmp21 =tl .load (in_ptr2 +(x0 ),xmask ,eviction_policy ='evict_last')
    tmp56_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp56_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp56_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        roffset =r0_offset 
        r0_2 =r0_index //32 
        r0_1 =(r0_index %32 )
        r0_3 =r0_index 
        tmp0 =r0_2 
        tmp1 =tmp0 .to (tl .float32 )
        tmp2 =0.4838709677419355 
        tmp3 =tmp1 *tmp2 
        tmp4 =0.0 
        tmp5 =triton_helpers .maximum (tmp3 ,tmp4 )
        tmp6 =tmp5 .to (tl .int32 )
        tmp7 =tl .full ([1 ,1 ],1 ,tl .int64 )
        tmp8 =tmp6 +tmp7 
        tmp9 =tl .full ([1 ,1 ],15 ,tl .int64 )
        tmp10 =triton_helpers .minimum (tmp8 ,tmp9 )
        tmp11 =r0_1 
        tmp12 =tmp11 .to (tl .float32 )
        tmp13 =tmp12 *tmp2 
        tmp14 =triton_helpers .maximum (tmp13 ,tmp4 )
        tmp15 =tmp14 .to (tl .int32 )
        tmp16 =tmp15 +tmp7 
        tmp17 =triton_helpers .minimum (tmp16 ,tmp9 )
        tmp18 =tl .load (in_ptr0 +(tmp17 +16 *tmp10 +256 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last')
        tmp20 =tmp18 -tmp19 
        tmp22 =256.0 
        tmp23 =tmp21 /tmp22 
        tmp24 =1e-05 
        tmp25 =tmp23 +tmp24 
        tmp26 =libdevice .rsqrt (tmp25 )
        tmp27 =tmp20 *tmp26 
        tmp28 =tl .load (in_ptr0 +(tmp15 +16 *tmp10 +256 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last')
        tmp29 =tmp28 -tmp19 
        tmp30 =tmp29 *tmp26 
        tmp31 =tmp27 -tmp30 
        tmp32 =tmp15 .to (tl .float32 )
        tmp33 =tmp14 -tmp32 
        tmp34 =triton_helpers .maximum (tmp33 ,tmp4 )
        tmp35 =1.0 
        tmp36 =triton_helpers .minimum (tmp34 ,tmp35 )
        tmp37 =tmp31 *tmp36 
        tmp38 =tmp30 +tmp37 
        tmp39 =tl .load (in_ptr0 +(tmp17 +16 *tmp6 +256 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last')
        tmp40 =tmp39 -tmp19 
        tmp41 =tmp40 *tmp26 
        tmp42 =tl .load (in_ptr0 +(tmp15 +16 *tmp6 +256 *x0 ),r0_mask &xmask ,eviction_policy ='evict_last')
        tmp43 =tmp42 -tmp19 
        tmp44 =tmp43 *tmp26 
        tmp45 =tmp41 -tmp44 
        tmp46 =tmp45 *tmp36 
        tmp47 =tmp44 +tmp46 
        tmp48 =tmp38 -tmp47 
        tmp49 =tmp6 .to (tl .float32 )
        tmp50 =tmp5 -tmp49 
        tmp51 =triton_helpers .maximum (tmp50 ,tmp4 )
        tmp52 =triton_helpers .minimum (tmp51 ,tmp35 )
        tmp53 =tmp48 *tmp52 
        tmp54 =tmp47 +tmp53 
        tmp55 =tl .broadcast_to (tmp54 ,[XBLOCK ,R0_BLOCK ])
        tmp56_mean_next ,tmp56_m2_next ,tmp56_weight_next =triton_helpers .welford_reduce (
        tmp55 ,tmp56_mean ,tmp56_m2 ,tmp56_weight ,roffset ==0 
        )
        tmp56_mean =tl .where (r0_mask &xmask ,tmp56_mean_next ,tmp56_mean )
        tmp56_m2 =tl .where (r0_mask &xmask ,tmp56_m2_next ,tmp56_m2 )
        tmp56_weight =tl .where (r0_mask &xmask ,tmp56_weight_next ,tmp56_weight )
        tl .store (in_out_ptr0 +(r0_3 +1024 *x0 ),tmp38 ,r0_mask &xmask )
        tl .store (in_out_ptr1 +(r0_3 +1024 *x0 ),tmp47 ,r0_mask &xmask )
    tmp59 ,tmp60 ,tmp61 =triton_helpers .welford (tmp56_mean ,tmp56_m2 ,tmp56_weight ,1 )
    tmp56 =tmp59 [:,None ]
    tmp57 =tmp60 [:,None ]
    tmp58 =tmp61 [:,None ]
    tmp87_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp87_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp87_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        roffset =r0_offset 
        r0_3 =r0_index 
        r0_2 =r0_index //32 
        tmp62 =tl .load (in_out_ptr1 +(r0_3 +1024 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp63 =tl .load (in_out_ptr0 +(r0_3 +1024 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp64 =tmp63 -tmp62 
        tmp65 =r0_2 
        tmp66 =tmp65 .to (tl .float32 )
        tmp67 =0.4838709677419355 
        tmp68 =tmp66 *tmp67 
        tmp69 =0.0 
        tmp70 =triton_helpers .maximum (tmp68 ,tmp69 )
        tmp71 =tmp70 .to (tl .int32 )
        tmp72 =tmp71 .to (tl .float32 )
        tmp73 =tmp70 -tmp72 
        tmp74 =triton_helpers .maximum (tmp73 ,tmp69 )
        tmp75 =1.0 
        tmp76 =triton_helpers .minimum (tmp74 ,tmp75 )
        tmp77 =tmp64 *tmp76 
        tmp78 =tmp62 +tmp77 
        tmp79 =tmp78 -tmp56 
        tmp80 =1024.0 
        tmp81 =tmp57 /tmp80 
        tmp82 =1e-05 
        tmp83 =tmp81 +tmp82 
        tmp84 =libdevice .rsqrt (tmp83 )
        tmp85 =tmp79 *tmp84 
        tmp86 =tl .broadcast_to (tmp85 ,[XBLOCK ,R0_BLOCK ])
        tmp87_mean_next ,tmp87_m2_next ,tmp87_weight_next =triton_helpers .welford_reduce (
        tmp86 ,tmp87_mean ,tmp87_m2 ,tmp87_weight ,roffset ==0 
        )
        tmp87_mean =tl .where (r0_mask &xmask ,tmp87_mean_next ,tmp87_mean )
        tmp87_m2 =tl .where (r0_mask &xmask ,tmp87_m2_next ,tmp87_m2 )
        tmp87_weight =tl .where (r0_mask &xmask ,tmp87_weight_next ,tmp87_weight )
        tl .store (in_out_ptr1 +(r0_3 +1024 *x0 ),tmp85 ,r0_mask &xmask )
    tmp90 ,tmp91 ,tmp92 =triton_helpers .welford (tmp87_mean ,tmp87_m2 ,tmp87_weight ,1 )
    tmp87 =tmp90 [:,None ]
    tmp88 =tmp91 [:,None ]
    tmp89 =tmp92 [:,None ]
    tl .store (out_ptr2 +(x0 ),tmp87 ,xmask )
    tl .store (out_ptr3 +(x0 ),tmp88 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__adaptive_avg_pool2d__native_batch_norm_legit__to_copy__unsafe_index_add_arange_clamp_mul_sub_2 (in_out_ptr1 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    tl .full ([XBLOCK ],True ,tl .int1 )
    x1 =((xindex //64 )%64 )
    x0 =(xindex %64 )
    x2 =xindex //4096 
    x4 =xindex 
    tmp19 =tl .load (in_ptr1 +(x2 ),None ,eviction_policy ='evict_last')
    tmp21 =tl .load (in_ptr2 +(x2 ),None ,eviction_policy ='evict_last')
    tmp0 =x1 
    tmp1 =tmp0 .to (tl .float32 )
    tmp2 =0.49206349206349204 
    tmp3 =tmp1 *tmp2 
    tmp4 =0.0 
    tmp5 =triton_helpers .maximum (tmp3 ,tmp4 )
    tmp6 =tmp5 .to (tl .int32 )
    tmp7 =tl .full ([1 ],1 ,tl .int64 )
    tmp8 =tmp6 +tmp7 
    tmp9 =tl .full ([1 ],31 ,tl .int64 )
    tmp10 =triton_helpers .minimum (tmp8 ,tmp9 )
    tmp11 =x0 
    tmp12 =tmp11 .to (tl .float32 )
    tmp13 =tmp12 *tmp2 
    tmp14 =triton_helpers .maximum (tmp13 ,tmp4 )
    tmp15 =tmp14 .to (tl .int32 )
    tmp16 =tmp15 +tmp7 
    tmp17 =triton_helpers .minimum (tmp16 ,tmp9 )
    tmp18 =tl .load (in_ptr0 +(tmp17 +32 *tmp10 +1024 *x2 ),None ,eviction_policy ='evict_last')
    tmp20 =tmp18 -tmp19 
    tmp22 =1024.0 
    tmp23 =tmp21 /tmp22 
    tmp24 =1e-05 
    tmp25 =tmp23 +tmp24 
    tmp26 =libdevice .rsqrt (tmp25 )
    tmp27 =tmp20 *tmp26 
    tmp28 =tl .load (in_ptr0 +(tmp15 +32 *tmp10 +1024 *x2 ),None ,eviction_policy ='evict_last')
    tmp29 =tmp28 -tmp19 
    tmp30 =tmp29 *tmp26 
    tmp31 =tmp27 -tmp30 
    tmp32 =tmp15 .to (tl .float32 )
    tmp33 =tmp14 -tmp32 
    tmp34 =triton_helpers .maximum (tmp33 ,tmp4 )
    tmp35 =1.0 
    tmp36 =triton_helpers .minimum (tmp34 ,tmp35 )
    tmp37 =tmp31 *tmp36 
    tmp38 =tmp30 +tmp37 
    tmp39 =tl .load (in_ptr0 +(tmp17 +32 *tmp6 +1024 *x2 ),None ,eviction_policy ='evict_last')
    tmp40 =tmp39 -tmp19 
    tmp41 =tmp40 *tmp26 
    tmp42 =tl .load (in_ptr0 +(tmp15 +32 *tmp6 +1024 *x2 ),None ,eviction_policy ='evict_last')
    tmp43 =tmp42 -tmp19 
    tmp44 =tmp43 *tmp26 
    tmp45 =tmp41 -tmp44 
    tmp46 =tmp45 *tmp36 
    tmp47 =tmp44 +tmp46 
    tmp48 =tmp38 -tmp47 
    tmp49 =tmp6 .to (tl .float32 )
    tmp50 =tmp5 -tmp49 
    tmp51 =triton_helpers .maximum (tmp50 ,tmp4 )
    tmp52 =triton_helpers .minimum (tmp51 ,tmp35 )
    tmp53 =tmp48 *tmp52 
    tmp54 =tmp47 +tmp53 
    tl .store (in_out_ptr1 +(x4 ),tmp54 ,None )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 =args 
    args .clear ()
    assert_size_stride (arg2_1 ,(1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,3 ,16 ,16 ),(768 ,256 ,16 ,1 ),torch .float32 )
        buf1 =empty_strided_cuda ((1 ,3 ,1 ,1 ),(3 ,1 ,3 ,3 ),torch .float32 )
        buf2 =empty_strided_cuda ((1 ,3 ,1 ,1 ),(3 ,1 ,3 ,3 ),torch .float32 )

        get_raw_stream (0 )
        triton_red_fused__adaptive_avg_pool2d__native_batch_norm_legit_0 [grid (3 )](arg2_1 ,buf0 ,buf1 ,buf2 ,3 ,256 ,XBLOCK =1 ,R0_BLOCK =256 ,num_warps =2 ,num_stages =1 )
        del arg2_1 
        buf4 =empty_strided_cuda ((1 ,3 ,32 ,32 ),(3072 ,1024 ,32 ,1 ),torch .float32 )
        buf5 =buf4 ;del buf4 
        buf6 =buf5 ;del buf5 
        buf7 =empty_strided_cuda ((1 ,3 ,32 ,32 ),(3072 ,1024 ,32 ,1 ),torch .float32 )
        buf8 =buf7 ;del buf7 
        buf12 =buf8 ;del buf8 
        buf13 =empty_strided_cuda ((1 ,3 ,1 ,1 ),(3 ,1 ,3 ,3 ),torch .float32 )
        buf14 =empty_strided_cuda ((1 ,3 ,1 ,1 ),(3 ,1 ,3 ,3 ),torch .float32 )

        get_raw_stream (0 )
        triton_red_fused__adaptive_avg_pool2d__native_batch_norm_legit__to_copy__unsafe_index_add_arange_clamp_mul_sub_1 [grid (3 )](buf6 ,buf12 ,buf0 ,buf1 ,buf2 ,buf13 ,buf14 ,3 ,1024 ,XBLOCK =1 ,R0_BLOCK =1024 ,num_warps =8 ,num_stages =1 )
        del buf0 
        del buf1 
        del buf2 
        del buf6 
        buf19 =empty_strided_cuda ((1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ),torch .float32 )
        buf20 =buf19 ;del buf19 
        buf21 =buf20 ;del buf20 

        get_raw_stream (0 )
        triton_poi_fused__adaptive_avg_pool2d__native_batch_norm_legit__to_copy__unsafe_index_add_arange_clamp_mul_sub_2 [grid (12288 )](buf21 ,buf12 ,buf13 ,buf14 ,12288 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del buf12 
        del buf13 
        del buf14 
    return (buf21 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =64 
    arg1_1 =64 
    arg2_1 =rand_strided ((1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
