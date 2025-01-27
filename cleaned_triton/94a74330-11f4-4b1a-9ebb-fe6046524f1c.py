
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
def triton_poi_fused__to_copy_bernoulli_mul_0 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,out_ptr0 ,load_seed_offset ,load_seed_offset1 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp5 =tl .load (in_ptr1 +(x0 ),xmask )
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =x0 
    tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tmp3 =tl .load (in_ptr0 +load_seed_offset1 )
    tmp4 =tl .rand (tmp3 ,(tmp1 ).to (tl .uint32 ))
    tmp6 =0.5 
    tmp7 =tmp2 <tmp6 
    tmp8 =tmp7 .to (tl .float32 )
    tmp9 =0.8864048946659319 
    tmp10 =tmp8 *tmp9 
    tmp11 =tmp5 *tmp10 
    tmp12 =-1.0 
    tmp13 =tmp8 +tmp12 
    tmp14 =1.558387861036063 
    tmp15 =tmp13 *tmp14 
    tmp16 =0.7791939305180315 
    tmp17 =tmp15 +tmp16 
    tmp18 =tmp11 +tmp17 
    tmp19 =20.0 
    tmp20 =tmp18 >tmp19 
    tmp21 =tl_math .exp (tmp18 )
    tmp22 =libdevice .log1p (tmp21 )
    tmp23 =tl .where (tmp20 ,tmp18 ,tmp22 )
    tmp24 =libdevice .tanh (tmp23 )
    tmp25 =tmp18 *tmp24 
    tmp26 =1.0 
    tmp27 =tmp25 *tmp26 
    tmp28 =tmp27 >tmp19 
    tmp29 =tl_math .exp (tmp27 )
    tmp30 =libdevice .log1p (tmp29 )
    tmp31 =tmp30 *tmp26 
    tmp32 =tl .where (tmp28 ,tmp25 ,tmp31 )
    tmp33 =tmp4 <tmp6 
    tmp34 =tmp33 .to (tl .float32 )
    tmp35 =tmp34 *tmp9 
    tmp36 =tmp32 *tmp35 
    tl .store (out_ptr0 +(x0 ),tmp4 ,xmask )
    tl .store (in_out_ptr0 +(x0 ),tmp36 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__to_copy_add_bernoulli_gather_mish_mul_softplus_1 (in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr0 ,load_seed_offset ,ks1 ,ks2 ,ks3 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    tl .full ([XBLOCK ],True ,tl .int1 )
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =tl .full ([1 ],0 ,tl .int32 )
    tmp2 =tl .full ([1 ],0 ,tl .int64 )
    tmp3 =ks1 *ks2 *ks3 
    tmp4 =triton_helpers .randint64 (tmp0 ,(tmp1 ).to (tl .uint32 ),tmp2 ,tmp3 )
    tmp5 =tmp4 +tmp3 
    tmp6 =tmp4 <0 
    tmp7 =tl .where (tmp6 ,tmp5 ,tmp4 )
    tl .device_assert ((0 <=tmp7 )&(tmp7 <ks1 *ks2 *ks3 ),"index out of bounds: 0 <= tmp7 < ks1*ks2*ks3")
    tmp9 =tl .load (in_ptr1 +(tmp7 ),None ,eviction_policy ='evict_last')
    tmp10 =tl .load (in_ptr2 +(tmp7 ),None ,eviction_policy ='evict_last')
    tmp11 =0.5 
    tmp12 =tmp10 <tmp11 
    tmp13 =tmp12 .to (tl .float32 )
    tmp14 =-1.0 
    tmp15 =tmp13 +tmp14 
    tmp16 =1.558387861036063 
    tmp17 =tmp15 *tmp16 
    tmp18 =0.7791939305180315 
    tmp19 =tmp17 +tmp18 
    tmp20 =tmp9 +tmp19 
    tmp21 =20.0 
    tmp22 =tmp20 >tmp21 
    tmp23 =tl_math .exp (tmp20 )
    tmp24 =libdevice .log1p (tmp23 )
    tmp25 =tl .where (tmp22 ,tmp20 ,tmp24 )
    tmp26 =libdevice .tanh (tmp25 )
    tmp27 =tmp20 *tmp26 
    tmp28 =1.0 
    tmp29 =tmp27 *tmp28 
    tmp30 =tmp29 >tmp21 
    tmp31 =tl_math .exp (tmp29 )
    tmp32 =libdevice .log1p (tmp31 )
    tmp33 =tmp32 *tmp28 
    tmp34 =tl .where (tmp30 ,tmp27 ,tmp33 )
    tl .store (out_ptr0 +(tl .full ([XBLOCK ],0 ,tl .int32 )),tmp34 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__to_copy_add_bernoulli_clamp_min_mish_mul_rsub_softplus_2 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(0 ))
    tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ])
    tmp4 =tl .load (in_out_ptr0 +(x0 ),xmask )
    tmp5 =tl .load (in_ptr1 +(x0 ),xmask )
    tmp2 =1.0 
    tmp3 =tmp2 -tmp1 
    tmp6 =0.5 
    tmp7 =tmp5 <tmp6 
    tmp8 =tmp7 .to (tl .float32 )
    tmp9 =-1.0 
    tmp10 =tmp8 +tmp9 
    tmp11 =1.558387861036063 
    tmp12 =tmp10 *tmp11 
    tmp13 =0.7791939305180315 
    tmp14 =tmp12 +tmp13 
    tmp15 =tmp4 +tmp14 
    tmp16 =20.0 
    tmp17 =tmp15 >tmp16 
    tmp18 =tl_math .exp (tmp15 )
    tmp19 =libdevice .log1p (tmp18 )
    tmp20 =tl .where (tmp17 ,tmp15 ,tmp19 )
    tmp21 =libdevice .tanh (tmp20 )
    tmp22 =tmp15 *tmp21 
    tmp23 =tmp22 *tmp2 
    tmp24 =tmp23 >tmp16 
    tmp25 =tl_math .exp (tmp23 )
    tmp26 =libdevice .log1p (tmp25 )
    tmp27 =tmp26 *tmp2 
    tmp28 =tl .where (tmp24 ,tmp22 ,tmp27 )
    tmp29 =tmp3 +tmp28 
    tmp30 =0.0 
    tmp31 =triton_helpers .maximum (tmp29 ,tmp30 )
    tl .store (in_out_ptr0 +(x0 ),tmp31 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused_arange_mean_ne_scalar_tensor_where_3 (in_ptr0 ,in_ptr1 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,load_seed_offset ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =2 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    _tmp16 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_1 =r0_index 
        tmp0 =r0_1 +x0 *((1 +ks0 *ks1 *ks2 )//2 )
        tmp1 =ks0 *ks1 *ks2 
        tmp2 =tmp0 <tmp1 
        tmp3 =tl .load (in_ptr0 +load_seed_offset )
        tmp4 =tl .full ([1 ,1 ],0 ,tl .int32 )
        tmp5 =tl .full ([1 ,1 ],0 ,tl .int64 )
        tmp6 =tl .broadcast_to (ks0 *ks1 *ks2 ,[XBLOCK ,R0_BLOCK ])
        tmp7 =triton_helpers .randint64 (tmp3 ,(tmp4 ).to (tl .uint32 ),tmp5 ,tmp6 )
        tmp8 =r0_1 +x0 *((1 +ks0 *ks1 *ks2 )//2 )
        tmp9 =tmp8 !=tmp7 
        tmp10 =tl .load (in_ptr1 +(r0_1 +x0 *((1 +ks0 *ks1 *ks2 )//2 )),r0_mask &tmp2 &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp11 =0.0 
        tmp12 =tl .where (tmp9 ,tmp10 ,tmp11 )
        tmp13 =tl .full (tmp12 .shape ,0 ,tmp12 .dtype )
        tmp14 =tl .where (tmp2 ,tmp12 ,tmp13 )
        tmp15 =tl .broadcast_to (tmp14 ,[XBLOCK ,R0_BLOCK ])
        tmp17 =_tmp16 +tmp15 
        _tmp16 =tl .where (r0_mask &xmask ,tmp17 ,_tmp16 )
    tmp16 =tl .sum (_tmp16 ,1 )[:,None ]
    tl .store (out_ptr0 +(x0 ),tmp16 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused_arange_mean_ne_scalar_tensor_where_4 (in_out_ptr0 ,in_ptr0 ,ks0 ,ks1 ,ks2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    R0_BLOCK :tl .constexpr =2 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_0 =r0_index 
    tmp0 =tl .load (in_ptr0 +(r0_0 ),None )
    tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
    tmp3 =tl .sum (tmp1 ,1 )[:,None ]
    tmp4 =ks0 *ks1 *ks2 
    tmp5 =tmp4 .to (tl .float32 )
    tmp6 =tmp3 /tmp5 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp6 ,None )

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
        buf1 =empty_strided_cuda ((1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ),torch .float32 )
        buf2 =empty_strided_cuda ((1 ,s0 *s1 *s2 ),(s0 *s1 *s2 ,1 ),torch .float32 )
        buf3 =reinterpret_tensor (buf1 ,(1 ,s0 *s1 *s2 ),(s0 *s1 *s2 ,1 ),0 );del buf1 

        triton_poi_fused__to_copy_bernoulli_mul_0_xnumel =s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused__to_copy_bernoulli_mul_0 [grid (triton_poi_fused__to_copy_bernoulli_mul_0_xnumel )](buf3 ,buf0 ,arg3_1 ,buf2 ,0 ,1 ,12288 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        buf4 =empty_strided_cuda ((1 ,1 ),(1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused__to_copy_add_bernoulli_gather_mish_mul_softplus_1 [grid (1 )](buf0 ,buf3 ,buf2 ,buf4 ,2 ,3 ,64 ,64 ,1 ,XBLOCK =1 ,num_warps =1 ,num_stages =1 )
        buf5 =buf3 ;del buf3 

        triton_poi_fused__to_copy_add_bernoulli_clamp_min_mish_mul_rsub_softplus_2_xnumel =s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused__to_copy_add_bernoulli_clamp_min_mish_mul_rsub_softplus_2 [grid (triton_poi_fused__to_copy_add_bernoulli_clamp_min_mish_mul_rsub_softplus_2_xnumel )](buf5 ,buf4 ,buf2 ,12288 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf2 
        buf6 =empty_strided_cuda ((2 ,),(1 ,),torch .float32 )

        (1 +s0 *s1 *s2 )//2 
        get_raw_stream (0 )
        triton_red_fused_arange_mean_ne_scalar_tensor_where_3 [grid (2 )](buf0 ,buf5 ,buf6 ,3 ,64 ,64 ,2 ,2 ,6144 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del buf0 
        del buf5 
        buf7 =reinterpret_tensor (buf4 ,(),(),0 );del buf4 
        buf8 =buf7 ;del buf7 

        get_raw_stream (0 )
        triton_per_fused_arange_mean_ne_scalar_tensor_where_4 [grid (1 )](buf8 ,buf6 ,3 ,64 ,64 ,1 ,2 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        del buf6 
    return (buf8 ,)

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
