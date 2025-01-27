
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
def triton_poi_fused_abs_gt_mul_reflection_pad1d_sign_sub_where_0 (in_ptr0 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %68 )
    x1 =xindex //68 
    x2 =xindex 
    tmp0 =tl .load (in_ptr0 +(126 +((-2 )*tl_math .abs ((-63 )+tl_math .abs ((-2 )+x0 )))+128 *x1 ),xmask ,eviction_policy ='evict_last')
    tmp1 =tl .load (in_ptr0 +(127 +((-2 )*tl_math .abs ((-63 )+tl_math .abs ((-2 )+x0 )))+128 *x1 ),xmask ,eviction_policy ='evict_last')
    tmp2 =triton_helpers .maximum (tmp1 ,tmp0 )
    tmp3 =tl_math .abs (tmp2 )
    tmp4 =0.5 
    tmp5 =tmp3 >tmp4 
    tmp6 =tl .full ([1 ],0 ,tl .int32 )
    tmp7 =tmp6 <tmp2 
    tmp8 =tmp7 .to (tl .int8 )
    tmp9 =tmp2 <tmp6 
    tmp10 =tmp9 .to (tl .int8 )
    tmp11 =tmp8 -tmp10 
    tmp12 =tmp11 .to (tmp2 .dtype )
    tmp13 =tmp12 *tmp4 
    tmp14 =tmp2 -tmp13 
    tmp15 =0.0 
    tmp16 =tmp2 *tmp15 
    tmp17 =tl .where (tmp5 ,tmp14 ,tmp16 )
    tl .store (out_ptr0 +(x2 ),tmp17 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__adaptive_avg_pool2d_1 (in_ptr0 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %10 )
    x1 =xindex //10 
    x2 =xindex 
    tmp0 =tl .full ([1 ],0 ,tl .int64 )
    tmp1 =tl .full ([1 ],1 ,tl .int64 )
    tmp2 =tmp0 <tmp1 
    tmp3 =(34 *x0 )//5 
    tmp4 =(77 +68 *x0 )//10 
    tmp5 =tmp3 <tmp4 
    tmp6 =tmp2 &tmp5 
    tmp7 =tl .load (in_ptr0 +(68 *x1 +((34 *x0 )//5 )),tmp6 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp8 =1 +((34 *x0 )//5 )
    tmp9 =tmp8 <tmp4 
    tmp10 =tmp2 &tmp9 
    tmp11 =tl .load (in_ptr0 +(1 +68 *x1 +((34 *x0 )//5 )),tmp10 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp12 =tmp11 +tmp7 
    tmp13 =2 +((34 *x0 )//5 )
    tmp14 =tmp13 <tmp4 
    tmp15 =tmp2 &tmp14 
    tmp16 =tl .load (in_ptr0 +(2 +68 *x1 +((34 *x0 )//5 )),tmp15 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp17 =tmp16 +tmp12 
    tmp18 =3 +((34 *x0 )//5 )
    tmp19 =tmp18 <tmp4 
    tmp20 =tmp2 &tmp19 
    tmp21 =tl .load (in_ptr0 +(3 +68 *x1 +((34 *x0 )//5 )),tmp20 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp22 =tmp21 +tmp17 
    tmp23 =4 +((34 *x0 )//5 )
    tmp24 =tmp23 <tmp4 
    tmp25 =tmp2 &tmp24 
    tmp26 =tl .load (in_ptr0 +(4 +68 *x1 +((34 *x0 )//5 )),tmp25 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp27 =tmp26 +tmp22 
    tmp28 =5 +((34 *x0 )//5 )
    tmp29 =tmp28 <tmp4 
    tmp30 =tmp2 &tmp29 
    tmp31 =tl .load (in_ptr0 +(5 +68 *x1 +((34 *x0 )//5 )),tmp30 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp32 =tmp31 +tmp27 
    tmp33 =6 +((34 *x0 )//5 )
    tmp34 =tmp33 <tmp4 
    tmp35 =tmp2 &tmp34 
    tmp36 =tl .load (in_ptr0 +(6 +68 *x1 +((34 *x0 )//5 )),tmp35 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp37 =tmp36 +tmp32 
    tmp38 =7 +((34 *x0 )//5 )
    tmp39 =tmp38 <tmp4 
    tmp40 =tmp2 &tmp39 
    tmp41 =tl .load (in_ptr0 +(7 +68 *x1 +((34 *x0 )//5 )),tmp40 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp42 =tmp41 +tmp37 
    tmp43 =1.0 
    tmp44 =tl .full (tmp43 .shape ,0.0 ,tmp43 .dtype )
    tmp45 =tl .where (tmp6 ,tmp43 ,tmp44 )
    tmp46 =1.0 
    tmp47 =tl .full (tmp46 .shape ,0.0 ,tmp46 .dtype )
    tmp48 =tl .where (tmp10 ,tmp46 ,tmp47 )
    tmp49 =tmp48 +tmp45 
    tmp50 =1.0 
    tmp51 =tl .full (tmp50 .shape ,0.0 ,tmp50 .dtype )
    tmp52 =tl .where (tmp15 ,tmp50 ,tmp51 )
    tmp53 =tmp52 +tmp49 
    tmp54 =1.0 
    tmp55 =tl .full (tmp54 .shape ,0.0 ,tmp54 .dtype )
    tmp56 =tl .where (tmp20 ,tmp54 ,tmp55 )
    tmp57 =tmp56 +tmp53 
    tmp58 =1.0 
    tmp59 =tl .full (tmp58 .shape ,0.0 ,tmp58 .dtype )
    tmp60 =tl .where (tmp25 ,tmp58 ,tmp59 )
    tmp61 =tmp60 +tmp57 
    tmp62 =1.0 
    tmp63 =tl .full (tmp62 .shape ,0.0 ,tmp62 .dtype )
    tmp64 =tl .where (tmp30 ,tmp62 ,tmp63 )
    tmp65 =tmp64 +tmp61 
    tmp66 =1.0 
    tmp67 =tl .full (tmp66 .shape ,0.0 ,tmp66 .dtype )
    tmp68 =tl .where (tmp35 ,tmp66 ,tmp67 )
    tmp69 =tmp68 +tmp65 
    tmp70 =1.0 
    tmp71 =tl .full (tmp70 .shape ,0.0 ,tmp70 .dtype )
    tmp72 =tl .where (tmp40 ,tmp70 ,tmp71 )
    tmp73 =tmp72 +tmp69 
    tmp74 =tmp42 /tmp73 
    tl .store (out_ptr0 +(x2 ),tmp74 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_bernoulli_2 (in_ptr0 ,out_ptr0 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    tl .full ([XBLOCK ],True ,tl .int1 )
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =tl .full ([1 ],0 ,tl .int32 )
    tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tl .store (out_ptr0 +(tl .full ([XBLOCK ],0 ,tl .int32 )),tmp2 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_abs_gt_mul_reflection_pad1d_sign_sub_where_3 (in_ptr0 ,in_ptr1 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %9 )
    x1 =xindex //9 
    x2 =xindex 
    tmp0 =tl .load (in_ptr0 +(8 +((-2 )*tl_math .abs ((-4 )+tl_math .abs ((-2 )+x0 )))+10 *x1 ),xmask ,eviction_policy ='evict_last')
    tmp1 =tl .load (in_ptr1 +(0 ))
    tmp2 =tl .broadcast_to (tmp1 ,[XBLOCK ])
    tmp9 =tl .load (in_ptr0 +(9 +((-2 )*tl_math .abs ((-4 )+tl_math .abs ((-2 )+x0 )))+10 *x1 ),xmask ,eviction_policy ='evict_last')
    tmp3 =0.5 
    tmp4 =tmp2 <tmp3 
    tmp5 =tmp4 .to (tl .float32 )
    tmp6 =2.0 
    tmp7 =tmp5 *tmp6 
    tmp8 =tmp0 *tmp7 
    tmp10 =tmp9 *tmp7 
    tmp11 =triton_helpers .maximum (tmp10 ,tmp8 )
    tmp12 =tl_math .abs (tmp11 )
    tmp13 =tmp12 >tmp3 
    tmp14 =tl .full ([1 ],0 ,tl .int32 )
    tmp15 =tmp14 <tmp11 
    tmp16 =tmp15 .to (tl .int8 )
    tmp17 =tmp11 <tmp14 
    tmp18 =tmp17 .to (tl .int8 )
    tmp19 =tmp16 -tmp18 
    tmp20 =tmp19 .to (tmp11 .dtype )
    tmp21 =tmp20 *tmp3 
    tmp22 =tmp11 -tmp21 
    tmp23 =0.0 
    tmp24 =tmp11 *tmp23 
    tmp25 =tl .where (tmp13 ,tmp22 ,tmp24 )
    tl .store (out_ptr0 +(x2 ),tmp25 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_bernoulli_4 (in_ptr0 ,out_ptr0 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    tl .full ([XBLOCK ],True ,tl .int1 )
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =tl .full ([1 ],0 ,tl .int32 )
    tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tl .store (out_ptr0 +(tl .full ([XBLOCK ],0 ,tl .int32 )),tmp2 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__to_copy_bernoulli_div_mul_5 (in_ptr0 ,in_ptr1 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %5 )
    x1 =xindex //5 
    x2 =xindex 
    tmp30 =tl .load (in_ptr1 +(0 ))
    tmp31 =tl .broadcast_to (tmp30 ,[XBLOCK ])
    tmp0 =tl .full ([1 ],0 ,tl .int64 )
    tmp1 =tl .full ([1 ],1 ,tl .int64 )
    tmp2 =tmp0 <tmp1 
    tmp3 =(9 *x0 )//5 
    tmp4 =(13 +9 *x0 )//5 
    tmp5 =tmp3 <tmp4 
    tmp6 =tmp2 &tmp5 
    tmp7 =tl .load (in_ptr0 +(9 *x1 +((9 *x0 )//5 )),tmp6 &xmask ,other =0.0 )
    tmp8 =1 +((9 *x0 )//5 )
    tmp9 =tmp8 <tmp4 
    tmp10 =tmp2 &tmp9 
    tmp11 =tl .load (in_ptr0 +(1 +9 *x1 +((9 *x0 )//5 )),tmp10 &xmask ,other =0.0 )
    tmp12 =tmp11 +tmp7 
    tmp13 =2 +((9 *x0 )//5 )
    tmp14 =tmp13 <tmp4 
    tmp15 =tmp2 &tmp14 
    tmp16 =tl .load (in_ptr0 +(2 +9 *x1 +((9 *x0 )//5 )),tmp15 &xmask ,other =0.0 )
    tmp17 =tmp16 +tmp12 
    tmp18 =1.0 
    tmp19 =tl .full (tmp18 .shape ,0.0 ,tmp18 .dtype )
    tmp20 =tl .where (tmp6 ,tmp18 ,tmp19 )
    tmp21 =1.0 
    tmp22 =tl .full (tmp21 .shape ,0.0 ,tmp21 .dtype )
    tmp23 =tl .where (tmp10 ,tmp21 ,tmp22 )
    tmp24 =tmp23 +tmp20 
    tmp25 =1.0 
    tmp26 =tl .full (tmp25 .shape ,0.0 ,tmp25 .dtype )
    tmp27 =tl .where (tmp15 ,tmp25 ,tmp26 )
    tmp28 =tmp27 +tmp24 
    tmp29 =tmp17 /tmp28 
    tmp32 =0.5 
    tmp33 =tmp31 <tmp32 
    tmp34 =tmp33 .to (tl .float32 )
    tmp35 =2.0 
    tmp36 =tmp34 *tmp35 
    tmp37 =tmp29 *tmp36 
    tl .store (out_ptr0 +(x2 ),tmp37 ,xmask )

def call (args ):
    arg0_1 ,arg1_1 =args 
    args .clear ()
    s0 =arg0_1 
    assert_size_stride (arg1_1 ,(1 ,s0 ,128 ),(128 *s0 ,128 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,s0 ,68 ),(68 *s0 ,68 ,1 ),torch .float32 )

        triton_poi_fused_abs_gt_mul_reflection_pad1d_sign_sub_where_0_xnumel =68 *s0 
        get_raw_stream (0 )
        triton_poi_fused_abs_gt_mul_reflection_pad1d_sign_sub_where_0 [grid (triton_poi_fused_abs_gt_mul_reflection_pad1d_sign_sub_where_0_xnumel )](arg1_1 ,buf0 ,204 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del arg1_1 
        buf1 =empty_strided_cuda ((1 ,s0 ,1 ,10 ),(10 *s0 ,10 ,10 ,1 ),torch .float32 )

        triton_poi_fused__adaptive_avg_pool2d_1_xnumel =10 *s0 
        get_raw_stream (0 )
        triton_poi_fused__adaptive_avg_pool2d_1 [grid (triton_poi_fused__adaptive_avg_pool2d_1_xnumel )](buf0 ,buf1 ,30 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        del buf0 
        buf2 =empty_strided_cuda ((2 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[2 ],out =buf2 )
        buf3 =empty_strided_cuda ((1 ,1 ,1 ,1 ,1 ),(1 ,1 ,1 ,1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_bernoulli_2 [grid (1 )](buf2 ,buf3 ,0 ,1 ,XBLOCK =1 ,num_warps =1 ,num_stages =1 )
        buf4 =empty_strided_cuda ((1 ,s0 ,9 ),(9 *s0 ,9 ,1 ),torch .float32 )

        triton_poi_fused_abs_gt_mul_reflection_pad1d_sign_sub_where_3_xnumel =9 *s0 
        get_raw_stream (0 )
        triton_poi_fused_abs_gt_mul_reflection_pad1d_sign_sub_where_3 [grid (triton_poi_fused_abs_gt_mul_reflection_pad1d_sign_sub_where_3_xnumel )](buf1 ,buf3 ,buf4 ,27 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        del buf1 
        buf5 =buf3 ;del buf3 

        get_raw_stream (0 )
        triton_poi_fused_bernoulli_4 [grid (1 )](buf2 ,buf5 ,1 ,1 ,XBLOCK =1 ,num_warps =1 ,num_stages =1 )
        del buf2 
        buf6 =empty_strided_cuda ((1 ,1 ,s0 ,1 ,5 ),(5 *s0 ,5 *s0 ,5 ,5 ,1 ),torch .float32 )

        triton_poi_fused__to_copy_bernoulli_div_mul_5_xnumel =5 *s0 
        get_raw_stream (0 )
        triton_poi_fused__to_copy_bernoulli_div_mul_5 [grid (triton_poi_fused__to_copy_bernoulli_div_mul_5_xnumel )](buf4 ,buf5 ,buf6 ,15 ,XBLOCK =16 ,num_warps =1 ,num_stages =1 )
        del buf4 
        del buf5 
    return (reinterpret_tensor (buf6 ,(1 ,s0 ,5 ),(5 *s0 ,5 ,1 ),0 ),)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =3 
    arg1_1 =rand_strided ((1 ,3 ,128 ),(384 ,128 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
