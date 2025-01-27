
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
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_bernoulli_0 (in_ptr0 ,out_ptr0 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
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
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_copy_1 (in_ptr0 ,in_ptr1 ,out_ptr0 ,ks0 ,ks1 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =xindex //ks0 
    x2 =xindex 
    tmp0 =x0 
    tmp1 =tl .full ([1 ],3 ,tl .int64 )
    tmp2 =tmp0 <tmp1 
    tmp3 =4 +ks1 +x0 
    tmp4 =tl .full ([1 ],3 ,tl .int64 )
    tmp5 =tmp3 >=tmp4 
    tmp6 =tl .broadcast_to (7 +ks1 ,[XBLOCK ])
    tmp7 =tmp3 <tmp6 
    tmp8 =tmp5 &tmp7 
    tmp9 =tmp8 &tmp2 
    tmp10 =(-1 )+ks1 +x0 
    tmp11 =tl .full ([1 ],0 ,tl .int64 )
    tmp12 =tmp10 >=tmp11 
    tmp13 =tl .broadcast_to (ks1 ,[XBLOCK ])
    tmp14 =tmp10 <tmp13 
    tmp15 =tmp12 &tmp14 
    tmp16 =tmp15 &tmp9 
    tmp17 =tl .load (in_ptr0 +((-1 )+ks1 +x0 +ks1 *x1 ),tmp16 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp18 =tl .load (in_ptr1 +(x1 ),tmp9 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp19 =0.5 
    tmp20 =tmp18 <tmp19 
    tmp21 =tmp20 .to (tl .float32 )
    tmp22 =2.0 
    tmp23 =tmp21 *tmp22 
    tmp24 =tmp17 *tmp23 
    tmp25 =tl .full (tmp24 .shape ,0.0 ,tmp24 .dtype )
    tmp26 =tl .where (tmp9 ,tmp24 ,tmp25 )
    tmp27 =float ("nan")
    tmp28 =tl .where (tmp8 ,tmp26 ,tmp27 )
    tmp29 =tl .full (tmp28 .shape ,0.0 ,tmp28 .dtype )
    tmp30 =tl .where (tmp2 ,tmp28 ,tmp29 )
    tmp31 =tmp0 >=tmp1 
    tmp32 =7 +ks1 
    tmp33 =tmp0 <tmp32 
    tmp34 =tmp31 &tmp33 
    tmp35 =(-5 )+x0 
    tmp36 =tl .full ([1 ],0 ,tl .int64 )
    tmp37 =tmp35 >=tmp36 
    tmp38 =tl .broadcast_to (ks1 ,[XBLOCK ])
    tmp39 =tmp35 <tmp38 
    tmp40 =tmp37 &tmp39 
    tmp41 =tmp40 &tmp34 
    tmp42 =tl .load (in_ptr0 +((-5 )+x0 +ks1 *x1 ),tmp41 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp43 =tl .load (in_ptr1 +(x1 ),tmp34 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp44 =0.5 
    tmp45 =tmp43 <tmp44 
    tmp46 =tmp45 .to (tl .float32 )
    tmp47 =2.0 
    tmp48 =tmp46 *tmp47 
    tmp49 =tmp42 *tmp48 
    tmp50 =tl .full (tmp49 .shape ,0.0 ,tmp49 .dtype )
    tmp51 =tl .where (tmp34 ,tmp49 ,tmp50 )
    tmp52 =float ("nan")
    tmp53 =tl .where (tmp34 ,tmp51 ,tmp52 )
    tmp54 =tl .where (tmp2 ,tmp30 ,tmp53 )
    tl .store (out_ptr0 +(x2 ),tmp54 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
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
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_copy_3 (in_ptr0 ,in_ptr1 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =xindex //ks0 
    x2 =xindex 
    tmp0 =x0 
    tmp1 =tl .full ([1 ],2 ,tl .int64 )
    tmp2 =tmp0 <tmp1 
    tmp3 =12 +ks1 +x0 
    tmp4 =tl .full ([1 ],2 ,tl .int64 )
    tmp5 =tmp3 >=tmp4 
    tmp6 =tl .broadcast_to (14 +ks1 ,[XBLOCK ])
    tmp7 =tmp3 <tmp6 
    tmp8 =tmp5 &tmp7 
    tmp9 =tmp8 &tmp2 
    tmp10 =9 +ks1 +x0 
    tmp11 =tl .full ([1 ],0 ,tl .int64 )
    tmp12 =tmp10 >=tmp11 
    tmp13 =tl .broadcast_to (ks2 ,[XBLOCK ])
    tmp14 =tmp10 <tmp13 
    tmp15 =tmp12 &tmp14 
    tmp16 =tmp15 &tmp9 
    tmp17 =9 +ks1 +x0 
    tmp18 =tl .broadcast_to (7 +ks1 ,[XBLOCK ])
    tmp19 =tmp17 >=tmp18 
    tmp20 =tmp19 &tmp16 
    tmp21 =tl .load (in_ptr0 +(5 +x0 +10 *x1 +ks1 *x1 ),tmp20 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp22 =tl .load (in_ptr0 +(9 +ks1 +x0 +10 *x1 +ks1 *x1 ),tmp16 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp23 =tl .where (tmp19 ,tmp21 ,tmp22 )
    tmp24 =tl .full (tmp23 .shape ,0.0 ,tmp23 .dtype )
    tmp25 =tl .where (tmp16 ,tmp23 ,tmp24 )
    tmp26 =tl .load (in_ptr1 +(x1 ),tmp9 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp27 =0.5 
    tmp28 =tmp26 <tmp27 
    tmp29 =tmp28 .to (tl .float32 )
    tmp30 =2.0 
    tmp31 =tmp29 *tmp30 
    tmp32 =tmp25 *tmp31 
    tmp33 =tl .full (tmp32 .shape ,0.0 ,tmp32 .dtype )
    tmp34 =tl .where (tmp9 ,tmp32 ,tmp33 )
    tmp35 =float ("nan")
    tmp36 =tl .where (tmp8 ,tmp34 ,tmp35 )
    tmp37 =tl .full (tmp36 .shape ,0.0 ,tmp36 .dtype )
    tmp38 =tl .where (tmp2 ,tmp36 ,tmp37 )
    tmp39 =tmp0 >=tmp1 
    tmp40 =14 +ks1 
    tmp41 =tmp0 <tmp40 
    tmp42 =tmp39 &tmp41 
    tmp43 =(-3 )+x0 
    tmp44 =tl .full ([1 ],0 ,tl .int64 )
    tmp45 =tmp43 >=tmp44 
    tmp46 =tl .broadcast_to (ks2 ,[XBLOCK ])
    tmp47 =tmp43 <tmp46 
    tmp48 =tmp45 &tmp47 
    tmp49 =tmp48 &tmp42 
    tmp50 =(-3 )+x0 
    tmp51 =tl .broadcast_to (7 +ks1 ,[XBLOCK ])
    tmp52 =tmp50 >=tmp51 
    tmp53 =tmp52 &tmp49 
    tmp54 =tl .load (in_ptr0 +((-7 )+x0 +((-1 )*ks1 )+10 *x1 +ks1 *x1 ),tmp53 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp55 =tl .load (in_ptr0 +((-3 )+x0 +10 *x1 +ks1 *x1 ),tmp49 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp56 =tl .where (tmp52 ,tmp54 ,tmp55 )
    tmp57 =tl .full (tmp56 .shape ,0.0 ,tmp56 .dtype )
    tmp58 =tl .where (tmp49 ,tmp56 ,tmp57 )
    tmp59 =tl .load (in_ptr1 +(x1 ),tmp42 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp60 =0.5 
    tmp61 =tmp59 <tmp60 
    tmp62 =tmp61 .to (tl .float32 )
    tmp63 =2.0 
    tmp64 =tmp62 *tmp63 
    tmp65 =tmp58 *tmp64 
    tmp66 =tl .full (tmp65 .shape ,0.0 ,tmp65 .dtype )
    tmp67 =tl .where (tmp42 ,tmp65 ,tmp66 )
    tmp68 =float ("nan")
    tmp69 =tl .where (tmp42 ,tmp67 ,tmp68 )
    tmp70 =tl .where (tmp2 ,tmp38 ,tmp69 )
    tl .store (out_ptr0 +(x2 ),tmp70 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_4 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x2 =xindex 
    tmp4 =tl .load (in_ptr0 +(x2 ),xmask ,eviction_policy ='evict_last')
    tmp0 =x0 
    tmp1 =14 +ks1 
    tmp2 =tmp0 >=tmp1 
    tmp3 =tl .load (in_ptr0 +((-12 )+x2 +((-1 )*ks1 )),tmp2 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp5 =tl .where (tmp2 ,tmp3 ,tmp4 )
    tl .store (out_ptr0 +(x2 ),tmp5 ,xmask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    assert_size_stride (arg2_1 ,(1 ,s0 ,s1 ),(s0 *s1 ,s1 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf1 =empty_strided_cuda ((2 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[2 ],out =buf1 )
        buf2 =empty_strided_cuda ((1 ,s0 ,1 ,1 ),(s0 ,1 ,s0 ,s0 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_bernoulli_0 [grid (s0 )](buf1 ,buf2 ,0 ,3 ,XBLOCK =4 ,num_warps =1 ,num_stages =1 )
        10 +s1 
        buf3 =empty_strided_cuda ((1 ,s0 ,10 +s1 ),(10 *s0 +s0 *s1 ,10 +s1 ,1 ),torch .float32 )

        triton_poi_fused_copy_1_xnumel =10 *s0 +s0 *s1 
        get_raw_stream (0 )
        triton_poi_fused_copy_1 [grid (triton_poi_fused_copy_1_xnumel )](arg2_1 ,buf2 ,buf3 ,74 ,64 ,222 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del arg2_1 
        buf5 =buf2 ;del buf2 

        get_raw_stream (0 )
        triton_poi_fused_bernoulli_2 [grid (s0 )](buf1 ,buf5 ,1 ,3 ,XBLOCK =4 ,num_warps =1 ,num_stages =1 )
        del buf1 
        16 +s1 
        buf6 =empty_strided_cuda ((1 ,s0 ,16 +s1 ),(16 *s0 +s0 *s1 ,16 +s1 ,1 ),torch .float32 )

        triton_poi_fused_copy_3_xnumel =16 *s0 +s0 *s1 
        get_raw_stream (0 )
        triton_poi_fused_copy_3 [grid (triton_poi_fused_copy_3_xnumel )](buf3 ,buf5 ,buf6 ,80 ,64 ,74 ,240 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del buf3 
        del buf5 
        buf7 =empty_strided_cuda ((1 ,s0 ,16 +s1 ),(16 *s0 +s0 *s1 ,16 +s1 ,1 ),torch .float32 )

        triton_poi_fused_4_xnumel =16 *s0 +s0 *s1 
        get_raw_stream (0 )
        triton_poi_fused_4 [grid (triton_poi_fused_4_xnumel )](buf6 ,buf7 ,80 ,64 ,240 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del buf6 
    return (buf7 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =3 
    arg1_1 =64 
    arg2_1 =rand_strided ((1 ,3 ,64 ),(192 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
