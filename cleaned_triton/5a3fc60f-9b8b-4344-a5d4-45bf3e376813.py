
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
def triton_poi_fused_copy_0 (in_ptr0 ,in_ptr1 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%ks2 )
    x2 =xindex //ks4 
    x3 =xindex 
    tmp0 =x0 
    tmp1 =tl .full ([1 ],2 ,tl .int64 )
    tmp2 =tmp0 >=tmp1 
    tmp3 =6 +ks1 
    tmp4 =tmp0 <tmp3 
    tmp5 =tmp2 &tmp4 
    tmp6 =x1 
    tmp7 =tl .full ([1 ],2 ,tl .int64 )
    tmp8 =tmp6 >=tmp7 
    tmp9 =tl .broadcast_to (6 +ks3 ,[XBLOCK ])
    tmp10 =tmp6 <tmp9 
    tmp11 =tmp8 &tmp10 
    tmp12 =tmp11 &tmp5 
    tmp13 =(-4 )+x1 
    tmp14 =tl .full ([1 ],0 ,tl .int64 )
    tmp15 =tmp13 >=tmp14 
    tmp16 =tl .broadcast_to (ks3 ,[XBLOCK ])
    tmp17 =tmp13 <tmp16 
    tmp18 =(-4 )+x0 
    tmp19 =tmp18 >=tmp14 
    tmp20 =tl .broadcast_to (ks1 ,[XBLOCK ])
    tmp21 =tmp18 <tmp20 
    tmp22 =tmp15 &tmp17 
    tmp23 =tmp22 &tmp19 
    tmp24 =tmp23 &tmp21 
    tmp25 =tmp24 &tmp12 
    tmp26 =tl .load (in_ptr0 +((-4 )+x0 +((-4 )*ks1 )+ks1 *x1 +ks1 *ks3 *x2 ),tmp25 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp27 =tl .full (tmp26 .shape ,0.0 ,tmp26 .dtype )
    tmp28 =tl .where (tmp12 ,tmp26 ,tmp27 )
    tmp29 =tl .load (in_ptr1 +(x3 ),tmp5 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp30 =tl .where (tmp11 ,tmp28 ,tmp29 )
    tmp31 =tl .full (tmp30 .shape ,0.0 ,tmp30 .dtype )
    tmp32 =tl .where (tmp5 ,tmp30 ,tmp31 )
    tmp33 =float ("nan")
    tmp34 =tl .where (tmp5 ,tmp32 ,tmp33 )
    tl .store (out_ptr0 +(x3 ),tmp34 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_1 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x1 =((xindex //ks0 )%ks1 )
    x0 =(xindex %ks0 )
    x3 =xindex 
    tmp39 =tl .load (in_ptr0 +(x3 ),xmask ,eviction_policy ='evict_last')
    tmp0 =x1 
    tmp1 =tl .full ([1 ],2 ,tl .int64 )
    tmp2 =tmp0 <tmp1 
    tmp3 =x0 
    tmp4 =tl .broadcast_to (6 +ks2 ,[XBLOCK ])
    tmp5 =tmp3 >=tmp4 
    tmp6 =tmp5 &tmp2 
    tmp7 =(-4 )+x0 +((-1 )*ks2 )
    tmp8 =tl .full ([1 ],2 ,tl .int64 )
    tmp9 =tmp7 <tmp8 
    tmp10 =tmp9 &tmp6 
    tmp11 =tl .load (in_ptr0 +(32 +x3 +4 *ks2 +8 *ks3 +ks2 *ks3 ),tmp10 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp12 =tl .load (in_ptr0 +(28 +x3 +3 *ks2 +8 *ks3 +ks2 *ks3 ),tmp6 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp13 =tl .where (tmp9 ,tmp11 ,tmp12 )
    tmp14 =tl .full (tmp13 .shape ,0.0 ,tmp13 .dtype )
    tmp15 =tl .where (tmp6 ,tmp13 ,tmp14 )
    tmp16 =tl .full ([1 ],2 ,tl .int64 )
    tmp17 =tmp3 <tmp16 
    tmp18 =tmp17 &tmp2 
    tmp19 =tl .load (in_ptr0 +(36 +x3 +5 *ks2 +8 *ks3 +ks2 *ks3 ),tmp18 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp20 =tl .load (in_ptr0 +(32 +x3 +4 *ks2 +8 *ks3 +ks2 *ks3 ),tmp2 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp21 =tl .where (tmp17 ,tmp19 ,tmp20 )
    tmp22 =tl .where (tmp5 ,tmp15 ,tmp21 )
    tmp23 =tl .full (tmp22 .shape ,0.0 ,tmp22 .dtype )
    tmp24 =tl .where (tmp2 ,tmp22 ,tmp23 )
    tmp25 =x0 
    tmp26 =6 +ks2 
    tmp27 =tmp25 >=tmp26 
    tmp28 =(-4 )+x0 +((-1 )*ks2 )
    tmp29 =tl .full ([1 ],2 ,tl .int64 )
    tmp30 =tmp28 <tmp29 
    tmp31 =tmp30 &tmp27 
    tmp32 =tl .load (in_ptr0 +(x3 ),tmp31 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp33 =tl .load (in_ptr0 +((-4 )+x3 +((-1 )*ks2 )),tmp27 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp34 =tl .where (tmp30 ,tmp32 ,tmp33 )
    tmp35 =tl .full (tmp34 .shape ,0.0 ,tmp34 .dtype )
    tmp36 =tl .where (tmp27 ,tmp34 ,tmp35 )
    tmp37 =tmp25 <tmp1 
    tmp38 =tl .load (in_ptr0 +(4 +ks2 +x3 ),tmp37 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp40 =tl .where (tmp37 ,tmp38 ,tmp39 )
    tmp41 =tl .where (tmp27 ,tmp36 ,tmp40 )
    tmp42 =tl .where (tmp2 ,tmp24 ,tmp41 )
    tl .store (out_ptr0 +(x3 ),tmp42 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_2 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x1 =((xindex //ks0 )%ks1 )
    x3 =xindex 
    tmp4 =tl .load (in_ptr0 +(x3 ),xmask ,eviction_policy ='evict_last')
    tmp0 =x1 
    tmp1 =6 +ks2 
    tmp2 =tmp0 >=tmp1 
    tmp3 =tl .load (in_ptr0 +((-32 )+x3 +((-8 )*ks2 )+((-4 )*ks3 )+((-1 )*ks2 *ks3 )),tmp2 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp5 =tl .where (tmp2 ,tmp3 ,tmp4 )
    tl .store (out_ptr0 +(x3 ),tmp5 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_avg_pool2d_3 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%ks1 )
    x2 =xindex //ks2 
    x3 =xindex 
    tmp0 =tl .load (in_ptr0 +(2 *x0 +16 *x1 +64 *x2 +2 *ks4 *x1 +8 *ks3 *x2 +8 *ks4 *x2 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp1 =tl .load (in_ptr0 +(1 +2 *x0 +16 *x1 +64 *x2 +2 *ks4 *x1 +8 *ks3 *x2 +8 *ks4 *x2 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp3 =tl .load (in_ptr0 +(8 +ks4 +2 *x0 +16 *x1 +64 *x2 +2 *ks4 *x1 +8 *ks3 *x2 +8 *ks4 *x2 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp5 =tl .load (in_ptr0 +(9 +ks4 +2 *x0 +16 *x1 +64 *x2 +2 *ks4 *x1 +8 *ks3 *x2 +8 *ks4 *x2 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp2 =tmp1 +tmp0 
    tmp4 =tmp3 +tmp2 
    tmp6 =tmp5 +tmp4 
    tmp7 =0.25 
    tmp8 =tmp6 *tmp7 
    tl .store (out_ptr0 +(x3 ),tmp8 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_permute_4 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =xindex //ks0 
    x2 =xindex 
    tmp0 =tl .load (in_ptr0 +(4 *(x0 //ks1 )+16 *x1 +(ks3 //2 )*(x0 //ks1 )+4 *x1 *(ks2 //2 )+4 *x1 *(ks3 //2 )+x1 *(ks2 //2 )*(ks3 //2 )+((x0 %ks1 ))),xmask ,eviction_policy ='evict_last')
    tl .store (out_ptr0 +(x2 ),tmp0 ,xmask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,s0 ,8 +s1 ,8 +s2 ),(64 *s0 +8 *s0 *s1 +8 *s0 *s2 +s0 *s1 *s2 ,64 +8 *s1 +8 *s2 +s1 *s2 ,8 +s2 ,1 ),torch .float32 )
        8 +s2 
        8 +s1 
        64 +8 *s1 +8 *s2 +s1 *s2 
        buf1 =empty_strided_cuda ((1 ,s0 ,8 +s1 ,8 +s2 ),(64 *s0 +8 *s0 *s1 +8 *s0 *s2 +s0 *s1 *s2 ,64 +8 *s1 +8 *s2 +s1 *s2 ,8 +s2 ,1 ),torch .float32 )

        triton_poi_fused_copy_0_xnumel =64 *s0 +8 *s0 *s1 +8 *s0 *s2 +s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_copy_0 [grid (triton_poi_fused_copy_0_xnumel )](arg3_1 ,buf0 ,buf1 ,40 ,32 ,40 ,32 ,1600 ,4800 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        buf2 =buf0 ;del buf0 

        triton_poi_fused_1_xnumel =64 *s0 +8 *s0 *s1 +8 *s0 *s2 +s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_1 [grid (triton_poi_fused_1_xnumel )](buf1 ,buf2 ,40 ,40 ,32 ,32 ,4800 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        buf3 =buf1 ;del buf1 

        triton_poi_fused_2_xnumel =64 *s0 +8 *s0 *s1 +8 *s0 *s2 +s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_2 [grid (triton_poi_fused_2_xnumel )](buf2 ,buf3 ,40 ,40 ,32 ,32 ,4800 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del buf2 
        4 +(s2 //2 )
        4 +(s1 //2 )
        16 +4 *(s1 //2 )+4 *(s2 //2 )+(s1 //2 )*(s2 //2 )
        buf4 =empty_strided_cuda ((1 ,s0 ,4 +(s1 //2 ),4 +(s2 //2 )),(16 *s0 +4 *s0 *(s1 //2 )+4 *s0 *(s2 //2 )+s0 *(s1 //2 )*(s2 //2 ),16 +4 *(s1 //2 )+4 *(s2 //2 )+(s1 //2 )*(s2 //2 ),4 +(s2 //2 ),1 ),torch .float32 )

        triton_poi_fused_avg_pool2d_3_xnumel =16 *s0 +4 *s0 *(s1 //2 )+4 *s0 *(s2 //2 )+s0 *(s1 //2 )*(s2 //2 )
        get_raw_stream (0 )
        triton_poi_fused_avg_pool2d_3 [grid (triton_poi_fused_avg_pool2d_3_xnumel )](buf3 ,buf4 ,20 ,20 ,400 ,32 ,32 ,1200 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf3 
        16 +4 *(s1 //2 )+4 *(s2 //2 )+(s1 //2 )*(s2 //2 )
        buf5 =empty_strided_cuda ((1 ,16 +4 *(s1 //2 )+4 *(s2 //2 )+(s1 //2 )*(s2 //2 ),s0 ),(16 *s0 +4 *s0 *(s1 //2 )+4 *s0 *(s2 //2 )+s0 *(s1 //2 )*(s2 //2 ),1 ,16 +4 *(s1 //2 )+4 *(s2 //2 )+(s1 //2 )*(s2 //2 )),torch .float32 )

        triton_poi_fused_permute_4_xnumel =16 *s0 +4 *s0 *(s1 //2 )+4 *s0 *(s2 //2 )+s0 *(s1 //2 )*(s2 //2 )
        get_raw_stream (0 )
        triton_poi_fused_permute_4 [grid (triton_poi_fused_permute_4_xnumel )](buf4 ,buf5 ,400 ,20 ,32 ,32 ,1200 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf4 
    return (buf5 ,4 +(s1 //2 ),4 +(s2 //2 ),)

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
