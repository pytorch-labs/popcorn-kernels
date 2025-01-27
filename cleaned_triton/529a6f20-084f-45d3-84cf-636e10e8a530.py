
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
def triton_poi_fused_avg_pool3d_constant_pad_nd_0 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x1 =xindex //ks0 
    x2 =xindex 
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
    tl .store (out_ptr0 +(x2 ),tmp47 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_copy_1 (in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %3 )
    x1 =((xindex //3 )%ks0 )
    x2 =((xindex //ks2 )%ks3 )
    x3 =xindex //ks5 
    x5 =xindex //3 
    x4 =xindex 
    tmp0 =x0 
    tmp1 =tl .full ([1 ],1 ,tl .int64 )
    tmp2 =tmp0 >=tmp1 
    tmp3 =tl .full ([1 ],2 ,tl .int64 )
    tmp4 =tmp0 <tmp3 
    tmp5 =tmp2 &tmp4 
    tmp6 =x1 
    tmp7 =tl .full ([1 ],1 ,tl .int64 )
    tmp8 =tmp6 >=tmp7 
    tmp9 =tl .broadcast_to (1 +ks1 ,[XBLOCK ])
    tmp10 =tmp6 <tmp9 
    tmp11 =tmp8 &tmp10 
    tmp12 =tmp11 &tmp5 
    tmp13 =x2 
    tmp14 =tl .full ([1 ],1 ,tl .int64 )
    tmp15 =tmp13 >=tmp14 
    tmp16 =tl .broadcast_to (1 +ks4 ,[XBLOCK ])
    tmp17 =tmp13 <tmp16 
    tmp18 =tmp15 &tmp17 
    tmp19 =tmp18 &tmp12 
    tmp20 =tl .load (in_ptr0 +((-1 )+x1 +((-1 )*ks1 )+ks1 *x2 +ks1 *ks4 *x3 ),tmp19 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp21 =tl .load (in_ptr1 +((-1 )+x1 +((-1 )*ks1 )+ks1 *x2 +ks1 *ks4 *x3 ),tmp19 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp22 =0.0001 
    tmp23 =tmp21 *tmp22 
    tmp24 =1.0 
    tmp25 =tmp23 +tmp24 
    tmp26 =0.75 
    tmp27 =libdevice .pow (tmp25 ,tmp26 )
    tmp28 =tmp20 /tmp27 
    tmp29 =tl .full (tmp28 .shape ,0.0 ,tmp28 .dtype )
    tmp30 =tl .where (tmp19 ,tmp28 ,tmp29 )
    tmp31 =tl .load (in_ptr2 +(1 +3 *x5 ),tmp12 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp32 =tl .where (tmp18 ,tmp30 ,tmp31 )
    tmp33 =tl .full (tmp32 .shape ,0.0 ,tmp32 .dtype )
    tmp34 =tl .where (tmp12 ,tmp32 ,tmp33 )
    tmp35 =tl .load (in_ptr2 +(1 +3 *x5 ),tmp5 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp36 =tl .where (tmp11 ,tmp34 ,tmp35 )
    tmp37 =tl .full (tmp36 .shape ,0.0 ,tmp36 .dtype )
    tmp38 =tl .where (tmp5 ,tmp36 ,tmp37 )
    tmp39 =float ("nan")
    tmp40 =tl .where (tmp5 ,tmp38 ,tmp39 )
    tl .store (out_ptr0 +(x4 ),tmp40 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_2 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x1 =((xindex //3 )%ks0 )
    x0 =(xindex %3 )
    x3 =xindex //3 
    x4 =xindex 
    tmp39 =tl .load (in_ptr0 +(x4 ),xmask ,eviction_policy ='evict_last')
    tmp0 =x1 
    tmp1 =tl .full ([1 ],1 ,tl .int64 )
    tmp2 =tmp0 <tmp1 
    tmp3 =x0 
    tmp4 =tl .full ([1 ],2 ,tl .int64 )
    tmp5 =tmp3 >=tmp4 
    tmp6 =tmp5 &tmp2 
    tmp7 =(-1 )+x0 
    tmp8 =tl .full ([1 ],1 ,tl .int64 )
    tmp9 =tmp7 <tmp8 
    tmp10 =tmp9 &tmp6 
    tmp11 =tl .load (in_ptr0 +(1 +3 *ks1 +3 *x3 ),tmp10 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp12 =tl .load (in_ptr0 +((-1 )+x4 +3 *ks1 ),tmp6 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp13 =tl .where (tmp9 ,tmp11 ,tmp12 )
    tmp14 =tl .full (tmp13 .shape ,0.0 ,tmp13 .dtype )
    tmp15 =tl .where (tmp6 ,tmp13 ,tmp14 )
    tmp16 =tl .full ([1 ],1 ,tl .int64 )
    tmp17 =tmp3 <tmp16 
    tmp18 =tmp17 &tmp2 
    tmp19 =tl .load (in_ptr0 +(1 +3 *ks1 +3 *x3 ),tmp18 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp20 =tl .load (in_ptr0 +(x4 +3 *ks1 ),tmp2 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp21 =tl .where (tmp17 ,tmp19 ,tmp20 )
    tmp22 =tl .where (tmp5 ,tmp15 ,tmp21 )
    tmp23 =tl .full (tmp22 .shape ,0.0 ,tmp22 .dtype )
    tmp24 =tl .where (tmp2 ,tmp22 ,tmp23 )
    tmp25 =x0 
    tmp26 =tl .full ([1 ],2 ,tl .int64 )
    tmp27 =tmp25 >=tmp26 
    tmp28 =(-1 )+x0 
    tmp29 =tl .full ([1 ],1 ,tl .int64 )
    tmp30 =tmp28 <tmp29 
    tmp31 =tmp30 &tmp27 
    tmp32 =tl .load (in_ptr0 +(1 +3 *x3 ),tmp31 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp33 =tl .load (in_ptr0 +((-1 )+x4 ),tmp27 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp34 =tl .where (tmp30 ,tmp32 ,tmp33 )
    tmp35 =tl .full (tmp34 .shape ,0.0 ,tmp34 .dtype )
    tmp36 =tl .where (tmp27 ,tmp34 ,tmp35 )
    tmp37 =tmp25 <tmp1 
    tmp38 =tl .load (in_ptr0 +(1 +3 *x3 ),tmp37 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp40 =tl .where (tmp37 ,tmp38 ,tmp39 )
    tmp41 =tl .where (tmp27 ,tmp36 ,tmp40 )
    tmp42 =tl .where (tmp2 ,tmp24 ,tmp41 )
    tl .store (out_ptr0 +(x4 ),tmp42 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_3 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x2 =((xindex //ks0 )%ks1 )
    x1 =((xindex //3 )%ks3 )
    x0 =(xindex %3 )
    x6 =xindex //ks0 
    x4 =xindex 
    tmp41 =tl .load (in_ptr0 +(x4 ),xmask ,eviction_policy ='evict_last')
    tmp0 =x2 
    tmp1 =1 +ks2 
    tmp2 =tmp0 >=tmp1 
    tmp3 =x2 +((-1 )*ks2 )
    tmp4 =tl .full ([1 ],1 ,tl .int64 )
    tmp5 =tmp3 <tmp4 
    tmp6 =tmp5 &tmp2 
    tmp7 =x1 
    tmp8 =tl .broadcast_to (1 +ks4 ,[XBLOCK ])
    tmp9 =tmp7 >=tmp8 
    tmp10 =tmp9 &tmp6 
    tmp11 =tl .load (in_ptr0 +(3 +x0 +6 *x6 +3 *ks4 *x6 ),tmp10 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp12 =tl .load (in_ptr0 +(x4 ),tmp6 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp13 =tl .where (tmp9 ,tmp11 ,tmp12 )
    tmp14 =tl .full (tmp13 .shape ,0.0 ,tmp13 .dtype )
    tmp15 =tl .where (tmp6 ,tmp13 ,tmp14 )
    tmp16 =x1 
    tmp17 =tl .broadcast_to (1 +ks4 ,[XBLOCK ])
    tmp18 =tmp16 >=tmp17 
    tmp19 =tmp18 &tmp2 
    tmp20 =tl .load (in_ptr0 +(3 +x0 +((-6 )*ks2 )+6 *x6 +((-3 )*ks2 *ks4 )+3 *ks4 *x6 ),tmp19 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp21 =tl .load (in_ptr0 +(x4 +((-6 )*ks2 )+((-3 )*ks2 *ks4 )),tmp2 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp22 =tl .where (tmp18 ,tmp20 ,tmp21 )
    tmp23 =tl .where (tmp5 ,tmp15 ,tmp22 )
    tmp24 =tl .full (tmp23 .shape ,0.0 ,tmp23 .dtype )
    tmp25 =tl .where (tmp2 ,tmp23 ,tmp24 )
    tmp26 =tl .full ([1 ],1 ,tl .int64 )
    tmp27 =tmp0 <tmp26 
    tmp28 =x1 
    tmp29 =tl .broadcast_to (1 +ks4 ,[XBLOCK ])
    tmp30 =tmp28 >=tmp29 
    tmp31 =tmp30 &tmp27 
    tmp32 =tl .load (in_ptr0 +(3 +x0 +6 *ks2 +6 *x6 +3 *ks2 *ks4 +3 *ks4 *x6 ),tmp31 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp33 =tl .load (in_ptr0 +(x4 +6 *ks2 +3 *ks2 *ks4 ),tmp27 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp34 =tl .where (tmp30 ,tmp32 ,tmp33 )
    tmp35 =tl .full (tmp34 .shape ,0.0 ,tmp34 .dtype )
    tmp36 =tl .where (tmp27 ,tmp34 ,tmp35 )
    tmp37 =x1 
    tmp38 =1 +ks4 
    tmp39 =tmp37 >=tmp38 
    tmp40 =tl .load (in_ptr0 +(3 +x0 +6 *x6 +3 *ks4 *x6 ),tmp39 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp42 =tl .where (tmp39 ,tmp40 ,tmp41 )
    tmp43 =tl .where (tmp27 ,tmp36 ,tmp42 )
    tmp44 =tl .where (tmp2 ,tmp25 ,tmp43 )
    tl .store (out_ptr0 +(x4 ),tmp44 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_4 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =xindex //ks0 
    x2 =xindex 
    tmp0 =tl .load (in_ptr0 +(6 *(x0 //ks1 )+12 *x1 +3 *ks3 *(x0 //ks1 )+6 *ks2 *x1 +6 *ks3 *x1 +3 *ks2 *ks3 *x1 +((x0 %ks1 ))),xmask ,eviction_policy ='evict_last')
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
        buf0 =empty_strided_cuda ((1 ,s0 ,2 +s1 ,2 +s2 ,3 ),(12 *s0 +6 *s0 *s1 +6 *s0 *s2 +3 *s0 *s1 *s2 ,12 +6 *s1 +6 *s2 +3 *s1 *s2 ,6 +3 *s2 ,3 ,1 ),torch .float32 )
        s1 *s2 
        buf1 =empty_strided_cuda ((1 ,1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ),torch .float32 )

        triton_poi_fused_avg_pool3d_constant_pad_nd_0_xnumel =s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_avg_pool3d_constant_pad_nd_0 [grid (triton_poi_fused_avg_pool3d_constant_pad_nd_0_xnumel )](arg3_1 ,buf1 ,1024 ,3 ,32 ,32 ,3072 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        2 +s2 
        6 +3 *s2 
        2 +s1 
        12 +6 *s1 +6 *s2 +3 *s1 *s2 
        buf2 =empty_strided_cuda ((1 ,s0 ,2 +s1 ,2 +s2 ,3 ),(12 *s0 +6 *s0 *s1 +6 *s0 *s2 +3 *s0 *s1 *s2 ,12 +6 *s1 +6 *s2 +3 *s1 *s2 ,6 +3 *s2 ,3 ,1 ),torch .float32 )

        triton_poi_fused_copy_1_xnumel =12 *s0 +6 *s0 *s1 +6 *s0 *s2 +3 *s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_copy_1 [grid (triton_poi_fused_copy_1_xnumel )](arg3_1 ,buf1 ,buf0 ,buf2 ,34 ,32 ,102 ,34 ,32 ,3468 ,10404 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        del buf1 
        buf3 =buf0 ;del buf0 

        triton_poi_fused_2_xnumel =12 *s0 +6 *s0 *s1 +6 *s0 *s2 +3 *s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_2 [grid (triton_poi_fused_2_xnumel )](buf2 ,buf3 ,34 ,32 ,10404 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        buf4 =buf2 ;del buf2 

        triton_poi_fused_3_xnumel =12 *s0 +6 *s0 *s1 +6 *s0 *s2 +3 *s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_3 [grid (triton_poi_fused_3_xnumel )](buf3 ,buf4 ,102 ,34 ,32 ,34 ,32 ,10404 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        12 +6 *s1 +6 *s2 +3 *s1 *s2 
        buf5 =reinterpret_tensor (buf3 ,(1 ,s0 ,12 +6 *s1 +6 *s2 +3 *s1 *s2 ),(12 *s0 +6 *s0 *s1 +6 *s0 *s2 +3 *s0 *s1 *s2 ,12 +6 *s1 +6 *s2 +3 *s1 *s2 ,1 ),0 );del buf3 

        triton_poi_fused_4_xnumel =12 *s0 +6 *s0 *s1 +6 *s0 *s2 +3 *s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_4 [grid (triton_poi_fused_4_xnumel )](buf4 ,buf5 ,3468 ,102 ,32 ,32 ,10404 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del buf4 
    return (buf5 ,)

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
