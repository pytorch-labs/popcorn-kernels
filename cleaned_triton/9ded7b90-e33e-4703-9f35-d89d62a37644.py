
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
def triton_poi_fused_avg_pool2d_pow_0 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x1 =((xindex //ks0 )%ks1 )
    x0 =(xindex %ks0 )
    x2 =xindex //ks5 
    x4 =xindex 
    tmp0 =(-1 )+2 *x1 
    tmp1 =tl .full ([1 ],0 ,tl .int64 )
    tmp2 =tmp0 >=tmp1 
    tmp3 =ks2 
    tmp4 =tmp0 <tmp3 
    tmp5 =(-1 )+((((2 *x0 )//(2 +ks4 ))%(2 +ks3 )))
    tmp6 =tmp5 >=tmp1 
    tmp7 =ks3 
    tmp8 =tmp5 <tmp7 
    tmp9 =(-1 )+(((2 *x0 )%(2 +ks4 )))
    tmp10 =tmp9 >=tmp1 
    tmp11 =ks4 
    tmp12 =tmp9 <tmp11 
    tmp13 =tmp2 &tmp4 
    tmp14 =tmp13 &tmp6 
    tmp15 =tmp14 &tmp8 
    tmp16 =tmp15 &tmp10 
    tmp17 =tmp16 &tmp12 
    tmp18 =tl .load (in_ptr0 +((-1 )+((-1 )*ks4 )+ks4 *((((2 *x0 )//(2 +ks4 ))%(2 +ks3 )))+((-1 )*ks3 *ks4 )+2 *ks3 *ks4 *x1 +ks2 *ks3 *ks4 *x2 +(((2 *x0 )%(2 +ks4 )))),tmp17 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp19 =tmp18 *tmp18 
    tmp20 =(-1 )+((((1 +2 *x0 )//(2 +ks4 ))%(2 +ks3 )))
    tmp21 =tmp20 >=tmp1 
    tmp22 =tmp20 <tmp7 
    tmp23 =(-1 )+(((1 +2 *x0 )%(2 +ks4 )))
    tmp24 =tmp23 >=tmp1 
    tmp25 =tmp23 <tmp11 
    tmp26 =tmp13 &tmp21 
    tmp27 =tmp26 &tmp22 
    tmp28 =tmp27 &tmp24 
    tmp29 =tmp28 &tmp25 
    tmp30 =tl .load (in_ptr0 +((-1 )+((-1 )*ks4 )+ks4 *((((1 +2 *x0 )//(2 +ks4 ))%(2 +ks3 )))+((-1 )*ks3 *ks4 )+2 *ks3 *ks4 *x1 +ks2 *ks3 *ks4 *x2 +(((1 +2 *x0 )%(2 +ks4 )))),tmp29 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp31 =tmp30 *tmp30 
    tmp32 =tmp31 +tmp19 
    tmp33 =2 *x1 
    tmp34 =tmp33 >=tmp1 
    tmp35 =tmp33 <tmp3 
    tmp36 =tmp34 &tmp35 
    tmp37 =tmp36 &tmp6 
    tmp38 =tmp37 &tmp8 
    tmp39 =tmp38 &tmp10 
    tmp40 =tmp39 &tmp12 
    tmp41 =tl .load (in_ptr0 +((-1 )+((-1 )*ks4 )+ks4 *((((2 *x0 )//(2 +ks4 ))%(2 +ks3 )))+2 *ks3 *ks4 *x1 +ks2 *ks3 *ks4 *x2 +(((2 *x0 )%(2 +ks4 )))),tmp40 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp42 =tmp41 *tmp41 
    tmp43 =tmp42 +tmp32 
    tmp44 =tmp36 &tmp21 
    tmp45 =tmp44 &tmp22 
    tmp46 =tmp45 &tmp24 
    tmp47 =tmp46 &tmp25 
    tmp48 =tl .load (in_ptr0 +((-1 )+((-1 )*ks4 )+ks4 *((((1 +2 *x0 )//(2 +ks4 ))%(2 +ks3 )))+2 *ks3 *ks4 *x1 +ks2 *ks3 *ks4 *x2 +(((1 +2 *x0 )%(2 +ks4 )))),tmp47 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp49 =tmp48 *tmp48 
    tmp50 =tmp49 +tmp43 
    tmp51 =0.25 
    tmp52 =tmp50 *tmp51 
    tl .store (out_ptr0 +(x4 ),tmp52 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_abs_mul_pow_relu_sign_1 (in_out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_out_ptr0 +(x0 ),xmask )
    tmp1 =tl .full ([1 ],0 ,tl .int32 )
    tmp2 =tmp1 <tmp0 
    tmp3 =tmp2 .to (tl .int8 )
    tmp4 =tmp0 <tmp1 
    tmp5 =tmp4 .to (tl .int8 )
    tmp6 =tmp3 -tmp5 
    tmp7 =tmp6 .to (tmp0 .dtype )
    tmp8 =tl_math .abs (tmp0 )
    tmp9 =triton_helpers .maximum (tmp1 ,tmp8 )
    tmp10 =tmp7 *tmp9 
    tmp11 =4.0 
    tmp12 =tmp10 *tmp11 
    tmp13 =libdevice .sqrt (tmp12 )
    tl .store (in_out_ptr0 +(x0 ),tmp13 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_abs_mul_pow_relu_sign_view_2 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =xindex //ks0 
    x2 =xindex 
    tmp0 =tl .load (in_ptr0 +(2 *x1 +2 *(x0 //ks1 )+ks3 *x1 +ks3 *(x0 //ks1 )+ks4 *x1 +ks4 *(x0 //ks1 )+x1 *((ks3 *ks4 )//2 )+(x0 //ks1 )*((ks3 *ks4 )//2 )+2 *x1 *(ks2 //2 )+ks3 *x1 *(ks2 //2 )+ks4 *x1 *(ks2 //2 )+x1 *(ks2 //2 )*((ks3 *ks4 )//2 )+((x0 %ks1 ))),xmask ,eviction_policy ='evict_last')
    tl .store (out_ptr0 +(x2 ),tmp0 ,xmask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ,arg4_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    s3 =arg3_1 
    assert_size_stride (arg4_1 ,(1 ,s0 ,s1 ,s2 ,s3 ),(s0 *s1 *s2 *s3 ,s1 *s2 *s3 ,s2 *s3 ,s3 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        2 +s2 +s3 +((s2 *s3 )//2 )
        1 +(s1 //2 )
        2 +s2 +s3 +2 *(s1 //2 )+s2 *(s1 //2 )+s3 *(s1 //2 )+(s1 //2 )*((s2 *s3 )//2 )+((s2 *s3 )//2 )
        buf0 =empty_strided_cuda ((1 ,s0 ,1 +(s1 //2 ),2 +s2 +s3 +((s2 *s3 )//2 )),(2 *s0 +s0 *s2 +s0 *s3 +s0 *((s2 *s3 )//2 )+2 *s0 *(s1 //2 )+s0 *s2 *(s1 //2 )+s0 *s3 *(s1 //2 )+s0 *(s1 //2 )*((s2 *s3 )//2 ),2 +s2 +s3 +2 *(s1 //2 )+s2 *(s1 //2 )+s3 *(s1 //2 )+(s1 //2 )*((s2 *s3 )//2 )+((s2 *s3 )//2 ),2 +s2 +s3 +((s2 *s3 )//2 ),1 ),torch .float32 )

        triton_poi_fused_avg_pool2d_pow_0_xnumel =2 *s0 +s0 *s2 +s0 *s3 +s0 *((s2 *s3 )//2 )+2 *s0 *(s1 //2 )+s0 *s2 *(s1 //2 )+s0 *s3 *(s1 //2 )+s0 *(s1 //2 )*((s2 *s3 )//2 )
        get_raw_stream (0 )
        triton_poi_fused_avg_pool2d_pow_0 [grid (triton_poi_fused_avg_pool2d_pow_0_xnumel )](arg4_1 ,buf0 ,578 ,17 ,32 ,32 ,32 ,9826 ,29478 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del arg4_1 
        buf1 =buf0 ;del buf0 

        triton_poi_fused_abs_mul_pow_relu_sign_1_xnumel =2 *s0 +s0 *s2 +s0 *s3 +s0 *((s2 *s3 )//2 )+2 *s0 *(s1 //2 )+s0 *s2 *(s1 //2 )+s0 *s3 *(s1 //2 )+s0 *(s1 //2 )*((s2 *s3 )//2 )
        get_raw_stream (0 )
        triton_poi_fused_abs_mul_pow_relu_sign_1 [grid (triton_poi_fused_abs_mul_pow_relu_sign_1_xnumel )](buf1 ,29478 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        2 +s2 +s3 +2 *(s1 //2 )+s2 *(s1 //2 )+s3 *(s1 //2 )+(s1 //2 )*((s2 *s3 )//2 )+((s2 *s3 )//2 )
        buf2 =empty_strided_cuda ((1 ,s0 ,2 +s2 +s3 +2 *(s1 //2 )+s2 *(s1 //2 )+s3 *(s1 //2 )+(s1 //2 )*((s2 *s3 )//2 )+((s2 *s3 )//2 )),(2 *s0 +s0 *s2 +s0 *s3 +s0 *((s2 *s3 )//2 )+2 *s0 *(s1 //2 )+s0 *s2 *(s1 //2 )+s0 *s3 *(s1 //2 )+s0 *(s1 //2 )*((s2 *s3 )//2 ),2 +s2 +s3 +2 *(s1 //2 )+s2 *(s1 //2 )+s3 *(s1 //2 )+(s1 //2 )*((s2 *s3 )//2 )+((s2 *s3 )//2 ),1 ),torch .float32 )

        triton_poi_fused_abs_mul_pow_relu_sign_view_2_xnumel =2 *s0 +s0 *s2 +s0 *s3 +s0 *((s2 *s3 )//2 )+2 *s0 *(s1 //2 )+s0 *s2 *(s1 //2 )+s0 *s3 *(s1 //2 )+s0 *(s1 //2 )*((s2 *s3 )//2 )
        get_raw_stream (0 )
        triton_poi_fused_abs_mul_pow_relu_sign_view_2 [grid (triton_poi_fused_abs_mul_pow_relu_sign_view_2_xnumel )](buf1 ,buf2 ,9826 ,578 ,32 ,32 ,32 ,29478 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del buf1 
    return (buf2 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =3 
    arg1_1 =32 
    arg2_1 =32 
    arg3_1 =32 
    arg4_1 =rand_strided ((1 ,3 ,32 ,32 ,32 ),(98304 ,32768 ,1024 ,32 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ,arg4_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
