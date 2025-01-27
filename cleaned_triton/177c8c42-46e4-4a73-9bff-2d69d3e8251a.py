
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
def triton_poi_fused_avg_pool3d_pow_0 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,ks6 ,ks7 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%ks1 )
    x2 =((xindex //ks2 )%ks3 )
    x3 =xindex //ks4 
    x4 =xindex 
    tmp0 =tl .load (in_ptr0 +(2 *x0 +2 *ks7 *x1 +2 *ks6 *ks7 *x2 +ks5 *ks6 *ks7 *x3 ),xmask ,eviction_policy ='evict_last')
    tmp2 =tl .load (in_ptr0 +(1 +2 *x0 +2 *ks7 *x1 +2 *ks6 *ks7 *x2 +ks5 *ks6 *ks7 *x3 ),xmask ,eviction_policy ='evict_last')
    tmp5 =tl .load (in_ptr0 +(ks7 +2 *x0 +2 *ks7 *x1 +2 *ks6 *ks7 *x2 +ks5 *ks6 *ks7 *x3 ),xmask ,eviction_policy ='evict_last')
    tmp8 =tl .load (in_ptr0 +(1 +ks7 +2 *x0 +2 *ks7 *x1 +2 *ks6 *ks7 *x2 +ks5 *ks6 *ks7 *x3 ),xmask ,eviction_policy ='evict_last')
    tmp11 =tl .load (in_ptr0 +(2 *x0 +ks6 *ks7 +2 *ks7 *x1 +2 *ks6 *ks7 *x2 +ks5 *ks6 *ks7 *x3 ),xmask ,eviction_policy ='evict_last')
    tmp14 =tl .load (in_ptr0 +(1 +2 *x0 +ks6 *ks7 +2 *ks7 *x1 +2 *ks6 *ks7 *x2 +ks5 *ks6 *ks7 *x3 ),xmask ,eviction_policy ='evict_last')
    tmp17 =tl .load (in_ptr0 +(ks7 +2 *x0 +ks6 *ks7 +2 *ks7 *x1 +2 *ks6 *ks7 *x2 +ks5 *ks6 *ks7 *x3 ),xmask ,eviction_policy ='evict_last')
    tmp20 =tl .load (in_ptr0 +(1 +ks7 +2 *x0 +ks6 *ks7 +2 *ks7 *x1 +2 *ks6 *ks7 *x2 +ks5 *ks6 *ks7 *x3 ),xmask ,eviction_policy ='evict_last')
    tmp1 =tmp0 *tmp0 
    tmp3 =tmp2 *tmp2 
    tmp4 =tmp3 +tmp1 
    tmp6 =tmp5 *tmp5 
    tmp7 =tmp6 +tmp4 
    tmp9 =tmp8 *tmp8 
    tmp10 =tmp9 +tmp7 
    tmp12 =tmp11 *tmp11 
    tmp13 =tmp12 +tmp10 
    tmp15 =tmp14 *tmp14 
    tmp16 =tmp15 +tmp13 
    tmp18 =tmp17 *tmp17 
    tmp19 =tmp18 +tmp16 
    tmp21 =tmp20 *tmp20 
    tmp22 =tmp21 +tmp19 
    tmp23 =0.125 
    tmp24 =tmp22 *tmp23 
    tl .store (out_ptr0 +(x4 ),tmp24 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_abs_avg_pool3d_mul_pow_relu_sign_1 (in_out_ptr0 ,in_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,ks6 ,ks7 ,ks8 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%ks1 )
    x2 =((xindex //ks2 )%ks3 )
    x3 =xindex //ks4 
    x4 =xindex 
    tmp0 =tl .load (in_ptr0 +(2 *x0 +2 *ks5 *x1 +2 *ks5 *ks6 *x2 +ks5 *ks6 *ks7 *x3 ),xmask ,eviction_policy ='evict_last')
    tmp15 =tl .load (in_ptr0 +(1 +2 *x0 +2 *ks5 *x1 +2 *ks5 *ks6 *x2 +ks5 *ks6 *ks7 *x3 ),xmask ,eviction_policy ='evict_last')
    tmp29 =tl .load (in_ptr0 +(ks5 +2 *x0 +2 *ks5 *x1 +2 *ks5 *ks6 *x2 +ks5 *ks6 *ks7 *x3 ),xmask ,eviction_policy ='evict_last')
    tmp43 =tl .load (in_ptr0 +(1 +ks5 +2 *x0 +2 *ks5 *x1 +2 *ks5 *ks6 *x2 +ks5 *ks6 *ks7 *x3 ),xmask ,eviction_policy ='evict_last')
    tmp57 =tl .load (in_ptr0 +(ks8 +2 *x0 +2 *ks5 *x1 +2 *ks5 *ks6 *x2 +ks5 *ks6 *ks7 *x3 ),xmask ,eviction_policy ='evict_last')
    tmp71 =tl .load (in_ptr0 +(1 +ks8 +2 *x0 +2 *ks5 *x1 +2 *ks5 *ks6 *x2 +ks5 *ks6 *ks7 *x3 ),xmask ,eviction_policy ='evict_last')
    tmp85 =tl .load (in_ptr0 +(ks5 +ks8 +2 *x0 +2 *ks5 *x1 +2 *ks5 *ks6 *x2 +ks5 *ks6 *ks7 *x3 ),xmask ,eviction_policy ='evict_last')
    tmp99 =tl .load (in_ptr0 +(1 +ks5 +ks8 +2 *x0 +2 *ks5 *x1 +2 *ks5 *ks6 *x2 +ks5 *ks6 *ks7 *x3 ),xmask ,eviction_policy ='evict_last')
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
    tmp11 =8.0 
    tmp12 =tmp10 *tmp11 
    tmp13 =libdevice .sqrt (tmp12 )
    tmp14 =tmp13 *tmp13 
    tmp16 =tmp1 <tmp15 
    tmp17 =tmp16 .to (tl .int8 )
    tmp18 =tmp15 <tmp1 
    tmp19 =tmp18 .to (tl .int8 )
    tmp20 =tmp17 -tmp19 
    tmp21 =tmp20 .to (tmp15 .dtype )
    tmp22 =tl_math .abs (tmp15 )
    tmp23 =triton_helpers .maximum (tmp1 ,tmp22 )
    tmp24 =tmp21 *tmp23 
    tmp25 =tmp24 *tmp11 
    tmp26 =libdevice .sqrt (tmp25 )
    tmp27 =tmp26 *tmp26 
    tmp28 =tmp27 +tmp14 
    tmp30 =tmp1 <tmp29 
    tmp31 =tmp30 .to (tl .int8 )
    tmp32 =tmp29 <tmp1 
    tmp33 =tmp32 .to (tl .int8 )
    tmp34 =tmp31 -tmp33 
    tmp35 =tmp34 .to (tmp29 .dtype )
    tmp36 =tl_math .abs (tmp29 )
    tmp37 =triton_helpers .maximum (tmp1 ,tmp36 )
    tmp38 =tmp35 *tmp37 
    tmp39 =tmp38 *tmp11 
    tmp40 =libdevice .sqrt (tmp39 )
    tmp41 =tmp40 *tmp40 
    tmp42 =tmp41 +tmp28 
    tmp44 =tmp1 <tmp43 
    tmp45 =tmp44 .to (tl .int8 )
    tmp46 =tmp43 <tmp1 
    tmp47 =tmp46 .to (tl .int8 )
    tmp48 =tmp45 -tmp47 
    tmp49 =tmp48 .to (tmp43 .dtype )
    tmp50 =tl_math .abs (tmp43 )
    tmp51 =triton_helpers .maximum (tmp1 ,tmp50 )
    tmp52 =tmp49 *tmp51 
    tmp53 =tmp52 *tmp11 
    tmp54 =libdevice .sqrt (tmp53 )
    tmp55 =tmp54 *tmp54 
    tmp56 =tmp55 +tmp42 
    tmp58 =tmp1 <tmp57 
    tmp59 =tmp58 .to (tl .int8 )
    tmp60 =tmp57 <tmp1 
    tmp61 =tmp60 .to (tl .int8 )
    tmp62 =tmp59 -tmp61 
    tmp63 =tmp62 .to (tmp57 .dtype )
    tmp64 =tl_math .abs (tmp57 )
    tmp65 =triton_helpers .maximum (tmp1 ,tmp64 )
    tmp66 =tmp63 *tmp65 
    tmp67 =tmp66 *tmp11 
    tmp68 =libdevice .sqrt (tmp67 )
    tmp69 =tmp68 *tmp68 
    tmp70 =tmp69 +tmp56 
    tmp72 =tmp1 <tmp71 
    tmp73 =tmp72 .to (tl .int8 )
    tmp74 =tmp71 <tmp1 
    tmp75 =tmp74 .to (tl .int8 )
    tmp76 =tmp73 -tmp75 
    tmp77 =tmp76 .to (tmp71 .dtype )
    tmp78 =tl_math .abs (tmp71 )
    tmp79 =triton_helpers .maximum (tmp1 ,tmp78 )
    tmp80 =tmp77 *tmp79 
    tmp81 =tmp80 *tmp11 
    tmp82 =libdevice .sqrt (tmp81 )
    tmp83 =tmp82 *tmp82 
    tmp84 =tmp83 +tmp70 
    tmp86 =tmp1 <tmp85 
    tmp87 =tmp86 .to (tl .int8 )
    tmp88 =tmp85 <tmp1 
    tmp89 =tmp88 .to (tl .int8 )
    tmp90 =tmp87 -tmp89 
    tmp91 =tmp90 .to (tmp85 .dtype )
    tmp92 =tl_math .abs (tmp85 )
    tmp93 =triton_helpers .maximum (tmp1 ,tmp92 )
    tmp94 =tmp91 *tmp93 
    tmp95 =tmp94 *tmp11 
    tmp96 =libdevice .sqrt (tmp95 )
    tmp97 =tmp96 *tmp96 
    tmp98 =tmp97 +tmp84 
    tmp100 =tmp1 <tmp99 
    tmp101 =tmp100 .to (tl .int8 )
    tmp102 =tmp99 <tmp1 
    tmp103 =tmp102 .to (tl .int8 )
    tmp104 =tmp101 -tmp103 
    tmp105 =tmp104 .to (tmp99 .dtype )
    tmp106 =tl_math .abs (tmp99 )
    tmp107 =triton_helpers .maximum (tmp1 ,tmp106 )
    tmp108 =tmp105 *tmp107 
    tmp109 =tmp108 *tmp11 
    tmp110 =libdevice .sqrt (tmp109 )
    tmp111 =tmp110 *tmp110 
    tmp112 =tmp111 +tmp98 
    tmp113 =0.125 
    tmp114 =tmp112 *tmp113 
    tmp115 =tmp1 <tmp114 
    tmp116 =tmp115 .to (tl .int8 )
    tmp117 =tmp114 <tmp1 
    tmp118 =tmp117 .to (tl .int8 )
    tmp119 =tmp116 -tmp118 
    tmp120 =tmp119 .to (tmp114 .dtype )
    tmp121 =tl_math .abs (tmp114 )
    tmp122 =triton_helpers .maximum (tmp1 ,tmp121 )
    tmp123 =tmp120 *tmp122 
    tmp124 =tmp123 *tmp11 
    tmp125 =libdevice .sqrt (tmp124 )
    tl .store (in_out_ptr0 +(x4 ),tmp125 ,xmask )

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
        s3 //2 
        s2 //2 
        (s2 //2 )*(s3 //2 )
        s1 //2 
        (s1 //2 )*(s2 //2 )*(s3 //2 )
        buf0 =empty_strided_cuda ((1 ,s0 ,s1 //2 ,s2 //2 ,s3 //2 ),(s0 *(s1 //2 )*(s2 //2 )*(s3 //2 ),(s1 //2 )*(s2 //2 )*(s3 //2 ),(s2 //2 )*(s3 //2 ),s3 //2 ,1 ),torch .float32 )

        triton_poi_fused_avg_pool3d_pow_0_xnumel =s0 *(s1 //2 )*(s2 //2 )*(s3 //2 )
        get_raw_stream (0 )
        triton_poi_fused_avg_pool3d_pow_0 [grid (triton_poi_fused_avg_pool3d_pow_0_xnumel )](arg4_1 ,buf0 ,32 ,32 ,1024 ,16 ,16384 ,32 ,64 ,64 ,49152 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del arg4_1 
        s3 //4 
        s2 //4 
        (s2 //4 )*(s3 //4 )
        s1 //4 
        (s1 //4 )*(s2 //4 )*(s3 //4 )
        buf1 =empty_strided_cuda ((1 ,s0 ,s1 //4 ,s2 //4 ,s3 //4 ),(s0 *(s1 //4 )*(s2 //4 )*(s3 //4 ),(s1 //4 )*(s2 //4 )*(s3 //4 ),(s2 //4 )*(s3 //4 ),s3 //4 ,1 ),torch .float32 )
        buf2 =buf1 ;del buf1 

        triton_poi_fused_abs_avg_pool3d_mul_pow_relu_sign_1_xnumel =s0 *(s1 //4 )*(s2 //4 )*(s3 //4 )
        get_raw_stream (0 )
        triton_poi_fused_abs_avg_pool3d_mul_pow_relu_sign_1 [grid (triton_poi_fused_abs_avg_pool3d_mul_pow_relu_sign_1_xnumel )](buf2 ,buf0 ,16 ,16 ,256 ,8 ,2048 ,32 ,32 ,16 ,1024 ,6144 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf0 
    return (reinterpret_tensor (buf2 ,(1 ,s0 ,(s1 //4 )*(s2 //4 )*(s3 //4 )),(s0 *(s1 //4 )*(s2 //4 )*(s3 //4 ),(s1 //4 )*(s2 //4 )*(s3 //4 ),1 ),0 ),s0 ,s1 //4 ,s2 //4 ,s3 //4 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =3 
    arg1_1 =32 
    arg2_1 =64 
    arg3_1 =64 
    arg4_1 =rand_strided ((1 ,3 ,32 ,64 ,64 ),(393216 ,131072 ,4096 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ,arg4_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
