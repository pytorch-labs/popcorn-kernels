
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
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0 (in_out_ptr0 ,in_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x1 =((xindex //2 )%20 )
    x2 =xindex //40 
    x4 =xindex 
    tmp0 =x1 
    tmp1 =tmp0 .to (tl .float32 )
    tmp2 =0.47368421052631576 
    tmp3 =tmp1 *tmp2 
    tmp4 =0.0 
    tmp5 =triton_helpers .maximum (tmp3 ,tmp4 )
    tmp6 =tmp5 .to (tl .int32 )
    tmp7 =tl .full ([1 ],1 ,tl .int64 )
    tmp8 =tmp6 +tmp7 
    tmp9 =tl .full ([1 ],9 ,tl .int64 )
    tmp10 =triton_helpers .minimum (tmp8 ,tmp9 )
    tmp11 =tl .load (in_ptr0 +(tmp10 +10 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp12 =tmp11 -tmp11 
    tmp13 =tmp12 *tmp4 
    tmp14 =tmp11 +tmp13 
    tmp15 =tl .load (in_ptr0 +(tmp6 +10 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp16 =tmp15 -tmp15 
    tmp17 =tmp16 *tmp4 
    tmp18 =tmp15 +tmp17 
    tmp19 =tmp14 -tmp18 
    tmp20 =tmp6 .to (tl .float32 )
    tmp21 =tmp5 -tmp20 
    tmp22 =triton_helpers .maximum (tmp21 ,tmp4 )
    tmp23 =1.0 
    tmp24 =triton_helpers .minimum (tmp22 ,tmp23 )
    tmp25 =tmp19 *tmp24 
    tmp26 =tmp18 +tmp25 
    tl .store (in_out_ptr0 +(x4 ),tmp26 ,xmask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )

        buf0 =torch .ops .aten .adaptive_max_pool2d .default (reinterpret_tensor (arg3_1 ,(1 ,s0 ,1 ,s1 *s2 ),(s0 *s1 *s2 ,s1 *s2 ,s1 *s2 ,1 ),0 ),[1 ,10 ])
        del arg3_1 
        buf1 =buf0 [0 ]
        del buf0 
        buf3 =empty_strided_cuda ((1 ,s0 ,20 ,2 ),(40 *s0 ,40 ,2 ,1 ),torch .float32 )
        buf4 =buf3 ;del buf3 

        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0_xnumel =40 *s0 
        get_raw_stream (0 )
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0 [grid (triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0_xnumel )](buf4 ,buf1 ,120 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf1 
    return (reinterpret_tensor (buf4 ,(1 ,40 ,s0 ),(40 *s0 ,s0 ,1 ),0 ),s0 ,)

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
