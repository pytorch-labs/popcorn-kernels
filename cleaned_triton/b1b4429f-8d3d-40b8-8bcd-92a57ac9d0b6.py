
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
def triton_poi_fused__adaptive_avg_pool2d_0 (in_ptr0 ,out_ptr0 ,ks0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %128 )
    x1 =xindex //128 
    x2 =xindex 
    tmp0 =tl .full ([1 ],0 ,tl .int64 )
    tmp1 =tl .full ([1 ],1 ,tl .int64 )
    tmp2 =tmp0 <tmp1 
    tmp3 =(25 *x0 )//32 
    tmp4 =(227 +100 *x0 )//128 
    tmp5 =tmp3 <tmp4 
    tmp6 =tmp2 &tmp5 
    tmp7 =tl .load (in_ptr0 +(x1 +ks0 *((25 *x0 )//32 )),tmp6 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp8 =1 +((25 *x0 )//32 )
    tmp9 =tmp8 <tmp4 
    tmp10 =tmp2 &tmp9 
    tmp11 =tl .load (in_ptr0 +(ks0 +x1 +ks0 *((25 *x0 )//32 )),tmp10 &xmask ,eviction_policy ='evict_last',other =0.0 )
    tmp12 =tmp11 +tmp7 
    tmp13 =1.0 
    tmp14 =tl .full (tmp13 .shape ,0.0 ,tmp13 .dtype )
    tmp15 =tl .where (tmp6 ,tmp13 ,tmp14 )
    tmp16 =1.0 
    tmp17 =tl .full (tmp16 .shape ,0.0 ,tmp16 .dtype )
    tmp18 =tl .where (tmp10 ,tmp16 ,tmp17 )
    tmp19 =tmp18 +tmp15 
    tmp20 =tmp12 /tmp19 
    tl .store (out_ptr0 +(x2 ),tmp20 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_permute_1 (in_ptr0 ,out_ptr0 ,ks0 ,ynumel ,xnumel ,YBLOCK :tl .constexpr ,XBLOCK :tl .constexpr ):
    ynumel =128 
    yoffset =tl .program_id (1 )*YBLOCK 
    yindex =yoffset +tl .arange (0 ,YBLOCK )[None ,:]
    ymask =yindex <ynumel 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    x1 =xindex 
    y0 =yindex 
    tmp0 =tl .load (in_ptr0 +(y0 +128 *x1 ),xmask &ymask ,eviction_policy ='evict_last')
    tl .store (out_ptr0 +(x1 +ks0 *y0 ),tmp0 ,xmask &ymask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 =args 
    args .clear ()
    s1 =arg1_1 
    assert_size_stride (arg2_1 ,(1 ,100 ,s1 ),(100 *s1 ,s1 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,s1 ,1 ,128 ),(128 *s1 ,128 ,128 ,1 ),torch .float32 )

        triton_poi_fused__adaptive_avg_pool2d_0_xnumel =128 *s1 
        get_raw_stream (0 )
        triton_poi_fused__adaptive_avg_pool2d_0 [grid (triton_poi_fused__adaptive_avg_pool2d_0_xnumel )](arg2_1 ,buf0 ,256 ,32768 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del arg2_1 
        buf1 =empty_strided_cuda ((1 ,128 ,s1 ),(128 *s1 ,s1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_permute_1 [grid (128 ,s1 )](buf0 ,buf1 ,256 ,128 ,256 ,XBLOCK =256 ,YBLOCK =1 ,num_warps =4 ,num_stages =1 )
        del buf0 
    return (buf1 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =100 
    arg1_1 =256 
    arg2_1 =rand_strided ((1 ,100 ,256 ),(25600 ,256 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
