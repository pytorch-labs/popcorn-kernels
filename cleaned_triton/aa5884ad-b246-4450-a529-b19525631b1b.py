
from ctypes import c_void_p ,c_long ,c_int 
import torch 
import math 
import random 
import os 
import tempfile 
from math import inf ,nan 
from cmath import nanj 
from torch ._inductor .hooks import run_intermediate_hooks 
from torch ._inductor .utils import maybe_profile 
from torch ._inductor .codegen .memory_planning import _align as align 
from torch import device ,empty_strided 
from torch ._inductor .async_compile import AsyncCompile 
from torch ._inductor .select_algorithm import extern_kernels 
from torch ._inductor .codegen .multi_kernel import MultiKernelCall 
import triton 
import triton .language as tl 
from torch ._inductor .runtime .triton_heuristics import (
grid ,
split_scan_grid ,
grid_combo_kernels ,
start_graph ,
end_graph ,
cooperative_reduction_grid ,
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

extern "C"void kernel (int64_t *out_ptr0 ,
const int64_t ks0 ,
const int64_t ks1 ,
const int64_t ks2 )
{
{
{
{
auto tmp0 =c10 ::div_floor_integer (static_cast <int64_t >(ks0 *(c10 ::div_floor_integer (static_cast <int64_t >(ks1 ),static_cast <int64_t >(c10 ::div_floor_integer (static_cast <int64_t >(ks1 ),static_cast <int64_t >(2 L )))))*(c10 ::div_floor_integer (static_cast <int64_t >(ks2 ),static_cast <int64_t >(c10 ::div_floor_integer (static_cast <int64_t >(ks2 ),static_cast <int64_t >(2 L )))))),static_cast <int64_t >(4 L ));
auto tmp1 =c10 ::convert <int64_t >(tmp0 );
out_ptr0 [static_cast <int64_t >(0 L )]=tmp1 ;
}
}
}
}
''')

#include "/tmp/torchinductor_sahanp/3b/c3bi5gk6mslf6u4iaqafhxm64z6u65e3eain4xlary5blqnvv6xx.h"
extern "C"  void kernel(const int64_t* in_ptr0,
                       int64_t* out_ptr0,
                       int64_t* out_ptr1,
                       const int64_t ks0,
                       const int64_t ks1,
                       const int64_t ks2)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = static_cast<int32_t>(0);
                auto tmp2 = static_cast<int64_t>(1);
                auto tmp3 = c10::div_floor_integer(static_cast<int64_t>(ks0*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(2L)))))*(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(2L)))))), static_cast<int64_t>(4L));
                auto tmp4 = c10::convert<int64_t>(tmp3);
                auto tmp5 = randint64_cpu(tmp0, tmp1, tmp2, tmp4);
                out_ptr0[static_cast<int64_t>(0L)] = tmp5;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(ks0*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(2L)))))*(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(2L)))))), static_cast<int64_t>(4L))); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(1L)];
                    auto tmp1 = x0;
                    auto tmp2 = c10::convert<int32_t>(tmp1);
                    auto tmp3 = static_cast<int64_t>(0);
                    auto tmp4 = static_cast<int64_t>(10);
                    auto tmp5 = randint64_cpu(tmp0, tmp2, tmp3, tmp4);
                    out_ptr1[static_cast<int64_t>(x0)] = tmp5;
                }
            }
        }
    }
}
''')

import triton 
import triton .language as tl 
from triton .compiler .compiler import AttrsDescriptor 

from torch ._inductor .runtime import triton_helpers ,triton_heuristics 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
from torch ._inductor .runtime .hints import AutotuneHint ,ReductionHint ,TileHint ,DeviceProperties 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_log_sigmoid_forward_2 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%ks1 )
    x2 =xindex //ks2 
    x3 =xindex //ks0 
    tmp0 =tl .load (in_ptr0 +(x0 +2 *x1 *(ks5 //2 )+2 *(ks5 //2 )*(((x0 %2 ))//2 )+4 *(ks4 //2 )*(ks5 //2 )*((((2 *((x1 %2 ))+4 *x2 +((x0 %2 )))//4 )%ks3 ))),xmask ,eviction_policy ='evict_last')
    tmp1 =0.5 
    tmp2 =tmp0 *tmp1 
    tmp3 =0.7071067811865476 
    tmp4 =tmp0 *tmp3 
    tmp5 =libdevice .erf (tmp4 )
    tmp6 =1.0 
    tmp7 =tmp5 +tmp6 
    tmp8 =tmp2 *tmp7 
    tmp9 =0.0 
    tmp10 =triton_helpers .minimum (tmp9 ,tmp8 )
    tmp11 =tl_math .abs (tmp8 )
    tmp12 =-tmp11 
    tmp13 =tl_math .exp (tmp12 )
    tmp14 =libdevice .log1p (tmp13 )
    tmp15 =tmp10 -tmp14 
    tl .store (out_ptr0 +(x0 +x3 *(ks5 //2 )*(triton_helpers .div_floor_integer (ks3 *(triton_helpers .div_floor_integer (ks4 ,ks4 //2 ))*(triton_helpers .div_floor_integer (ks5 ,ks5 //2 )),2 *(triton_helpers .div_floor_integer (ks3 *(triton_helpers .div_floor_integer (ks4 ,ks4 //2 ))*(triton_helpers .div_floor_integer (ks5 ,ks5 //2 )),4 ))))),tmp15 ,xmask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    buf3 =empty_strided_cpu ((1 ,),(1 ,),torch .int64 )
    cpp_fused_full_0 (buf3 ,s0 ,s1 ,s2 )
    buf1 =empty_strided_cpu ((2 ,),(1 ,),torch .int64 )

    aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[2 ],out =buf1 )
    buf4 =empty_strided_cpu ((1 ,),(1 ,),torch .int64 )
    buf2 =empty_strided_cpu ((1 ,(s0 *(s1 //(s1 //2 ))*(s2 //(s2 //2 )))//4 ),((s0 *(s1 //(s1 //2 ))*(s2 //(s2 //2 )))//4 ,1 ),torch .int64 )
    cpp_fused_randint_1 (buf1 ,buf4 ,buf2 ,s0 ,s1 ,s2 )
    del buf1 
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        ps0 =2 *(s2 //2 )
        ps1 =2 *(s1 //2 )
        ps2 =4 *(s1 //2 )*(s2 //2 )
        buf0 =empty_strided_cuda ((1 ,(s0 *(s1 //(s1 //2 ))*(s2 //(s2 //2 )))//4 ,2 *(s1 //2 ),2 *(s2 //2 )),(2 *(s1 //2 )*(s2 //2 )*((s0 *(s1 //(s1 //2 ))*(s2 //(s2 //2 )))//4 )*((s0 *(s1 //(s1 //2 ))*(s2 //(s2 //2 )))//(2 *((s0 *(s1 //(s1 //2 ))*(s2 //(s2 //2 )))//4 ))),2 *(s1 //2 )*(s2 //2 )*((s0 *(s1 //(s1 //2 ))*(s2 //(s2 //2 )))//(2 *((s0 *(s1 //(s1 //2 ))*(s2 //(s2 //2 )))//4 ))),(s2 //2 )*((s0 *(s1 //(s1 //2 ))*(s2 //(s2 //2 )))//(2 *((s0 *(s1 //(s1 //2 ))*(s2 //(s2 //2 )))//4 ))),1 ),torch .float32 )

        triton_poi_fused_log_sigmoid_forward_2_xnumel =4 *(s1 //2 )*(s2 //2 )*((s0 *(s1 //(s1 //2 ))*(s2 //(s2 //2 )))//4 )
        stream0 =get_raw_stream (0 )
        triton_poi_fused_log_sigmoid_forward_2 [grid (triton_poi_fused_log_sigmoid_forward_2_xnumel )](arg3_1 ,buf0 ,64 ,64 ,4096 ,3 ,64 ,64 ,12288 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
    return (buf0 ,buf2 ,buf3 ,buf4 ,)

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
