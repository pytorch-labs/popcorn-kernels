
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
def triton_poi_fused__to_copy_add_bernoulli_mul_0 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    tl .full ([XBLOCK ],True ,tl .int1 )
    x0 =xindex 
    tmp3 =tl .load (in_ptr1 +(x0 ),None )
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =x0 
    tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tmp4 =0.5 
    tmp5 =tmp2 <tmp4 
    tmp6 =tmp5 .to (tl .float32 )
    tmp7 =0.8864048946659319 
    tmp8 =tmp6 *tmp7 
    tmp9 =tmp3 *tmp8 
    tmp10 =-1.0 
    tmp11 =tmp6 +tmp10 
    tmp12 =1.558387861036063 
    tmp13 =tmp11 *tmp12 
    tmp14 =0.7791939305180315 
    tmp15 =tmp13 +tmp14 
    tmp16 =tmp9 +tmp15 
    tl .store (in_out_ptr0 +(x0 ),tmp16 ,None )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,64 ,64 ),(4096 *s0 ,4096 ,64 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[1 ],out =buf0 )
        buf1 =empty_strided_cuda ((1 ,s0 ,64 ,64 ),(4096 *s0 ,4096 ,64 ,1 ),torch .float32 )
        buf2 =buf1 ;del buf1 

        triton_poi_fused__to_copy_add_bernoulli_mul_0_xnumel =4096 *s0 
        get_raw_stream (0 )
        triton_poi_fused__to_copy_add_bernoulli_mul_0 [grid (triton_poi_fused__to_copy_add_bernoulli_mul_0_xnumel )](buf2 ,buf0 ,arg3_1 ,0 ,12288 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        del buf0 

        buf3 =torch .ops .aten .adaptive_max_pool2d .default (buf2 ,[5 ,5 ])
        del buf2 
        buf4 =buf3 [0 ]
        del buf3 
    return (reinterpret_tensor (buf4 ,(1 ,s0 ,25 ),(25 *s0 ,25 ,1 ),0 ),)

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
