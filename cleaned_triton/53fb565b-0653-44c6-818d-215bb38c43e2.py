
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
def triton_per_fused__native_batch_norm_legit_0 (in_ptr0 ,out_ptr2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    xnumel =64 
    R0_BLOCK :tl .constexpr =128 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_1 =r0_index 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(r0_1 +128 *x0 ),xmask ,other =0.0 )
    tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
    tl .where (xmask ,tmp1 ,0 )
    tmp4 =tl .broadcast_to (tmp1 ,[XBLOCK ,R0_BLOCK ])
    tmp6 =tl .where (xmask ,tmp4 ,0 )
    tmp7 =tl .sum (tmp6 ,1 )[:,None ]
    tmp8 =tl .full ([XBLOCK ,1 ],128 ,tl .int32 )
    tmp9 =tmp8 .to (tl .float32 )
    tmp10 =tmp7 /tmp9 
    tmp11 =tmp1 -tmp10 
    tmp12 =tmp11 *tmp11 
    tmp13 =tl .broadcast_to (tmp12 ,[XBLOCK ,R0_BLOCK ])
    tmp15 =tl .where (xmask ,tmp13 ,0 )
    tmp16 =tl .sum (tmp15 ,1 )[:,None ]
    tmp17 =tmp0 -tmp10 
    tmp18 =128.0 
    tmp19 =tmp16 /tmp18 
    tmp20 =1e-05 
    tmp21 =tmp19 +tmp20 
    tmp22 =libdevice .rsqrt (tmp21 )
    tmp23 =tmp17 *tmp22 
    tl .store (out_ptr2 +(r0_1 +128 *x0 ),tmp23 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_1 (in_out_ptr0 ,in_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    tl .full ([XBLOCK ],True ,tl .int1 )
    x1 =((xindex //2 )%128 )
    x2 =xindex //256 
    x4 =xindex 
    tmp0 =x1 
    tmp1 =tmp0 .to (tl .float32 )
    tmp2 =0.49606299212598426 
    tmp3 =tmp1 *tmp2 
    tmp4 =0.0 
    tmp5 =triton_helpers .maximum (tmp3 ,tmp4 )
    tmp6 =tmp5 .to (tl .int32 )
    tmp7 =tl .full ([1 ],1 ,tl .int64 )
    tmp8 =tmp6 +tmp7 
    tmp9 =tl .full ([1 ],63 ,tl .int64 )
    tmp10 =triton_helpers .minimum (tmp8 ,tmp9 )
    tmp11 =tl .load (in_ptr0 +(2 *tmp10 +128 *x2 ),None ,eviction_policy ='evict_last')
    tmp12 =tl .load (in_ptr0 +(1 +2 *tmp10 +128 *x2 ),None ,eviction_policy ='evict_last')
    tmp13 =triton_helpers .maximum (tmp12 ,tmp11 )
    tmp14 =tmp13 -tmp13 
    tmp15 =tmp14 *tmp4 
    tmp16 =tmp13 +tmp15 
    tmp17 =tl .load (in_ptr0 +(2 *tmp6 +128 *x2 ),None ,eviction_policy ='evict_last')
    tmp18 =tl .load (in_ptr0 +(1 +2 *tmp6 +128 *x2 ),None ,eviction_policy ='evict_last')
    tmp19 =triton_helpers .maximum (tmp18 ,tmp17 )
    tmp20 =tmp19 -tmp19 
    tmp21 =tmp20 *tmp4 
    tmp22 =tmp19 +tmp21 
    tmp23 =tmp16 -tmp22 
    tmp24 =tmp6 .to (tl .float32 )
    tmp25 =tmp5 -tmp24 
    tmp26 =triton_helpers .maximum (tmp25 ,tmp4 )
    tmp27 =1.0 
    tmp28 =triton_helpers .minimum (tmp26 ,tmp27 )
    tmp29 =tmp23 *tmp28 
    tmp30 =tmp22 +tmp29 
    tl .store (in_out_ptr0 +(x4 ),tmp30 ,None )

def call (args ):
    arg0_1 ,=args 
    args .clear ()
    assert_size_stride (arg0_1 ,(1 ,64 ,128 ),(8192 ,128 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf3 =empty_strided_cuda ((1 ,64 ,128 ),(8192 ,128 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused__native_batch_norm_legit_0 [grid (64 )](arg0_1 ,buf3 ,64 ,128 ,XBLOCK =8 ,num_warps =8 ,num_stages =1 )
        del arg0_1 
        buf4 =empty_strided_cuda ((1 ,64 ,128 ,2 ),(16384 ,256 ,2 ,1 ),torch .float32 )
        buf5 =buf4 ;del buf4 

        get_raw_stream (0 )
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_1 [grid (16384 )](buf5 ,buf3 ,16384 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del buf3 
    return (buf5 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =rand_strided ((1 ,64 ,128 ),(8192 ,128 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
