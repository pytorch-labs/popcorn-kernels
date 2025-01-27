
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
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0 (in_out_ptr0 ,in_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    tl .full ([XBLOCK ],True ,tl .int1 )
    x1 =((xindex //64 )%64 )
    x0 =(xindex %64 )
    x2 =xindex //4096 
    x4 =xindex 
    tmp0 =x1 
    tmp1 =tmp0 .to (tl .float32 )
    tmp2 =0.49206349206349204 
    tmp3 =tmp1 *tmp2 
    tmp4 =0.0 
    tmp5 =triton_helpers .maximum (tmp3 ,tmp4 )
    tmp6 =tmp5 .to (tl .int32 )
    tmp7 =tl .full ([1 ],1 ,tl .int64 )
    tmp8 =tmp6 +tmp7 
    tmp9 =tl .full ([1 ],31 ,tl .int64 )
    tmp10 =triton_helpers .minimum (tmp8 ,tmp9 )
    tmp11 =x0 
    tmp12 =tmp11 .to (tl .float32 )
    tmp13 =tmp12 *tmp2 
    tmp14 =triton_helpers .maximum (tmp13 ,tmp4 )
    tmp15 =tmp14 .to (tl .int32 )
    tmp16 =tl .load (in_ptr0 +(tmp15 +32 *tmp10 +1024 *x2 ),None ,eviction_policy ='evict_last')
    tmp17 =tmp15 +tmp7 
    tmp18 =triton_helpers .minimum (tmp17 ,tmp9 )
    tmp19 =tl .load (in_ptr0 +(tmp18 +32 *tmp10 +1024 *x2 ),None ,eviction_policy ='evict_last')
    tmp20 =tmp19 -tmp16 
    tmp21 =tmp15 .to (tl .float32 )
    tmp22 =tmp14 -tmp21 
    tmp23 =triton_helpers .maximum (tmp22 ,tmp4 )
    tmp24 =1.0 
    tmp25 =triton_helpers .minimum (tmp23 ,tmp24 )
    tmp26 =tmp20 *tmp25 
    tmp27 =tmp16 +tmp26 
    tmp28 =tl .load (in_ptr0 +(tmp15 +32 *tmp6 +1024 *x2 ),None ,eviction_policy ='evict_last')
    tmp29 =tl .load (in_ptr0 +(tmp18 +32 *tmp6 +1024 *x2 ),None ,eviction_policy ='evict_last')
    tmp30 =tmp29 -tmp28 
    tmp31 =tmp30 *tmp25 
    tmp32 =tmp28 +tmp31 
    tmp33 =tmp27 -tmp32 
    tmp34 =tmp6 .to (tl .float32 )
    tmp35 =tmp5 -tmp34 
    tmp36 =triton_helpers .maximum (tmp35 ,tmp4 )
    tmp37 =triton_helpers .minimum (tmp36 ,tmp24 )
    tmp38 =tmp33 *tmp37 
    tmp39 =tmp32 +tmp38 
    tl .store (in_out_ptr0 +(x4 ),tmp39 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__softmax_binary_cross_entropy_backward_1 (in_ptr0 ,in_ptr1 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    tl .full ([XBLOCK ],True ,tl .int1 )
    x2 =xindex 
    x0 =(xindex %4096 )
    tmp0 =tl .load (in_ptr0 +(x2 ),None )
    tmp1 =tl .load (in_ptr0 +(x0 ),None ,eviction_policy ='evict_last')
    tmp2 =tl .load (in_ptr0 +(4096 +x0 ),None ,eviction_policy ='evict_last')
    tmp4 =tl .load (in_ptr0 +(8192 +x0 ),None ,eviction_policy ='evict_last')
    tmp17 =tl .load (in_ptr1 +(x2 ),None )
    tmp3 =triton_helpers .maximum (tmp1 ,tmp2 )
    tmp5 =triton_helpers .maximum (tmp3 ,tmp4 )
    tmp6 =tmp0 -tmp5 
    tmp7 =tl_math .exp (tmp6 )
    tmp8 =tmp1 -tmp5 
    tmp9 =tl_math .exp (tmp8 )
    tmp10 =tmp2 -tmp5 
    tmp11 =tl_math .exp (tmp10 )
    tmp12 =tmp9 +tmp11 
    tmp13 =tmp4 -tmp5 
    tmp14 =tl_math .exp (tmp13 )
    tmp15 =tmp12 +tmp14 
    tmp16 =tmp7 /tmp15 
    tmp18 =tmp16 +tmp17 
    tl .store (out_ptr0 +(x2 ),tmp18 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused_binary_cross_entropy_zeros_like_2 (in_ptr0 ,out_ptr0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =2 
    r0_numel =6144 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    _tmp13 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_1 =r0_index 
        tmp0 =tl .load (in_ptr0 +(r0_1 +6144 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp1 =-tmp0 
        tmp2 =libdevice .log1p (tmp1 )
        tmp3 =-100.0 
        tmp4 =triton_helpers .maximum (tmp2 ,tmp3 )
        tmp5 =-1.0 
        tmp6 =tmp5 *tmp4 
        tmp7 =tl_math .log (tmp0 )
        tmp8 =triton_helpers .maximum (tmp7 ,tmp3 )
        tmp9 =0.0 
        tmp10 =tmp9 *tmp8 
        tmp11 =tmp6 -tmp10 
        tmp12 =tl .broadcast_to (tmp11 ,[XBLOCK ,R0_BLOCK ])
        tmp14 =_tmp13 +tmp12 
        _tmp13 =tl .where (r0_mask &xmask ,tmp14 ,_tmp13 )
    tmp13 =tl .sum (_tmp13 ,1 )[:,None ]
    tl .store (out_ptr0 +(x0 ),tmp13 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused_binary_cross_entropy_zeros_like_3 (in_out_ptr0 ,in_ptr0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    R0_BLOCK :tl .constexpr =2 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_0 =r0_index 
    tmp0 =tl .load (in_ptr0 +(r0_0 ),None )
    tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
    tmp3 =tl .sum (tmp1 ,1 )[:,None ]
    tmp4 =12288.0 
    tmp5 =tmp3 /tmp4 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp5 ,None )

def call (args ):
    primals_1 ,primals_2 =args 
    args .clear ()
    assert_size_stride (primals_1 ,(1 ,3 ,32 ,32 ),(3072 ,1024 ,32 ,1 ))
    assert_size_stride (primals_2 ,(1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ),torch .float32 )
        buf1 =buf0 ;del buf0 

        get_raw_stream (0 )
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0 [grid (12288 )](buf1 ,primals_1 ,12288 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del primals_1 
        buf2 =empty_strided_cuda ((1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused__softmax_binary_cross_entropy_backward_1 [grid (12288 )](buf1 ,primals_2 ,buf2 ,12288 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf1 
        del primals_2 
        buf3 =empty_strided_cuda ((2 ,),(1 ,),torch .float32 )

        get_raw_stream (0 )
        triton_red_fused_binary_cross_entropy_zeros_like_2 [grid (2 )](buf2 ,buf3 ,2 ,6144 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        buf4 =empty_strided_cuda ((),(),torch .float32 )
        buf5 =buf4 ;del buf4 

        get_raw_stream (0 )
        triton_per_fused_binary_cross_entropy_zeros_like_3 [grid (1 )](buf5 ,buf3 ,1 ,2 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        del buf3 
    return (buf5 ,buf2 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    primals_1 =rand_strided ((1 ,3 ,32 ,32 ),(3072 ,1024 ,32 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_2 =rand_strided ((1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([primals_1 ,primals_2 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
