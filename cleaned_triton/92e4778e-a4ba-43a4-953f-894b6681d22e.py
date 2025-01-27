
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
def triton_red_fused__unsafe_index_binary_cross_entropy_with_logits_constant_pad_nd_sigmoid_softplus_zeros_like_0 (in_out_ptr0 ,in_ptr0 ,ks0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    _tmp39 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_0 =r0_index 
        tmp0 =4.0 
        tmp1 =ks0 
        tmp2 =tmp1 .to (tl .float32 )
        tmp3 =tmp0 +tmp2 
        tmp4 =tmp3 .to (tl .float64 )
        tmp5 =tl .full ([1 ,1 ],2.0 ,tl .float64 )
        tmp6 =tmp5 *tmp4 
        tmp7 =tmp4 /tmp6 
        tmp8 =tmp7 .to (tl .float32 )
        tmp9 =r0_0 
        tmp10 =tmp9 .to (tl .float32 )
        tmp11 =tmp10 *tmp8 
        tmp12 =tmp11 .to (tl .int64 )
        tmp13 =(-2 )+tmp12 
        tmp14 =tmp13 .to (tl .int32 )
        tmp15 =tl .full ([1 ,1 ],0 ,tl .int64 )
        tmp16 =tmp14 >=tmp15 
        tmp17 =tmp14 <tmp1 
        tmp18 =tmp16 &tmp17 
        tmp19 =tl .load (in_ptr0 +(tl .broadcast_to ((-2 )+tmp12 ,[XBLOCK ,R0_BLOCK ])),r0_mask &tmp18 ,eviction_policy ='evict_last',other =0.0 )
        tmp20 =1.0 
        tmp21 =tmp19 *tmp20 
        tmp22 =20.0 
        tmp23 =tmp21 >tmp22 
        tmp24 =tl_math .exp (tmp21 )
        tmp25 =libdevice .log1p (tmp24 )
        tmp26 =tmp25 *tmp20 
        tmp27 =tl .where (tmp23 ,tmp19 ,tmp26 )
        tmp28 =tl .sigmoid (tmp27 )
        tmp29 =tmp20 *tmp28 
        tmp30 =0.0 
        tmp31 =triton_helpers .minimum (tmp30 ,tmp28 )
        tmp32 =tl_math .abs (tmp28 )
        tmp33 =-tmp32 
        tmp34 =tl_math .exp (tmp33 )
        tmp35 =libdevice .log1p (tmp34 )
        tmp36 =tmp31 -tmp35 
        tmp37 =tmp29 -tmp36 
        tmp38 =tl .broadcast_to (tmp37 ,[XBLOCK ,R0_BLOCK ])
        tmp40 =_tmp39 +tmp38 
        _tmp39 =tl .where (r0_mask ,tmp40 ,_tmp39 )
    tmp39 =tl .sum (_tmp39 ,1 )[:,None ]
    tmp41 =8 +2 *ks0 
    tmp42 =tmp41 .to (tl .float32 )
    tmp43 =tmp39 /tmp42 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp43 ,None )

def call (args ):
    arg0_1 ,arg1_1 =args 
    args .clear ()
    s0 =arg0_1 
    assert_size_stride (arg1_1 ,(1 ,1 ,s0 ),(s0 ,s0 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf1 =empty_strided_cuda ((),(),torch .float32 )
        buf2 =buf1 ;del buf1 

        8 +2 *s0 
        get_raw_stream (0 )
        triton_red_fused__unsafe_index_binary_cross_entropy_with_logits_constant_pad_nd_sigmoid_softplus_zeros_like_0 [grid (1 )](buf2 ,arg1_1 ,32 ,1 ,72 ,XBLOCK =1 ,R0_BLOCK =128 ,num_warps =2 ,num_stages =1 )
        del arg1_1 
    return (buf2 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =32 
    arg1_1 =rand_strided ((1 ,1 ,32 ),(32 ,32 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
