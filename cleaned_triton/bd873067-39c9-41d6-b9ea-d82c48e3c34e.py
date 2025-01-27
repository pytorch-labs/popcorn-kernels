
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
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused_abs_gt_hardtanh_mse_loss_mul_sign_sub_where_0 (in_ptr0 ,out_ptr0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =2 
    r0_numel =6144 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    _tmp22 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_1 =r0_index 
        tmp0 =tl .load (in_ptr0 +(r0_1 +6144 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp1 =tl_math .abs (tmp0 )
        tmp2 =0.5 
        tmp3 =tmp1 >tmp2 
        tmp4 =tl .full ([1 ,1 ],0 ,tl .int32 )
        tmp5 =tmp4 <tmp0 
        tmp6 =tmp5 .to (tl .int8 )
        tmp7 =tmp0 <tmp4 
        tmp8 =tmp7 .to (tl .int8 )
        tmp9 =tmp6 -tmp8 
        tmp10 =tmp9 .to (tmp0 .dtype )
        tmp11 =tmp10 *tmp2 
        tmp12 =tmp0 -tmp11 
        tmp13 =0.0 
        tmp14 =tmp0 *tmp13 
        tmp15 =tl .where (tmp3 ,tmp12 ,tmp14 )
        tmp16 =-1.0 
        tmp17 =triton_helpers .maximum (tmp15 ,tmp16 )
        tmp18 =1.0 
        tmp19 =triton_helpers .minimum (tmp17 ,tmp18 )
        tmp20 =tmp19 *tmp19 
        tmp21 =tl .broadcast_to (tmp20 ,[XBLOCK ,R0_BLOCK ])
        tmp23 =_tmp22 +tmp21 
        _tmp22 =tl .where (r0_mask &xmask ,tmp23 ,_tmp22 )
    tmp22 =tl .sum (_tmp22 ,1 )[:,None ]
    tl .store (out_ptr0 +(x0 ),tmp22 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused_abs_gt_hardtanh_mse_loss_mul_sign_sub_where_1 (in_out_ptr0 ,in_ptr0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
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
    arg0_1 ,=args 
    args .clear ()
    assert_size_stride (arg0_1 ,(1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((2 ,),(1 ,),torch .float32 )

        get_raw_stream (0 )
        triton_red_fused_abs_gt_hardtanh_mse_loss_mul_sign_sub_where_0 [grid (2 )](arg0_1 ,buf0 ,2 ,6144 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del arg0_1 
        buf1 =empty_strided_cuda ((),(),torch .float32 )
        buf2 =buf1 ;del buf1 

        get_raw_stream (0 )
        triton_per_fused_abs_gt_hardtanh_mse_loss_mul_sign_sub_where_1 [grid (1 )](buf2 ,buf0 ,1 ,2 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        del buf0 
    return (buf2 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =rand_strided ((1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
