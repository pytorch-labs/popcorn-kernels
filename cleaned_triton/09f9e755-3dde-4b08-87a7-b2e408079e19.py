
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
def triton_red_fused_abs_gt_hardsigmoid_mean_mul_sign_sub_where_0 (in_out_ptr0 ,in_ptr0 ,ks0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    _tmp30 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_0 =r0_index 
        tmp0 =tl .load (in_ptr0 +(2 *r0_0 ),r0_mask ,eviction_policy ='evict_last',other =0.0 )
        tmp1 =tl .load (in_ptr0 +(1 +2 *r0_0 ),r0_mask ,eviction_policy ='evict_last',other =0.0 )
        tmp3 =tl .load (in_ptr0 +(2 +2 *r0_0 ),r0_mask ,eviction_policy ='evict_last',other =0.0 )
        tmp2 =tmp1 +tmp0 
        tmp4 =tmp3 +tmp2 
        tmp5 =0.3333333333333333 
        tmp6 =tmp4 *tmp5 
        tmp7 =3.0 
        tmp8 =tmp6 +tmp7 
        tmp9 =0.0 
        tmp10 =triton_helpers .maximum (tmp8 ,tmp9 )
        tmp11 =6.0 
        tmp12 =triton_helpers .minimum (tmp10 ,tmp11 )
        tmp13 =0.16666666666666666 
        tmp14 =tmp12 *tmp13 
        tmp15 =tl_math .abs (tmp14 )
        tmp16 =0.5 
        tmp17 =tmp15 >tmp16 
        tmp18 =tl .full ([1 ,1 ],0 ,tl .int32 )
        tmp19 =tmp18 <tmp14 
        tmp20 =tmp19 .to (tl .int8 )
        tmp21 =tmp14 <tmp18 
        tmp22 =tmp21 .to (tl .int8 )
        tmp23 =tmp20 -tmp22 
        tmp24 =tmp23 .to (tmp14 .dtype )
        tmp25 =tmp24 *tmp16 
        tmp26 =tmp14 -tmp25 
        tmp27 =tmp14 *tmp9 
        tmp28 =tl .where (tmp17 ,tmp26 ,tmp27 )
        tmp29 =tl .broadcast_to (tmp28 ,[XBLOCK ,R0_BLOCK ])
        tmp31 =_tmp30 +tmp29 
        _tmp30 =tl .where (r0_mask ,tmp31 ,_tmp30 )
    tmp30 =tl .sum (_tmp30 ,1 )[:,None ]
    tmp32 =triton_helpers .div_floor_integer ((-1 )+ks0 ,2 )
    tmp33 =tmp32 .to (tl .float32 )
    tmp34 =tmp30 /tmp33 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp34 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused_mul_smooth_l1_loss_smooth_l1_loss_backward_1 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    r0_numel =10 
    R0_BLOCK :tl .constexpr =16 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    r0_mask =r0_index <r0_numel 
    r0_0 =r0_index 
    tmp0 =tl .load (in_ptr0 +(0 ))
    tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
    tmp2 =tl .load (in_ptr1 +(r0_0 ),r0_mask ,other =0.0 )
    tmp4 =tl .load (in_ptr2 +(r0_0 ),r0_mask ,other =0.0 )
    tmp3 =tmp1 *tmp2 
    tmp5 =tmp3 +tmp4 
    tmp6 =tl_math .abs (tmp5 )
    tmp7 =1.0 
    tmp8 =tmp6 <tmp7 
    tmp9 =tmp6 *tmp6 
    tmp10 =0.5 
    tmp11 =tmp9 *tmp10 
    tmp12 =tmp11 *tmp7 
    tmp13 =tmp6 -tmp10 
    tmp14 =tl .where (tmp8 ,tmp12 ,tmp13 )
    tmp15 =tl .broadcast_to (tmp14 ,[XBLOCK ,R0_BLOCK ])
    tmp17 =tl .where (r0_mask ,tmp15 ,0 )
    tmp18 =tl .sum (tmp17 ,1 )[:,None ]
    tmp19 =10.0 
    tmp20 =tmp18 /tmp19 
    tl .store (out_ptr0 +(tl .broadcast_to (r0_0 ,[XBLOCK ,R0_BLOCK ])),tmp5 ,r0_mask )
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp20 ,None )

def call (args ):
    primals_1 ,primals_2 ,primals_3 ,primals_4 =args 
    args .clear ()
    s0 =primals_1 
    assert_size_stride (primals_2 ,(s0 ,),(1 ,))
    assert_size_stride (primals_3 ,(10 ,),(1 ,))
    assert_size_stride (primals_4 ,(10 ,),(1 ,))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,1 ),(1 ,1 ),torch .float32 )
        buf1 =buf0 ;del buf0 

        ((-1 )+s0 )//2 
        get_raw_stream (0 )
        triton_red_fused_abs_gt_hardsigmoid_mean_mul_sign_sub_where_0 [grid (1 )](buf1 ,primals_2 ,10 ,1 ,4 ,XBLOCK =1 ,R0_BLOCK =4 ,num_warps =2 ,num_stages =1 )
        del primals_2 
        buf2 =empty_strided_cuda ((1 ,10 ),(10 ,1 ),torch .float32 )
        buf3 =empty_strided_cuda ((),(),torch .float32 )
        buf4 =buf3 ;del buf3 

        get_raw_stream (0 )
        triton_per_fused_mul_smooth_l1_loss_smooth_l1_loss_backward_1 [grid (1 )](buf4 ,buf1 ,primals_3 ,primals_4 ,buf2 ,1 ,10 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        del primals_3 
        del primals_4 
    return (buf4 ,buf1 ,buf2 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    primals_1 =10 
    primals_2 =rand_strided ((10 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_3 =rand_strided ((10 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_4 =rand_strided ((10 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([primals_1 ,primals_2 ,primals_3 ,primals_4 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
