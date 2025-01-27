
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
def triton_poi_fused_add_view_0 (in_out_ptr0 ,in_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =20 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_out_ptr0 +(x0 ),xmask )
    tmp1 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp2 =tmp0 +tmp1 
    tl .store (in_out_ptr0 +(x0 ),tmp2 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused__softmax_add_clamp_min_fill_mean_ne_sub_where_zeros_like_1 (in_out_ptr0 ,in_out_ptr1 ,in_ptr0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    r0_numel =10 
    R0_BLOCK :tl .constexpr =16 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    r0_mask =r0_index <r0_numel 
    r0_0 =r0_index 
    tmp0 =tl .load (in_out_ptr0 +(r0_0 ),r0_mask ,other =0.0 )
    tmp1 =tl .load (in_ptr0 +(r0_0 ),r0_mask ,other =0.0 )
    tmp2 =tmp0 +tmp1 
    tmp3 =tl .broadcast_to (tmp2 ,[XBLOCK ,R0_BLOCK ])
    tmp5 =tl .where (r0_mask ,tmp3 ,float ("-inf"))
    tmp6 =triton_helpers .max2 (tmp5 ,1 )[:,None ]
    tmp7 =tmp2 -tmp6 
    tmp8 =tl_math .exp (tmp7 )
    tmp9 =tl .broadcast_to (tmp8 ,[XBLOCK ,R0_BLOCK ])
    tmp11 =tl .where (r0_mask ,tmp9 ,0 )
    tmp12 =tl .sum (tmp11 ,1 )[:,None ]
    tmp13 =tmp8 /tmp12 
    tmp14 =1.0 
    tmp15 =tmp14 -tmp13 
    tmp16 =0.0 
    tmp17 =triton_helpers .maximum (tmp15 ,tmp16 )
    tmp18 =tl .full ([1 ,1 ],False ,tl .int1 )
    tmp19 =tl .where (tmp18 ,tmp17 ,tmp16 )
    tmp20 =tl .full ([1 ,1 ],True ,tl .int1 )
    tmp21 =tl .where (tmp20 ,tmp13 ,tmp16 )
    tmp22 =tmp19 +tmp21 
    tmp23 =tl .broadcast_to (tmp22 ,[XBLOCK ,R0_BLOCK ])
    tmp25 =tl .where (r0_mask ,tmp23 ,0 )
    tmp26 =tl .sum (tmp25 ,1 )[:,None ]
    tmp27 =10.0 
    tmp28 =tmp26 /tmp27 
    tl .store (in_out_ptr0 +(tl .broadcast_to (r0_0 ,[XBLOCK ,R0_BLOCK ])),tmp13 ,r0_mask )
    tl .debug_barrier ()
    tl .store (in_out_ptr1 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp28 ,None )

def call (args ):
    primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 =args 
    args .clear ()
    assert_size_stride (primals_1 ,(1 ,10 ),(10 ,1 ))
    assert_size_stride (primals_2 ,(20 ,10 ,10 ),(100 ,10 ,1 ))
    assert_size_stride (primals_3 ,(20 ,),(1 ,))
    assert_size_stride (primals_4 ,(10 ,20 ,20 ),(400 ,20 ,1 ))
    assert_size_stride (primals_5 ,(10 ,),(1 ,))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )

        buf0 =torch .ops .aten ._trilinear .default (primals_1 ,primals_2 ,primals_1 ,[1 ,3 ],[0 ],[1 ,2 ],[2 ,3 ])
        del primals_2 
        buf1 =buf0 
        del buf0 
        buf2 =buf1 ;del buf1 

        get_raw_stream (0 )
        triton_poi_fused_add_view_0 [grid (20 )](buf2 ,primals_3 ,20 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        del primals_3 

        buf3 =torch .ops .aten ._trilinear .default (buf2 ,primals_4 ,buf2 ,[1 ,3 ],[0 ],[1 ,2 ],[2 ,3 ])
        buf4 =buf3 
        del buf3 
        buf7 =buf4 ;del buf4 
        buf8 =empty_strided_cuda ((),(),torch .float32 )
        buf9 =buf8 ;del buf8 

        get_raw_stream (0 )
        triton_per_fused__softmax_add_clamp_min_fill_mean_ne_sub_where_zeros_like_1 [grid (1 )](buf7 ,buf9 ,primals_5 ,1 ,10 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        del primals_5 
    return (buf9 ,primals_4 ,primals_1 ,buf2 ,buf7 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    primals_1 =rand_strided ((1 ,10 ),(10 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_2 =rand_strided ((20 ,10 ,10 ),(100 ,10 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_3 =rand_strided ((20 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_4 =rand_strided ((10 ,20 ,20 ),(400 ,20 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_5 =rand_strided ((10 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
