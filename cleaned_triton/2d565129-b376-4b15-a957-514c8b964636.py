
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
def triton_per_fused_add_clamp_min_fill_huber_loss_mean_mul_ne_neg_randn_like_sub_where_zeros_like_0 (in_out_ptr0 ,in_ptr0 ,load_seed_offset ,load_seed_offset1 ,load_seed_offset2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    r0_numel =100 
    R0_BLOCK :tl .constexpr =128 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    r0_mask =r0_index <r0_numel 
    r0_0 =r0_index 
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =r0_0 
    tmp2 =tl .randn (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tmp3 =tl .load (in_ptr0 +load_seed_offset1 )
    tmp4 =tl .randn (tmp3 ,(tmp1 ).to (tl .uint32 ))
    tmp5 =tl .load (in_ptr0 +load_seed_offset2 )
    tmp6 =tl .randn (tmp5 ,(tmp1 ).to (tl .uint32 ))
    tmp7 =-tmp2 
    tmp8 =tmp4 -tmp6 
    tmp9 =tmp7 *tmp8 
    tmp10 =0.0 
    tmp11 =tmp9 +tmp10 
    tmp12 =triton_helpers .maximum (tmp11 ,tmp10 )
    tmp13 =tl .broadcast_to (tmp12 ,[XBLOCK ,R0_BLOCK ])
    tmp15 =tl .where (r0_mask ,tmp13 ,0 )
    tmp16 =tl .sum (tmp15 ,1 )[:,None ]
    tmp17 =1.0 
    tmp18 =tmp2 !=tmp17 
    tmp19 =tmp17 -tmp4 
    tmp20 =triton_helpers .maximum (tmp19 ,tmp10 )
    tmp21 =tl .where (tmp18 ,tmp20 ,tmp10 )
    tmp22 =-1.0 
    tmp23 =tmp2 !=tmp22 
    tmp24 =tl .where (tmp23 ,tmp4 ,tmp10 )
    tmp25 =tmp21 +tmp24 
    tmp26 =tl .broadcast_to (tmp25 ,[XBLOCK ,R0_BLOCK ])
    tmp28 =tl .where (r0_mask ,tmp26 ,0 )
    tmp29 =tl .sum (tmp28 ,1 )[:,None ]
    tmp30 =tmp4 -tmp2 
    tmp31 =tl_math .abs (tmp30 )
    tmp32 =tmp31 <tmp17 
    tmp33 =0.5 
    tmp34 =tmp31 *tmp33 
    tmp35 =tmp34 *tmp31 
    tmp36 =tmp31 -tmp33 
    tmp37 =tmp36 *tmp17 
    tmp38 =tl .where (tmp32 ,tmp35 ,tmp37 )
    tmp39 =tl .broadcast_to (tmp38 ,[XBLOCK ,R0_BLOCK ])
    tmp41 =tl .where (r0_mask ,tmp39 ,0 )
    tmp42 =tl .sum (tmp41 ,1 )[:,None ]
    tmp43 =100.0 
    tmp44 =tmp16 /tmp43 
    tmp45 =tmp29 /tmp43 
    tmp46 =tmp44 +tmp45 
    tmp47 =tmp42 /tmp43 
    tmp48 =tmp46 +tmp47 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp48 ,None )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    assert_size_stride (arg1_1 ,(1 ,100 ),(100 ,1 ))
    assert_size_stride (arg2_1 ,(10 ,10 ),(10 ,1 ))
    assert_size_stride (arg3_1 ,(10 ,10 ),(10 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((4 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[4 ],out =buf0 )
        buf5 =empty_strided_cuda ((),(),torch .float32 )
        buf8 =buf5 ;del buf5 

        get_raw_stream (0 )
        triton_per_fused_add_clamp_min_fill_huber_loss_mean_mul_ne_neg_randn_like_sub_where_zeros_like_0 [grid (1 )](buf8 ,buf0 ,3 ,1 ,2 ,1 ,100 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        del buf0 
    return (buf8 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =100 
    arg1_1 =rand_strided ((1 ,100 ),(100 ,1 ),device ='cuda:0',dtype =torch .float32 )
    arg2_1 =rand_strided ((10 ,10 ),(10 ,1 ),device ='cuda:0',dtype =torch .float32 )
    arg3_1 =rand_strided ((10 ,10 ),(10 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
