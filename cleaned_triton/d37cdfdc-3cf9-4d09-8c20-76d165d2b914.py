
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
def triton_red_fused_add_binary_cross_entropy_clamp_min_div_eq_exp_fill_mean_mul_randint_sigmoid_sqrt_sub_sum_where_zeros_like_0 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,load_seed_offset ,load_seed_offset1 ,ks2 ,load_seed_offset2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    _tmp20 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    _tmp24 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    _tmp34 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_0 =r0_index 
        tmp8 =tl .load (in_ptr1 +(r0_0 ),r0_mask ,eviction_policy ='evict_first',other =0.0 )
        tmp0 =tl .load (in_ptr0 +load_seed_offset )
        tmp1 =r0_0 
        tmp2 =tl .full ([1 ,1 ],0 ,tl .int64 )
        tmp3 =tl .full ([1 ,1 ],2 ,tl .int64 )
        tmp4 =triton_helpers .randint64 (tmp0 ,(tmp1 ).to (tl .uint32 ),tmp2 ,tmp3 )
        tmp5 =tmp4 .to (tl .float32 )
        tmp6 =1.0 
        tmp7 =tmp5 -tmp6 
        tmp9 =tl .sigmoid (tmp8 )
        tmp10 =-tmp9 
        tmp11 =libdevice .log1p (tmp10 )
        tmp12 =-100.0 
        tmp13 =triton_helpers .maximum (tmp11 ,tmp12 )
        tmp14 =tmp7 *tmp13 
        tmp15 =tl_math .log (tmp9 )
        tmp16 =triton_helpers .maximum (tmp15 ,tmp12 )
        tmp17 =tmp5 *tmp16 
        tmp18 =tmp14 -tmp17 
        tmp19 =tl .broadcast_to (tmp18 ,[XBLOCK ,R0_BLOCK ])
        tmp21 =_tmp20 +tmp19 
        _tmp20 =tl .where (r0_mask ,tmp21 ,_tmp20 )
        tmp22 =tmp8 *tmp8 
        tmp23 =tl .broadcast_to (tmp22 ,[XBLOCK ,R0_BLOCK ])
        tmp25 =_tmp24 +tmp23 
        _tmp24 =tl .where (r0_mask ,tmp25 ,_tmp24 )
        tmp26 =tl_math .exp (tmp8 )
        tmp27 =tl .load (in_ptr0 +load_seed_offset1 )
        tmp28 =tl .full ([1 ,1 ],10 ,tl .int64 )
        tmp29 =triton_helpers .randint64 (tmp27 ,(tmp1 ).to (tl .uint32 ),tmp2 ,tmp28 )
        tmp30 =tmp29 .to (tl .float32 )
        tmp31 =tmp30 *tmp8 
        tmp32 =tmp26 -tmp31 
        tmp33 =tl .broadcast_to (tmp32 ,[XBLOCK ,R0_BLOCK ])
        tmp35 =_tmp34 +tmp33 
        _tmp34 =tl .where (r0_mask ,tmp35 ,_tmp34 )
    tmp20 =tl .sum (_tmp20 ,1 )[:,None ]
    tmp24 =tl .sum (_tmp24 ,1 )[:,None ]
    tmp34 =tl .sum (_tmp34 ,1 )[:,None ]
    tmp36 =125 *ks2 
    tmp37 =tmp36 .to (tl .float32 )
    tmp38 =tmp20 /tmp37 
    tmp39 =tl .load (in_ptr0 +load_seed_offset2 )
    tmp40 =tl .full ([1 ,1 ],0 ,tl .int32 )
    tmp41 =tl .full ([1 ,1 ],0 ,tl .int64 )
    tmp42 =tl .full ([1 ,1 ],2 ,tl .int64 )
    tmp43 =triton_helpers .randint64 (tmp39 ,(tmp40 ).to (tl .uint32 ),tmp41 ,tmp42 )
    tmp44 =tmp43 .to (tl .float32 )
    tmp45 =1.0 
    tmp46 =tmp44 ==tmp45 
    tmp47 =9.999999960041972e-13 
    tmp48 =tmp24 +tmp47 
    tmp49 =tmp48 *tmp48 
    tmp50 =libdevice .sqrt (tmp49 )
    tmp51 =tmp24 /tmp50 
    tmp52 =tmp45 -tmp51 
    tmp53 =0.0 
    tmp54 =tl .where (tmp46 ,tmp52 ,tmp53 )
    tmp55 =-1.0 
    tmp56 =tmp44 ==tmp55 
    tmp57 =tmp51 -tmp53 
    tmp58 =triton_helpers .maximum (tmp57 ,tmp53 )
    tmp59 =tl .where (tmp56 ,tmp58 ,tmp53 )
    tmp60 =tmp54 +tmp59 
    tmp61 =tmp60 /tmp45 
    tmp62 =tmp38 +tmp61 
    tmp63 =tmp34 /tmp37 
    tmp64 =tmp62 +tmp63 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp64 ,None )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ,arg4_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    s3 =arg3_1 
    assert_size_stride (arg4_1 ,(1 ,s0 ,s1 ,s2 ,s3 ),(s0 *s1 *s2 *s3 ,s1 *s2 *s3 ,s2 *s3 ,s3 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )

        buf0 =torch .ops .aten .adaptive_max_pool3d .default (arg4_1 ,[5 ,5 ,5 ])
        del arg4_1 
        buf1 =buf0 [0 ]
        del buf0 
        buf3 =empty_strided_cuda ((3 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[3 ],out =buf3 )
        buf4 =empty_strided_cuda ((),(),torch .float32 )
        buf9 =buf4 ;del buf4 

        125 *s0 
        get_raw_stream (0 )
        triton_red_fused_add_binary_cross_entropy_clamp_min_div_eq_exp_fill_mean_mul_randint_sigmoid_sqrt_sub_sum_where_zeros_like_0 [grid (1 )](buf9 ,buf3 ,buf1 ,0 ,2 ,10 ,1 ,1 ,1250 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del buf1 
        del buf3 
    return (buf9 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =10 
    arg1_1 =20 
    arg2_1 =20 
    arg3_1 =20 
    arg4_1 =rand_strided ((1 ,10 ,20 ,20 ,20 ),(80000 ,8000 ,400 ,20 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ,arg4_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
