
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
def triton_red_fused__log_softmax__to_copy_add_bernoulli_hardswish_mse_loss_mul_zeros_like_0 (in_out_ptr0 ,in_out_ptr1 ,in_ptr0 ,load_seed_offset ,load_seed_offset1 ,load_seed_offset2 ,ks3 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    _tmp57 =tl .full ([XBLOCK ,R0_BLOCK ],float ("-inf"),tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_0 =r0_index 
        tmp7 =tl .load (in_out_ptr0 +(r0_0 ),r0_mask ,eviction_policy ='evict_first',other =0.0 )
        tmp0 =tl .load (in_ptr0 +load_seed_offset )
        tmp1 =r0_0 
        tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
        tmp3 =tl .load (in_ptr0 +load_seed_offset1 )
        tmp4 =tl .rand (tmp3 ,(tmp1 ).to (tl .uint32 ))
        tmp5 =tl .load (in_ptr0 +load_seed_offset2 )
        tmp6 =tl .rand (tmp5 ,(tmp1 ).to (tl .uint32 ))
        tmp8 =3.0 
        tmp9 =tmp7 +tmp8 
        tmp10 =0.0 
        tmp11 =triton_helpers .maximum (tmp9 ,tmp10 )
        tmp12 =6.0 
        tmp13 =triton_helpers .minimum (tmp11 ,tmp12 )
        tmp14 =tmp7 *tmp13 
        tmp15 =0.16666666666666666 
        tmp16 =tmp14 *tmp15 
        tmp17 =0.5 
        tmp18 =tmp2 <tmp17 
        tmp19 =tmp18 .to (tl .float32 )
        tmp20 =0.8864048946659319 
        tmp21 =tmp19 *tmp20 
        tmp22 =tmp16 *tmp21 
        tmp23 =-1.0 
        tmp24 =tmp19 +tmp23 
        tmp25 =1.558387861036063 
        tmp26 =tmp24 *tmp25 
        tmp27 =0.7791939305180315 
        tmp28 =tmp26 +tmp27 
        tmp29 =tmp22 +tmp28 
        tmp30 =tmp29 +tmp8 
        tmp31 =triton_helpers .maximum (tmp30 ,tmp10 )
        tmp32 =triton_helpers .minimum (tmp31 ,tmp12 )
        tmp33 =tmp29 *tmp32 
        tmp34 =tmp33 *tmp15 
        tmp35 =tmp6 <tmp17 
        tmp36 =tmp35 .to (tl .float32 )
        tmp37 =tmp36 *tmp20 
        tmp38 =tmp34 *tmp37 
        tmp39 =tmp36 +tmp23 
        tmp40 =tmp39 *tmp25 
        tmp41 =tmp40 +tmp27 
        tmp42 =tmp38 +tmp41 
        tmp43 =tmp42 +tmp8 
        tmp44 =triton_helpers .maximum (tmp43 ,tmp10 )
        tmp45 =triton_helpers .minimum (tmp44 ,tmp12 )
        tmp46 =tmp42 *tmp45 
        tmp47 =tmp46 *tmp15 
        tmp48 =tmp4 <tmp17 
        tmp49 =tmp48 .to (tl .float32 )
        tmp50 =tmp49 *tmp20 
        tmp51 =tmp47 *tmp50 
        tmp52 =tmp49 +tmp23 
        tmp53 =tmp52 *tmp25 
        tmp54 =tmp53 +tmp27 
        tmp55 =tmp51 +tmp54 
        tmp56 =tl .broadcast_to (tmp55 ,[XBLOCK ,R0_BLOCK ])
        tmp58 =triton_helpers .maximum (_tmp57 ,tmp56 )
        _tmp57 =tl .where (r0_mask ,tmp58 ,_tmp57 )
        tl .store (in_out_ptr0 +(tl .broadcast_to (r0_0 ,[XBLOCK ,R0_BLOCK ])),tmp55 ,r0_mask )
    tmp57 =triton_helpers .max2 (_tmp57 ,1 )[:,None ]
    _tmp63 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_0 =r0_index 
        tmp59 =tl .load (in_out_ptr0 +(r0_0 ),r0_mask ,eviction_policy ='evict_last',other =0.0 )
        tmp60 =tmp59 -tmp57 
        tmp61 =tl_math .exp (tmp60 )
        tmp62 =tl .broadcast_to (tmp61 ,[XBLOCK ,R0_BLOCK ])
        tmp64 =_tmp63 +tmp62 
        _tmp63 =tl .where (r0_mask ,tmp64 ,_tmp63 )
    tmp63 =tl .sum (_tmp63 ,1 )[:,None ]
    _tmp73 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_0 =r0_index 
        tmp65 =tl .load (in_out_ptr0 +(r0_0 ),r0_mask ,eviction_policy ='evict_first',other =0.0 )
        tmp66 =tmp65 -tmp57 
        tmp67 =tl_math .log (tmp63 )
        tmp68 =tmp66 -tmp67 
        tmp69 =0.0 
        tmp70 =tmp68 -tmp69 
        tmp71 =tmp70 *tmp70 
        tmp72 =tl .broadcast_to (tmp71 ,[XBLOCK ,R0_BLOCK ])
        tmp74 =_tmp73 +tmp72 
        _tmp73 =tl .where (r0_mask ,tmp74 ,_tmp73 )
    tmp73 =tl .sum (_tmp73 ,1 )[:,None ]
    tmp75 =49 *ks3 
    tmp76 =tmp75 .to (tl .float32 )
    tmp77 =tmp73 /tmp76 
    tl .debug_barrier ()
    tl .store (in_out_ptr1 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp77 ,None )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,64 ,64 ),(4096 *s0 ,4096 ,64 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )

        buf0 =torch .ops .aten ._adaptive_avg_pool2d .default (arg3_1 ,[7 ,7 ])
        del arg3_1 
        buf1 =buf0 
        del buf0 
        buf2 =empty_strided_cuda ((3 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[3 ],out =buf2 )
        buf5 =buf1 ;del buf1 
        buf7 =buf5 ;del buf5 
        buf8 =empty_strided_cuda ((1 ,1 ),(1 ,1 ),torch .float32 )
        buf10 =reinterpret_tensor (buf8 ,(),(),0 );del buf8 
        buf11 =buf10 ;del buf10 

        49 *s0 
        get_raw_stream (0 )
        triton_red_fused__log_softmax__to_copy_add_bernoulli_hardswish_mse_loss_mul_zeros_like_0 [grid (1 )](buf7 ,buf11 ,buf2 ,0 ,2 ,1 ,3 ,1 ,147 ,XBLOCK =1 ,R0_BLOCK =256 ,num_warps =2 ,num_stages =1 )
        del buf2 
        del buf7 
    return (buf11 ,)

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
