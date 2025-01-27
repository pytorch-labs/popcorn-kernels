
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
def triton_per_fused_mean_sub_0 (in_ptr0 ,out_ptr0 ,out_ptr1 ,out_ptr2 ,xnumel ,r0_numel ):
    XBLOCK :tl .constexpr =1 
    R0_BLOCK :tl .constexpr =1024 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    tl .full ([1 ],xoffset ,tl .int32 )
    tl .full ([R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[:]
    tl .full ([R0_BLOCK ],True ,tl .int1 )
    r0_0 =(r0_index %32 )
    r0_1 =r0_index //32 
    r0_2 =r0_index 
    tmp0 =tl .load (in_ptr0 +(2 *(r0_0 //2 )+128 *(r0_1 //2 )+4096 *((r0_0 %2 ))+8192 *((r0_1 %2 ))),None ,eviction_policy ='evict_last')
    tmp2 =tl .load (in_ptr0 +(1 +2 *(r0_0 //2 )+128 *(r0_1 //2 )+4096 *((r0_0 %2 ))+8192 *((r0_1 %2 ))),None ,eviction_policy ='evict_last')
    tmp5 =tl .load (in_ptr0 +(64 +2 *(r0_0 //2 )+128 *(r0_1 //2 )+4096 *((r0_0 %2 ))+8192 *((r0_1 %2 ))),None ,eviction_policy ='evict_last')
    tmp8 =tl .load (in_ptr0 +(65 +2 *(r0_0 //2 )+128 *(r0_1 //2 )+4096 *((r0_0 %2 ))+8192 *((r0_1 %2 ))),None ,eviction_policy ='evict_last')
    tmp26 =tl .load (in_ptr0 +(32 +2 *(r0_0 //2 )+128 *(r0_1 //2 )+4096 *((r0_0 %2 ))+8192 *((r0_1 %2 ))),None ,eviction_policy ='evict_last')
    tmp28 =tl .load (in_ptr0 +(33 +2 *(r0_0 //2 )+128 *(r0_1 //2 )+4096 *((r0_0 %2 ))+8192 *((r0_1 %2 ))),None ,eviction_policy ='evict_last')
    tmp31 =tl .load (in_ptr0 +(96 +2 *(r0_0 //2 )+128 *(r0_1 //2 )+4096 *((r0_0 %2 ))+8192 *((r0_1 %2 ))),None ,eviction_policy ='evict_last')
    tmp34 =tl .load (in_ptr0 +(97 +2 *(r0_0 //2 )+128 *(r0_1 //2 )+4096 *((r0_0 %2 ))+8192 *((r0_1 %2 ))),None ,eviction_policy ='evict_last')
    tmp50 =tl .load (in_ptr0 +(2048 +2 *(r0_0 //2 )+128 *(r0_1 //2 )+4096 *((r0_0 %2 ))+8192 *((r0_1 %2 ))),None ,eviction_policy ='evict_last')
    tmp52 =tl .load (in_ptr0 +(2049 +2 *(r0_0 //2 )+128 *(r0_1 //2 )+4096 *((r0_0 %2 ))+8192 *((r0_1 %2 ))),None ,eviction_policy ='evict_last')
    tmp55 =tl .load (in_ptr0 +(2112 +2 *(r0_0 //2 )+128 *(r0_1 //2 )+4096 *((r0_0 %2 ))+8192 *((r0_1 %2 ))),None ,eviction_policy ='evict_last')
    tmp58 =tl .load (in_ptr0 +(2113 +2 *(r0_0 //2 )+128 *(r0_1 //2 )+4096 *((r0_0 %2 ))+8192 *((r0_1 %2 ))),None ,eviction_policy ='evict_last')
    tmp1 =tmp0 *tmp0 
    tmp3 =tmp2 *tmp2 
    tmp4 =tmp3 +tmp1 
    tmp6 =tmp5 *tmp5 
    tmp7 =tmp6 +tmp4 
    tmp9 =tmp8 *tmp8 
    tmp10 =tmp9 +tmp7 
    tmp11 =0.25 
    tmp12 =tmp10 *tmp11 
    tmp13 =tl .full ([1 ],0 ,tl .int32 )
    tmp14 =tmp13 <tmp12 
    tmp15 =tmp14 .to (tl .int8 )
    tmp16 =tmp12 <tmp13 
    tmp17 =tmp16 .to (tl .int8 )
    tmp18 =tmp15 -tmp17 
    tmp19 =tmp18 .to (tmp12 .dtype )
    tmp20 =tl_math .abs (tmp12 )
    tmp21 =triton_helpers .maximum (tmp13 ,tmp20 )
    tmp22 =tmp19 *tmp21 
    tmp23 =4.0 
    tmp24 =tmp22 *tmp23 
    tmp25 =libdevice .sqrt (tmp24 )
    tmp27 =tmp26 *tmp26 
    tmp29 =tmp28 *tmp28 
    tmp30 =tmp29 +tmp27 
    tmp32 =tmp31 *tmp31 
    tmp33 =tmp32 +tmp30 
    tmp35 =tmp34 *tmp34 
    tmp36 =tmp35 +tmp33 
    tmp37 =tmp36 *tmp11 
    tmp38 =tmp13 <tmp37 
    tmp39 =tmp38 .to (tl .int8 )
    tmp40 =tmp37 <tmp13 
    tmp41 =tmp40 .to (tl .int8 )
    tmp42 =tmp39 -tmp41 
    tmp43 =tmp42 .to (tmp37 .dtype )
    tmp44 =tl_math .abs (tmp37 )
    tmp45 =triton_helpers .maximum (tmp13 ,tmp44 )
    tmp46 =tmp43 *tmp45 
    tmp47 =tmp46 *tmp23 
    tmp48 =libdevice .sqrt (tmp47 )
    tmp49 =tmp25 -tmp48 
    tmp51 =tmp50 *tmp50 
    tmp53 =tmp52 *tmp52 
    tmp54 =tmp53 +tmp51 
    tmp56 =tmp55 *tmp55 
    tmp57 =tmp56 +tmp54 
    tmp59 =tmp58 *tmp58 
    tmp60 =tmp59 +tmp57 
    tmp61 =tmp60 *tmp11 
    tmp62 =tmp13 <tmp61 
    tmp63 =tmp62 .to (tl .int8 )
    tmp64 =tmp61 <tmp13 
    tmp65 =tmp64 .to (tl .int8 )
    tmp66 =tmp63 -tmp65 
    tmp67 =tmp66 .to (tmp61 .dtype )
    tmp68 =tl_math .abs (tmp61 )
    tmp69 =triton_helpers .maximum (tmp13 ,tmp68 )
    tmp70 =tmp67 *tmp69 
    tmp71 =tmp70 *tmp23 
    tmp72 =libdevice .sqrt (tmp71 )
    tmp73 =tmp25 -tmp72 
    tmp74 =tl .broadcast_to (tmp25 ,[R0_BLOCK ])
    tmp76 =triton_helpers .promote_to_tensor (tl .sum (tmp74 ,0 ))
    tl .store (out_ptr0 +(tl .broadcast_to (r0_2 ,[R0_BLOCK ])),tmp49 ,None )
    tl .store (out_ptr1 +(tl .broadcast_to (r0_2 ,[R0_BLOCK ])),tmp73 ,None )
    tl .store (out_ptr2 +(tl .full ([1 ],0 ,tl .int32 )),tmp76 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused_add_norm_1 (in_ptr0 ,out_ptr0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    xnumel =32 
    R0_BLOCK :tl .constexpr =32 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_1 =r0_index 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(r0_1 +32 *x0 ),xmask ,other =0.0 )
    tmp1 =1e-06 
    tmp2 =tmp0 +tmp1 
    tmp3 =tmp2 *tmp2 
    tmp4 =tl .broadcast_to (tmp3 ,[XBLOCK ,R0_BLOCK ])
    tmp6 =tl .where (xmask ,tmp4 ,0 )
    tmp7 =tl .sum (tmp6 ,1 )[:,None ]
    tl .store (out_ptr0 +(x0 ),tmp7 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused_add_clamp_min_fill_mean_ne_norm_sub_where_zeros_like_2 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    R0_BLOCK :tl .constexpr =32 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_0 =r0_index 
    tmp0 =tl .load (in_ptr0 +(r0_0 ),None )
    tmp4 =tl .load (in_ptr1 +(r0_0 ),None )
    tmp14 =tl .load (in_ptr2 +(0 ))
    tmp15 =tl .broadcast_to (tmp14 ,[XBLOCK ,1 ])
    tmp1 =libdevice .sqrt (tmp0 )
    tmp2 =1.0 
    tmp3 =tmp1 +tmp2 
    tmp5 =libdevice .sqrt (tmp4 )
    tmp6 =tmp3 -tmp5 
    tmp7 =0.0 
    tmp8 =triton_helpers .maximum (tmp6 ,tmp7 )
    tmp9 =tl .broadcast_to (tmp8 ,[XBLOCK ,R0_BLOCK ])
    tmp11 =tl .sum (tmp9 ,1 )[:,None ]
    tmp12 =32.0 
    tmp13 =tmp11 /tmp12 
    tmp16 =1024.0 
    tmp17 =tmp15 /tmp16 
    tmp18 =tmp2 -tmp17 
    tmp19 =triton_helpers .maximum (tmp18 ,tmp7 )
    tmp20 =tl .full ([1 ,1 ],False ,tl .int1 )
    tmp21 =tl .where (tmp20 ,tmp19 ,tmp7 )
    tmp22 =tl .full ([1 ,1 ],True ,tl .int1 )
    tmp23 =tl .where (tmp22 ,tmp17 ,tmp7 )
    tmp24 =tmp21 +tmp23 
    tmp25 =tmp24 /tmp2 
    tmp26 =tmp13 +tmp25 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp26 ,None )

def call (args ):
    arg0_1 ,=args 
    args .clear ()
    assert_size_stride (arg0_1 ,(1 ,4 ,64 ,64 ),(16384 ,4096 ,64 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,1 ,32 ,32 ),(1024 ,1024 ,32 ,1 ),torch .float32 )
        buf2 =empty_strided_cuda ((1 ,1 ,32 ,32 ),(1024 ,1024 ,32 ,1 ),torch .float32 )
        buf5 =empty_strided_cuda ((1 ,),(1 ,),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused_mean_sub_0 [grid (1 )](arg0_1 ,buf0 ,buf2 ,buf5 ,1 ,1024 ,num_warps =8 ,num_stages =1 )
        del arg0_1 
        buf1 =empty_strided_cuda ((1 ,1 ,32 ),(32 ,32 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused_add_norm_1 [grid (32 )](buf0 ,buf1 ,32 ,32 ,XBLOCK =32 ,num_warps =8 ,num_stages =1 )
        del buf0 
        buf3 =empty_strided_cuda ((1 ,1 ,32 ),(32 ,32 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused_add_norm_1 [grid (32 )](buf2 ,buf3 ,32 ,32 ,XBLOCK =32 ,num_warps =8 ,num_stages =1 )
        del buf2 
        buf4 =empty_strided_cuda ((),(),torch .float32 )
        buf6 =buf4 ;del buf4 

        get_raw_stream (0 )
        triton_per_fused_add_clamp_min_fill_mean_ne_norm_sub_where_zeros_like_2 [grid (1 )](buf6 ,buf1 ,buf3 ,buf5 ,1 ,32 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        del buf1 
        del buf3 
        del buf5 
    return (buf6 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =rand_strided ((1 ,4 ,64 ,64 ),(16384 ,4096 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
