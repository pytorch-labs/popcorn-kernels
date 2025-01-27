
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
def triton_poi_fused_constant_pad_nd_0 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x1 =((xindex //ks0 )%ks1 )
    x0 =(xindex %ks0 )
    x2 =xindex //ks4 
    x4 =xindex 
    tmp0 =(-2 )+x1 
    tmp1 =tl .full ([1 ],0 ,tl .int64 )
    tmp2 =tmp0 >=tmp1 
    tmp3 =ks2 
    tmp4 =tmp0 <tmp3 
    tmp5 =(-2 )+x0 
    tmp6 =tmp5 >=tmp1 
    tmp7 =ks3 
    tmp8 =tmp5 <tmp7 
    tmp9 =tmp2 &tmp4 
    tmp10 =tmp9 &tmp6 
    tmp11 =tmp10 &tmp8 
    tmp12 =tl .load (in_ptr0 +((-2 )+x0 +((-2 )*ks3 )+ks3 *x1 +ks2 *ks3 *x2 ),tmp11 &xmask ,eviction_policy ='evict_last',other =3.0 )
    tl .store (out_ptr0 +(x4 ),tmp12 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__adaptive_avg_pool2d_1 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(4 *(((x0 //ks0 )%ks1 ))+16 *(x0 //(16 +4 *ks2 +4 *ks3 +ks2 *ks3 ))+ks3 *(((x0 //ks0 )%ks1 ))+4 *ks2 *(x0 //(16 +4 *ks2 +4 *ks3 +ks2 *ks3 ))+4 *ks3 *(x0 //(16 +4 *ks2 +4 *ks3 +ks2 *ks3 ))+ks2 *ks3 *(x0 //(16 +4 *ks2 +4 *ks3 +ks2 *ks3 ))+((x0 %ks0 ))),xmask ,eviction_policy ='evict_last')
    tl .store (out_ptr0 +(x0 ),tmp0 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused_add_clamp_min_elu_mean_mul_norm_randn_like_sub_2 (in_out_ptr0 ,in_out_ptr1 ,in_ptr0 ,load_seed_offset ,load_seed_offset1 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
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
    tmp1 =0.0 
    tmp2 =tmp0 >tmp1 
    tmp3 =1.0507009873554805 
    tmp4 =tmp0 *tmp3 
    tmp5 =1.0 
    tmp6 =tmp0 *tmp5 
    tmp7 =libdevice .expm1 (tmp6 )
    tmp8 =1.7580993408473766 
    tmp9 =tmp7 *tmp8 
    tmp10 =tl .where (tmp2 ,tmp4 ,tmp9 )
    tmp11 =tmp10 >tmp1 
    tmp12 =tmp10 *tmp3 
    tmp13 =tmp10 *tmp5 
    tmp14 =libdevice .expm1 (tmp13 )
    tmp15 =tmp14 *tmp8 
    tmp16 =tl .where (tmp11 ,tmp12 ,tmp15 )
    tmp17 =tmp16 >tmp1 
    tmp18 =tmp16 *tmp3 
    tmp19 =tmp16 *tmp5 
    tmp20 =libdevice .expm1 (tmp19 )
    tmp21 =tmp20 *tmp8 
    tmp22 =tl .where (tmp17 ,tmp18 ,tmp21 )
    tmp23 =tmp22 >tmp1 
    tmp24 =tmp22 *tmp3 
    tmp25 =tmp22 *tmp5 
    tmp26 =libdevice .expm1 (tmp25 )
    tmp27 =tmp26 *tmp8 
    tmp28 =tl .where (tmp23 ,tmp24 ,tmp27 )
    tmp29 =tmp28 >tmp1 
    tmp30 =tmp28 *tmp3 
    tmp31 =tmp28 *tmp5 
    tmp32 =libdevice .expm1 (tmp31 )
    tmp33 =tmp32 *tmp8 
    tmp34 =tl .where (tmp29 ,tmp30 ,tmp33 )
    tmp35 =tl .load (in_ptr0 +load_seed_offset )
    tmp36 =r0_0 
    tmp37 =tl .randn (tmp35 ,(tmp36 ).to (tl .uint32 ))
    tmp38 =tl .load (in_ptr0 +load_seed_offset1 )
    tmp39 =tl .randn (tmp38 ,(tmp36 ).to (tl .uint32 ))
    tmp40 =0.1 
    tmp41 =tmp39 *tmp40 
    tmp42 =tmp34 +tmp41 
    tmp43 =tmp34 -tmp42 
    tmp44 =1e-06 
    tmp45 =tmp43 +tmp44 
    tmp46 =tmp45 *tmp45 
    tmp47 =tl .broadcast_to (tmp46 ,[XBLOCK ,R0_BLOCK ])
    tmp49 =tl .where (r0_mask ,tmp47 ,0 )
    tmp50 =tl .sum (tmp49 ,1 )[:,None ]
    tmp51 =0.2 
    tmp52 =tmp37 *tmp51 
    tmp53 =tmp34 +tmp52 
    tmp54 =tmp34 -tmp53 
    tmp55 =tmp54 +tmp44 
    tmp56 =tmp55 *tmp55 
    tmp57 =tl .broadcast_to (tmp56 ,[XBLOCK ,R0_BLOCK ])
    tmp59 =tl .where (r0_mask ,tmp57 ,0 )
    tmp60 =tl .sum (tmp59 ,1 )[:,None ]
    tmp61 =libdevice .sqrt (tmp50 )
    tmp62 =tmp61 +tmp5 
    tmp63 =libdevice .sqrt (tmp60 )
    tmp64 =tmp62 -tmp63 
    tmp65 =triton_helpers .maximum (tmp64 ,tmp1 )
    tmp66 =tmp65 /tmp5 
    tl .debug_barrier ()
    tl .store (in_out_ptr1 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp66 ,None )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        4 +s2 
        4 +s1 
        16 +4 *s1 +4 *s2 +s1 *s2 
        buf0 =empty_strided_cuda ((1 ,s0 ,4 +s1 ,4 +s2 ),(16 *s0 +4 *s0 *s1 +4 *s0 *s2 +s0 *s1 *s2 ,16 +4 *s1 +4 *s2 +s1 *s2 ,4 +s2 ,1 ),torch .float32 )

        triton_poi_fused_constant_pad_nd_0_xnumel =16 *s0 +4 *s0 *s1 +4 *s0 *s2 +s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_constant_pad_nd_0 [grid (triton_poi_fused_constant_pad_nd_0_xnumel )](arg3_1 ,buf0 ,36 ,36 ,32 ,32 ,1296 ,3888 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        buf1 =empty_strided_cuda ((1 ,1 ,16 *s0 +4 *s0 *s1 +4 *s0 *s2 +s0 *s1 *s2 ),(16 *s0 +4 *s0 *s1 +4 *s0 *s2 +s0 *s1 *s2 ,16 *s0 +4 *s0 *s1 +4 *s0 *s2 +s0 *s1 *s2 ,1 ),torch .float32 )

        triton_poi_fused__adaptive_avg_pool2d_1_xnumel =16 *s0 +4 *s0 *s1 +4 *s0 *s2 +s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused__adaptive_avg_pool2d_1 [grid (triton_poi_fused__adaptive_avg_pool2d_1_xnumel )](buf0 ,buf1 ,36 ,36 ,32 ,32 ,3888 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf0 

        buf2 =torch .ops .aten ._adaptive_avg_pool2d .default (buf1 ,[1 ,10 ])
        del buf1 
        buf3 =buf2 
        del buf2 
        buf5 =empty_strided_cuda ((2 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[2 ],out =buf5 )
        buf4 =reinterpret_tensor (buf3 ,(1 ,10 ),(10 ,1 ),0 );del buf3 
        buf7 =empty_strided_cuda ((1 ,),(1 ,),torch .float32 )
        buf10 =reinterpret_tensor (buf7 ,(),(),0 );del buf7 

        get_raw_stream (0 )
        triton_per_fused_add_clamp_min_elu_mean_mul_norm_randn_like_sub_2 [grid (1 )](buf4 ,buf10 ,buf5 ,1 ,0 ,1 ,10 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        del buf4 
        del buf5 
    return (buf10 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =3 
    arg1_1 =32 
    arg2_1 =32 
    arg3_1 =rand_strided ((1 ,3 ,32 ,32 ),(3072 ,1024 ,32 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
