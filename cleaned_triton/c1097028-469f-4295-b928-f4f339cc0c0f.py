
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
def triton_poi_fused_rand_0 (in_ptr0 ,out_ptr0 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =x0 
    tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tl .store (out_ptr0 +(x0 ),tmp2 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused_add_clamp_min_fill_fractional_max_pool2d_mean_ne_replication_pad2d_sub_where_zeros_like_1 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,ks0 ,ks1 ,ks2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    _tmp60 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        r0_2 =r0_index //196 
        r0_1 =((r0_index //14 )%14 )
        r0_0 =(r0_index %14 )
        tmp0 =tl .load (in_ptr0 +(2 *r0_2 ),r0_mask ,eviction_policy ='evict_last',other =0.0 )
        tmp21 =tl .load (in_ptr0 +(1 +2 *r0_2 ),r0_mask ,eviction_policy ='evict_last',other =0.0 )
        tmp1 =(2 +ks0 )/13 
        tmp2 =tmp1 .to (tl .float32 )
        tmp3 =r0_1 
        tmp4 =tmp3 .to (tl .float32 )
        tmp5 =tmp4 +tmp0 
        tmp6 =tmp5 *tmp2 
        tmp7 =libdevice .floor (tmp6 )
        tmp8 =tmp0 *tmp2 
        tmp9 =libdevice .floor (tmp8 )
        tmp10 =tmp7 -tmp9 
        tmp11 =tmp10 .to (tl .int64 )
        tmp12 =tl .full ([1 ,1 ],13 ,tl .int64 )
        tmp13 =tmp4 <tmp12 
        tmp14 =2 +ks0 
        tmp15 =tl .where (tmp13 ,tmp11 ,tmp14 )
        tmp16 =4 +ks0 
        tmp17 =tmp15 +tmp16 
        tmp18 =tmp15 <0 
        tmp19 =tl .where (tmp18 ,tmp17 ,tmp15 )
        tl .device_assert (((0 <=tmp19 )&(tmp19 <4 +ks0 ))|~(r0_mask ),"index out of bounds: 0 <= tmp19 < 4 + ks0")
        tmp22 =(2 +ks1 )/13 
        tmp23 =tmp22 .to (tl .float32 )
        tmp24 =r0_0 
        tmp25 =tmp24 .to (tl .float32 )
        tmp26 =tmp25 +tmp21 
        tmp27 =tmp26 *tmp23 
        tmp28 =libdevice .floor (tmp27 )
        tmp29 =tmp21 *tmp23 
        tmp30 =libdevice .floor (tmp29 )
        tmp31 =tmp28 -tmp30 
        tmp32 =tmp31 .to (tl .int64 )
        tmp33 =tmp25 <tmp12 
        tmp34 =2 +ks1 
        tmp35 =tl .where (tmp33 ,tmp32 ,tmp34 )
        tmp36 =4 +ks1 
        tmp37 =tmp35 +tmp36 
        tmp38 =tmp35 <0 
        tmp39 =tl .where (tmp38 ,tmp37 ,tmp35 )
        tl .device_assert (((0 <=tmp39 )&(tmp39 <4 +ks1 ))|~(r0_mask ),"index out of bounds: 0 <= tmp39 < 4 + ks1")
        tmp41 =tl .load (in_ptr1 +(ks1 *(((-1 )+ks0 )*(((-1 )+ks0 )<=(((0 )*((0 )>=((-2 )+tmp19 ))+((-2 )+tmp19 )*(((-2 )+tmp19 )>(0 )))))+(((0 )*((0 )>=((-2 )+tmp19 ))+((-2 )+tmp19 )*(((-2 )+tmp19 )>(0 ))))*((((0 )*((0 )>=((-2 )+tmp19 ))+((-2 )+tmp19 )*(((-2 )+tmp19 )>(0 ))))<((-1 )+ks0 )))+ks0 *ks1 *r0_2 +(((-1 )+ks1 )*(((-1 )+ks1 )<=(((0 )*((0 )>=((-2 )+tmp39 ))+((-2 )+tmp39 )*(((-2 )+tmp39 )>(0 )))))+(((0 )*((0 )>=((-2 )+tmp39 ))+((-2 )+tmp39 )*(((-2 )+tmp39 )>(0 ))))*((((0 )*((0 )>=((-2 )+tmp39 ))+((-2 )+tmp39 )*(((-2 )+tmp39 )>(0 ))))<((-1 )+ks1 )))),r0_mask ,eviction_policy ='evict_last',other =0.0 )
        tmp42 =tl .load (in_ptr1 +(ks1 *(((-1 )+ks0 )*(((-1 )+ks0 )<=(((0 )*((0 )>=((-2 )+tmp19 ))+((-2 )+tmp19 )*(((-2 )+tmp19 )>(0 )))))+(((0 )*((0 )>=((-2 )+tmp19 ))+((-2 )+tmp19 )*(((-2 )+tmp19 )>(0 ))))*((((0 )*((0 )>=((-2 )+tmp19 ))+((-2 )+tmp19 )*(((-2 )+tmp19 )>(0 ))))<((-1 )+ks0 )))+ks0 *ks1 *r0_2 +(((-1 )+ks1 )*(((-1 )+ks1 )<=(((0 )*((0 )>=((-1 )+tmp39 ))+((-1 )+tmp39 )*(((-1 )+tmp39 )>(0 )))))+(((0 )*((0 )>=((-1 )+tmp39 ))+((-1 )+tmp39 )*(((-1 )+tmp39 )>(0 ))))*((((0 )*((0 )>=((-1 )+tmp39 ))+((-1 )+tmp39 )*(((-1 )+tmp39 )>(0 ))))<((-1 )+ks1 )))),r0_mask ,eviction_policy ='evict_last',other =0.0 )
        tmp43 =triton_helpers .maximum (tmp42 ,tmp41 )
        tmp44 =tl .load (in_ptr1 +(ks1 *(((-1 )+ks0 )*(((-1 )+ks0 )<=(((0 )*((0 )>=((-1 )+tmp19 ))+((-1 )+tmp19 )*(((-1 )+tmp19 )>(0 )))))+(((0 )*((0 )>=((-1 )+tmp19 ))+((-1 )+tmp19 )*(((-1 )+tmp19 )>(0 ))))*((((0 )*((0 )>=((-1 )+tmp19 ))+((-1 )+tmp19 )*(((-1 )+tmp19 )>(0 ))))<((-1 )+ks0 )))+ks0 *ks1 *r0_2 +(((-1 )+ks1 )*(((-1 )+ks1 )<=(((0 )*((0 )>=((-2 )+tmp39 ))+((-2 )+tmp39 )*(((-2 )+tmp39 )>(0 )))))+(((0 )*((0 )>=((-2 )+tmp39 ))+((-2 )+tmp39 )*(((-2 )+tmp39 )>(0 ))))*((((0 )*((0 )>=((-2 )+tmp39 ))+((-2 )+tmp39 )*(((-2 )+tmp39 )>(0 ))))<((-1 )+ks1 )))),r0_mask ,eviction_policy ='evict_last',other =0.0 )
        tmp45 =triton_helpers .maximum (tmp44 ,tmp43 )
        tmp46 =tl .load (in_ptr1 +(ks1 *(((-1 )+ks0 )*(((-1 )+ks0 )<=(((0 )*((0 )>=((-1 )+tmp19 ))+((-1 )+tmp19 )*(((-1 )+tmp19 )>(0 )))))+(((0 )*((0 )>=((-1 )+tmp19 ))+((-1 )+tmp19 )*(((-1 )+tmp19 )>(0 ))))*((((0 )*((0 )>=((-1 )+tmp19 ))+((-1 )+tmp19 )*(((-1 )+tmp19 )>(0 ))))<((-1 )+ks0 )))+ks0 *ks1 *r0_2 +(((-1 )+ks1 )*(((-1 )+ks1 )<=(((0 )*((0 )>=((-1 )+tmp39 ))+((-1 )+tmp39 )*(((-1 )+tmp39 )>(0 )))))+(((0 )*((0 )>=((-1 )+tmp39 ))+((-1 )+tmp39 )*(((-1 )+tmp39 )>(0 ))))*((((0 )*((0 )>=((-1 )+tmp39 ))+((-1 )+tmp39 )*(((-1 )+tmp39 )>(0 ))))<((-1 )+ks1 )))),r0_mask ,eviction_policy ='evict_last',other =0.0 )
        tmp47 =triton_helpers .maximum (tmp46 ,tmp45 )
        tmp48 =libdevice .tanh (tmp47 )
        tmp49 =tmp47 -tmp48 
        tmp50 =1.0 
        tmp51 =tmp50 -tmp49 
        tmp52 =0.0 
        tmp53 =triton_helpers .maximum (tmp51 ,tmp52 )
        tmp54 =tl .full ([1 ,1 ],False ,tl .int1 )
        tmp55 =tl .where (tmp54 ,tmp53 ,tmp52 )
        tmp56 =tl .full ([1 ,1 ],True ,tl .int1 )
        tmp57 =tl .where (tmp56 ,tmp49 ,tmp52 )
        tmp58 =tmp55 +tmp57 
        tmp59 =tl .broadcast_to (tmp58 ,[XBLOCK ,R0_BLOCK ])
        tmp61 =_tmp60 +tmp59 
        _tmp60 =tl .where (r0_mask ,tmp61 ,_tmp60 )
    tmp60 =tl .sum (_tmp60 ,1 )[:,None ]
    tmp62 =196 *ks2 
    tmp63 =tmp62 .to (tl .float32 )
    tmp64 =tmp60 /tmp63 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp64 ,None )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[1 ],out =buf0 )
        buf1 =empty_strided_cuda ((1 ,s0 ,2 ),(2 *s0 ,2 ,1 ),torch .float32 )

        triton_poi_fused_rand_0_xnumel =2 *s0 
        get_raw_stream (0 )
        triton_poi_fused_rand_0 [grid (triton_poi_fused_rand_0_xnumel )](buf0 ,buf1 ,0 ,6 ,XBLOCK =8 ,num_warps =1 ,num_stages =1 )
        del buf0 
        buf3 =empty_strided_cuda ((),(),torch .float32 )
        buf4 =buf3 ;del buf3 

        196 *s0 
        get_raw_stream (0 )
        triton_red_fused_add_clamp_min_fill_fractional_max_pool2d_mean_ne_replication_pad2d_sub_where_zeros_like_1 [grid (1 )](buf4 ,buf1 ,arg3_1 ,28 ,28 ,3 ,1 ,588 ,XBLOCK =1 ,R0_BLOCK =1024 ,num_warps =8 ,num_stages =1 )
        del arg3_1 
        del buf1 
    return (buf4 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =3 
    arg1_1 =28 
    arg2_1 =28 
    arg3_1 =rand_strided ((1 ,3 ,28 ,28 ),(2352 ,784 ,28 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
