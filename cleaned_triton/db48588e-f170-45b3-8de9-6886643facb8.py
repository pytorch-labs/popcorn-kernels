
import torch 
from torch ._inductor .select_algorithm import extern_kernels 
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
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_hardsigmoid_view_0 (in_ptr0 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =128 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp1 =3.0 
    tmp2 =tmp0 +tmp1 
    tmp3 =0.0 
    tmp4 =triton_helpers .maximum (tmp2 ,tmp3 )
    tmp5 =6.0 
    tmp6 =triton_helpers .minimum (tmp4 ,tmp5 )
    tmp7 =0.16666666666666666 
    tmp8 =tmp6 *tmp7 
    tl .store (out_ptr0 +(x0 ),tmp8 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused_add_arange_clamp_min_gather_ge_mean_ne_randint_rsub_scalar_tensor_where_1 (in_out_ptr0 ,in_out_ptr1 ,in_ptr0 ,out_ptr0 ,out_ptr1 ,load_seed_offset ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    R0_BLOCK :tl .constexpr =16 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_0 =r0_index 
    tmp15 =tl .load (in_ptr0 +(r0_0 ),None )
    tmp0 =tl .load (in_out_ptr0 +load_seed_offset )
    tmp1 =tl .full ([1 ,1 ],0 ,tl .int32 )
    tmp2 =tl .full ([1 ,1 ],0 ,tl .int64 )
    tmp3 =tl .full ([1 ,1 ],16 ,tl .int64 )
    tmp4 =triton_helpers .randint64 (tmp0 ,(tmp1 ).to (tl .uint32 ),tmp2 ,tmp3 )
    tmp5 =r0_0 
    tmp6 =tmp5 !=tmp4 
    tmp7 =tl .full ([XBLOCK ,R0_BLOCK ],16 ,tl .int32 )
    tmp8 =tmp4 +tmp7 
    tmp9 =tmp4 <0 
    tmp10 =tl .where (tmp9 ,tmp8 ,tmp4 )
    tl .device_assert ((0 <=tmp10 )&(tmp10 <16 ),"index out of bounds: 0 <= tmp10 < 16")
    tmp12 =tl .load (in_ptr0 +(tmp10 ),None ,eviction_policy ='evict_last')
    tmp13 =1.0 
    tmp14 =tmp13 -tmp12 
    tmp16 =tmp14 +tmp15 
    tmp17 =0.0 
    tmp18 =triton_helpers .maximum (tmp16 ,tmp17 )
    tmp19 =tl .where (tmp6 ,tmp18 ,tmp17 )
    tmp20 =tl .broadcast_to (tmp19 ,[XBLOCK ,R0_BLOCK ])
    tmp22 =tl .sum (tmp20 ,1 )[:,None ]
    tmp23 =tmp16 >=tmp17 
    tmp24 =16.0 
    tmp25 =tmp22 /tmp24 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp4 ,None )
    tl .store (out_ptr0 +(tl .broadcast_to (r0_0 ,[XBLOCK ,R0_BLOCK ])),tmp6 ,None )
    tl .store (out_ptr1 +(tl .broadcast_to (r0_0 ,[XBLOCK ,R0_BLOCK ])),tmp23 ,None )
    tl .debug_barrier ()
    tl .store (in_out_ptr1 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp25 ,None )

def call (args ):
    primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 =args 
    args .clear ()
    assert_size_stride (primals_1 ,(1 ,128 ),(128 ,1 ))
    assert_size_stride (primals_2 ,(64 ,128 ),(128 ,1 ))
    assert_size_stride (primals_3 ,(64 ,),(1 ,))
    assert_size_stride (primals_4 ,(32 ,64 ),(64 ,1 ))
    assert_size_stride (primals_5 ,(32 ,),(1 ,))
    assert_size_stride (primals_6 ,(16 ,32 ),(32 ,1 ))
    assert_size_stride (primals_7 ,(16 ,),(1 ,))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,128 ),(128 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_hardsigmoid_view_0 [grid (128 )](primals_1 ,buf0 ,128 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del primals_1 
        buf1 =empty_strided_cuda ((1 ,64 ),(64 ,1 ),torch .float32 )

        extern_kernels .addmm (primals_3 ,buf0 ,reinterpret_tensor (primals_2 ,(128 ,64 ),(1 ,128 ),0 ),alpha =1 ,beta =1 ,out =buf1 )
        del primals_2 
        del primals_3 
        buf2 =empty_strided_cuda ((1 ,32 ),(32 ,1 ),torch .float32 )

        extern_kernels .addmm (primals_5 ,buf1 ,reinterpret_tensor (primals_4 ,(64 ,32 ),(1 ,64 ),0 ),alpha =1 ,beta =1 ,out =buf2 )
        del primals_5 
        buf3 =empty_strided_cuda ((1 ,16 ),(16 ,1 ),torch .float32 )

        extern_kernels .addmm (primals_7 ,buf2 ,reinterpret_tensor (primals_6 ,(32 ,16 ),(1 ,32 ),0 ),alpha =1 ,beta =1 ,out =buf3 )
        del primals_7 
        buf4 =empty_strided_cuda ((1 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[1 ],out =buf4 )
        buf5 =buf4 ;del buf4 
        buf6 =empty_strided_cuda ((1 ,16 ),(16 ,1 ),torch .bool )
        buf7 =empty_strided_cuda ((),(),torch .float32 )
        buf8 =empty_strided_cuda ((1 ,16 ),(16 ,1 ),torch .bool )
        buf9 =buf7 ;del buf7 

        get_raw_stream (0 )
        triton_per_fused_add_arange_clamp_min_gather_ge_mean_ne_randint_rsub_scalar_tensor_where_1 [grid (1 )](buf5 ,buf9 ,buf3 ,buf6 ,buf8 ,0 ,1 ,16 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
    return (buf3 ,buf9 ,buf0 ,buf1 ,buf2 ,reinterpret_tensor (buf5 ,(1 ,1 ),(1 ,1 ),0 ),buf6 ,buf8 ,primals_6 ,primals_4 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    primals_1 =rand_strided ((1 ,128 ),(128 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_2 =rand_strided ((64 ,128 ),(128 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_3 =rand_strided ((64 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_4 =rand_strided ((32 ,64 ),(64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_5 =rand_strided ((32 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_6 =rand_strided ((16 ,32 ),(32 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_7 =rand_strided ((16 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
