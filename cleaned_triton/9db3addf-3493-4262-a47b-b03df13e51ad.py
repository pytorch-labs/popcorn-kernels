
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
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_addmm_relu_0 (in_out_ptr0 ,in_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =64 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_out_ptr0 +(x0 ),xmask )
    tmp1 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp2 =tmp0 +tmp1 
    tmp3 =tl .full ([1 ],0 ,tl .int32 )
    tmp4 =triton_helpers .maximum (tmp3 ,tmp2 )
    tl .store (in_out_ptr0 +(x0 ),tmp4 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_addmm_relu_1 (in_out_ptr0 ,in_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =32 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_out_ptr0 +(x0 ),xmask )
    tmp1 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp2 =tmp0 +tmp1 
    tmp3 =tl .full ([1 ],0 ,tl .int32 )
    tmp4 =triton_helpers .maximum (tmp3 ,tmp2 )
    tl .store (in_out_ptr0 +(x0 ),tmp4 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_addmm_relu_2 (in_out_ptr0 ,in_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =16 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_out_ptr0 +(x0 ),xmask )
    tmp1 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp2 =tmp0 +tmp1 
    tmp3 =tl .full ([1 ],0 ,tl .int32 )
    tmp4 =triton_helpers .maximum (tmp3 ,tmp2 )
    tl .store (in_out_ptr0 +(x0 ),tmp4 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused__log_softmax__softmax_div_mul_randn_like_relu_sub_sum_xlogy_3 (in_out_ptr0 ,in_out_ptr1 ,in_ptr0 ,in_ptr1 ,out_ptr0 ,out_ptr1 ,out_ptr2 ,out_ptr3 ,load_seed_offset ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    R0_BLOCK :tl .constexpr =8 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_0 =r0_index 
    tmp0 =tl .load (in_ptr0 +(r0_0 ),None )
    tmp1 =tl .full ([1 ,1 ],0 ,tl .int32 )
    tmp2 =triton_helpers .maximum (tmp1 ,tmp0 )
    tmp3 =tl .broadcast_to (tmp2 ,[XBLOCK ,R0_BLOCK ])
    tmp5 =triton_helpers .max2 (tmp3 ,1 )[:,None ]
    tmp6 =tmp2 -tmp5 
    tmp7 =tl_math .exp (tmp6 )
    tmp8 =tl .broadcast_to (tmp7 ,[XBLOCK ,R0_BLOCK ])
    tmp10 =tl .sum (tmp8 ,1 )[:,None ]
    tmp11 =tl_math .log (tmp10 )
    tmp12 =tl .load (in_ptr1 +load_seed_offset )
    tmp13 =r0_0 
    tmp14 =tl .randn (tmp12 ,(tmp13 ).to (tl .uint32 ))
    tmp15 =tl .broadcast_to (tmp14 ,[XBLOCK ,R0_BLOCK ])
    tmp17 =triton_helpers .max2 (tmp15 ,1 )[:,None ]
    tmp18 =tmp14 -tmp17 
    tmp19 =tl_math .exp (tmp18 )
    tmp20 =tl .broadcast_to (tmp19 ,[XBLOCK ,R0_BLOCK ])
    tmp22 =tl .sum (tmp20 ,1 )[:,None ]
    tmp23 =tmp19 /tmp22 
    tmp24 =libdevice .isnan (tmp23 ).to (tl .int1 )
    tmp25 =0.0 
    tmp26 =tmp23 ==tmp25 
    tmp27 =tl_math .log (tmp23 )
    tmp28 =tmp23 *tmp27 
    tmp29 =tl .where (tmp26 ,tmp25 ,tmp28 )
    tmp30 =float ("nan")
    tmp31 =tl .where (tmp24 ,tmp30 ,tmp29 )
    tmp32 =tmp6 -tmp11 
    tmp33 =tmp23 *tmp32 
    tmp34 =tmp31 -tmp33 
    tmp35 =tl .broadcast_to (tmp34 ,[XBLOCK ,R0_BLOCK ])
    tmp37 =tl .sum (tmp35 ,1 )[:,None ]
    tmp38 =1.0 
    tmp39 =tmp37 *tmp38 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp11 ,None )
    tl .store (out_ptr1 +(tl .broadcast_to (r0_0 ,[XBLOCK ,R0_BLOCK ])),tmp14 ,None )
    tl .debug_barrier ()
    tl .store (in_out_ptr1 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp39 ,None )
    tl .store (out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp5 ,None )
    tl .store (out_ptr2 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp17 ,None )
    tl .store (out_ptr3 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp22 ,None )

def call (args ):
    primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ,primals_8 ,primals_9 =args 
    args .clear ()
    assert_size_stride (primals_1 ,(1 ,128 ),(128 ,1 ))
    assert_size_stride (primals_2 ,(64 ,128 ),(128 ,1 ))
    assert_size_stride (primals_3 ,(64 ,),(1 ,))
    assert_size_stride (primals_4 ,(32 ,64 ),(64 ,1 ))
    assert_size_stride (primals_5 ,(32 ,),(1 ,))
    assert_size_stride (primals_6 ,(16 ,32 ),(32 ,1 ))
    assert_size_stride (primals_7 ,(16 ,),(1 ,))
    assert_size_stride (primals_8 ,(8 ,16 ),(16 ,1 ))
    assert_size_stride (primals_9 ,(8 ,),(1 ,))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,64 ),(64 ,1 ),torch .float32 )

        extern_kernels .mm (primals_1 ,reinterpret_tensor (primals_2 ,(128 ,64 ),(1 ,128 ),0 ),out =buf0 )
        del primals_2 
        buf1 =buf0 ;del buf0 

        get_raw_stream (0 )
        triton_poi_fused_addmm_relu_0 [grid (64 )](buf1 ,primals_3 ,64 ,XBLOCK =64 ,num_warps =1 ,num_stages =1 )
        del primals_3 
        buf2 =empty_strided_cuda ((1 ,32 ),(32 ,1 ),torch .float32 )

        extern_kernels .mm (buf1 ,reinterpret_tensor (primals_4 ,(64 ,32 ),(1 ,64 ),0 ),out =buf2 )
        buf3 =buf2 ;del buf2 

        get_raw_stream (0 )
        triton_poi_fused_addmm_relu_1 [grid (32 )](buf3 ,primals_5 ,32 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        del primals_5 
        buf4 =empty_strided_cuda ((1 ,16 ),(16 ,1 ),torch .float32 )

        extern_kernels .mm (buf3 ,reinterpret_tensor (primals_6 ,(32 ,16 ),(1 ,32 ),0 ),out =buf4 )
        buf5 =buf4 ;del buf4 

        get_raw_stream (0 )
        triton_poi_fused_addmm_relu_2 [grid (16 )](buf5 ,primals_7 ,16 ,XBLOCK =16 ,num_warps =1 ,num_stages =1 )
        del primals_7 
        buf6 =empty_strided_cuda ((1 ,8 ),(8 ,1 ),torch .float32 )

        extern_kernels .addmm (primals_9 ,buf5 ,reinterpret_tensor (primals_8 ,(16 ,8 ),(1 ,16 ),0 ),alpha =1 ,beta =1 ,out =buf6 )
        del primals_9 
        buf7 =empty_strided_cuda ((1 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[1 ],out =buf7 )
        buf11 =empty_strided_cuda ((1 ,1 ),(1 ,1 ),torch .float32 )
        buf12 =empty_strided_cuda ((1 ,1 ),(1 ,1 ),torch .float32 )
        buf13 =buf12 ;del buf12 
        buf8 =empty_strided_cuda ((1 ,8 ),(8 ,1 ),torch .float32 )
        buf9 =empty_strided_cuda ((1 ,1 ),(1 ,1 ),torch .float32 )
        buf10 =empty_strided_cuda ((1 ,1 ),(1 ,1 ),torch .float32 )
        buf14 =empty_strided_cuda ((),(),torch .float32 )
        buf15 =buf14 ;del buf14 

        get_raw_stream (0 )
        triton_per_fused__log_softmax__softmax_div_mul_randn_like_relu_sub_sum_xlogy_3 [grid (1 )](buf13 ,buf15 ,buf6 ,buf7 ,buf11 ,buf8 ,buf9 ,buf10 ,0 ,1 ,8 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        del buf7 
    return (buf15 ,primals_1 ,buf1 ,buf3 ,buf5 ,buf6 ,buf8 ,buf9 ,buf10 ,buf11 ,buf13 ,primals_8 ,primals_6 ,primals_4 ,)

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
    primals_8 =rand_strided ((8 ,16 ),(16 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_9 =rand_strided ((8 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ,primals_8 ,primals_9 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
