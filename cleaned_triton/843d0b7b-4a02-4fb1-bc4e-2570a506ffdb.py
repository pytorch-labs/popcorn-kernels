
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
def triton_poi_fused_pow_0 (in_ptr0 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp1 =tmp0 *tmp0 
    tl .store (out_ptr0 +(x0 ),tmp1 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_abs_mul_pow_relu_sign_1 (in_out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_out_ptr0 +(x0 ),xmask )
    tmp1 =tl .full ([1 ],0 ,tl .int32 )
    tmp2 =tmp1 <tmp0 
    tmp3 =tmp2 .to (tl .int8 )
    tmp4 =tmp0 <tmp1 
    tmp5 =tmp4 .to (tl .int8 )
    tmp6 =tmp3 -tmp5 
    tmp7 =tmp6 .to (tmp0 .dtype )
    tmp8 =tl_math .abs (tmp0 )
    tmp9 =triton_helpers .maximum (tmp1 ,tmp8 )
    tmp10 =tmp7 *tmp9 
    tmp11 =27.0 
    tmp12 =tmp10 *tmp11 
    tmp13 =libdevice .sqrt (tmp12 )
    tl .store (in_out_ptr0 +(x0 ),tmp13 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__adaptive_avg_pool2d_2 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +((triton_helpers .div_floor_integer (x0 ,1 +(triton_helpers .div_floor_integer ((-3 )+ks0 ,2 ))*(triton_helpers .div_floor_integer ((-3 )+ks1 ,2 ))+(triton_helpers .div_floor_integer ((-3 )+ks0 ,2 ))+(triton_helpers .div_floor_integer ((-3 )+ks1 ,2 ))))*(triton_helpers .div_floor_integer ((-3 )+ks0 ,2 ))+(triton_helpers .div_floor_integer (x0 ,1 +(triton_helpers .div_floor_integer ((-3 )+ks0 ,2 ))*(triton_helpers .div_floor_integer ((-3 )+ks1 ,2 ))+(triton_helpers .div_floor_integer ((-3 )+ks0 ,2 ))+(triton_helpers .div_floor_integer ((-3 )+ks1 ,2 ))))*(triton_helpers .div_floor_integer ((-3 )+ks1 ,2 ))+(triton_helpers .div_floor_integer ((-3 )+ks1 ,2 ))*(((x0 //(1 +(triton_helpers .div_floor_integer ((-3 )+ks1 ,2 ))))%(1 +(triton_helpers .div_floor_integer ((-3 )+ks0 ,2 )))))+(triton_helpers .div_floor_integer (x0 ,1 +(triton_helpers .div_floor_integer ((-3 )+ks0 ,2 ))*(triton_helpers .div_floor_integer ((-3 )+ks1 ,2 ))+(triton_helpers .div_floor_integer ((-3 )+ks0 ,2 ))+(triton_helpers .div_floor_integer ((-3 )+ks1 ,2 ))))*(triton_helpers .div_floor_integer ((-3 )+ks0 ,2 ))*(triton_helpers .div_floor_integer ((-3 )+ks1 ,2 ))+(triton_helpers .div_floor_integer (x0 ,1 +(triton_helpers .div_floor_integer ((-3 )+ks0 ,2 ))*(triton_helpers .div_floor_integer ((-3 )+ks1 ,2 ))+(triton_helpers .div_floor_integer ((-3 )+ks0 ,2 ))+(triton_helpers .div_floor_integer ((-3 )+ks1 ,2 ))))+((x0 %(1 +(triton_helpers .div_floor_integer ((-3 )+ks1 ,2 )))))+(((x0 //(1 +(triton_helpers .div_floor_integer ((-3 )+ks1 ,2 ))))%(1 +(triton_helpers .div_floor_integer ((-3 )+ks0 ,2 )))))),xmask ,eviction_policy ='evict_last')
    tl .store (out_ptr0 +(x0 ),tmp0 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_addmm_relu_3 (in_out_ptr0 ,in_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =50 
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
def triton_per_fused_huber_loss_4 (in_out_ptr0 ,in_ptr0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    r0_numel =10 
    R0_BLOCK :tl .constexpr =16 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    r0_mask =r0_index <r0_numel 
    r0_0 =r0_index 
    tmp0 =tl .load (in_ptr0 +(r0_0 ),r0_mask ,other =0.0 )
    tmp1 =tl_math .abs (tmp0 )
    tmp2 =1.0 
    tmp3 =tmp1 <tmp2 
    tmp4 =0.5 
    tmp5 =tmp1 *tmp4 
    tmp6 =tmp5 *tmp1 
    tmp7 =tmp1 -tmp4 
    tmp8 =tmp7 *tmp2 
    tmp9 =tl .where (tmp3 ,tmp6 ,tmp8 )
    tmp10 =tl .broadcast_to (tmp9 ,[XBLOCK ,R0_BLOCK ])
    tmp12 =tl .where (r0_mask ,tmp10 ,0 )
    tmp13 =tl .sum (tmp12 ,1 )[:,None ]
    tmp14 =10.0 
    tmp15 =tmp13 /tmp14 
    tl .debug_barrier ()
    tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp15 ,None )

def call (args ):
    primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ,primals_8 =args 
    args .clear ()
    s0 =primals_1 
    s1 =primals_2 
    s2 =primals_3 
    assert_size_stride (primals_4 ,(1 ,1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    assert_size_stride (primals_5 ,(50 ,10 ),(10 ,1 ))
    assert_size_stride (primals_6 ,(50 ,),(1 ,))
    assert_size_stride (primals_7 ,(10 ,50 ),(50 ,1 ))
    assert_size_stride (primals_8 ,(10 ,),(1 ,))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ),torch .float32 )

        triton_poi_fused_pow_0_xnumel =s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_pow_0 [grid (triton_poi_fused_pow_0_xnumel )](primals_4 ,buf0 ,262144 ,XBLOCK =1024 ,num_warps =4 ,num_stages =1 )
        del primals_4 

        buf1 =torch .ops .aten .avg_pool3d .default (buf0 ,[3 ,3 ,3 ],[2 ,2 ,2 ],[0 ,0 ,0 ],False ,True ,None )
        del buf0 
        buf2 =buf1 
        del buf1 
        buf3 =reinterpret_tensor (buf2 ,(1 ,1 ,1 +(((-3 )+s0 )//2 ),1 +(((-3 )+s1 )//2 ),1 +(((-3 )+s2 )//2 )),(1 +(((-3 )+s0 )//2 )*(((-3 )+s1 )//2 )+(((-3 )+s0 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s1 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s0 )//2 )*(((-3 )+s1 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s0 )//2 )+(((-3 )+s1 )//2 )+(((-3 )+s2 )//2 ),1 ,1 +(((-3 )+s1 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s1 )//2 )+(((-3 )+s2 )//2 ),1 +(((-3 )+s2 )//2 ),1 ),0 );del buf2 

        triton_poi_fused_abs_mul_pow_relu_sign_1_xnumel =1 +(((-3 )+s0 )//2 )*(((-3 )+s1 )//2 )+(((-3 )+s0 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s1 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s0 )//2 )*(((-3 )+s1 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s0 )//2 )+(((-3 )+s1 )//2 )+(((-3 )+s2 )//2 )
        get_raw_stream (0 )
        triton_poi_fused_abs_mul_pow_relu_sign_1 [grid (triton_poi_fused_abs_mul_pow_relu_sign_1_xnumel )](buf3 ,29791 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        buf4 =empty_strided_cuda ((1 ,1 ,1 ,1 +(((-3 )+s0 )//2 )*(((-3 )+s1 )//2 )+(((-3 )+s0 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s1 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s0 )//2 )*(((-3 )+s1 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s0 )//2 )+(((-3 )+s1 )//2 )+(((-3 )+s2 )//2 )),(1 +(((-3 )+s0 )//2 )*(((-3 )+s1 )//2 )+(((-3 )+s0 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s1 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s0 )//2 )*(((-3 )+s1 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s0 )//2 )+(((-3 )+s1 )//2 )+(((-3 )+s2 )//2 ),1 +(((-3 )+s0 )//2 )*(((-3 )+s1 )//2 )+(((-3 )+s0 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s1 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s0 )//2 )*(((-3 )+s1 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s0 )//2 )+(((-3 )+s1 )//2 )+(((-3 )+s2 )//2 ),1 +(((-3 )+s0 )//2 )*(((-3 )+s1 )//2 )+(((-3 )+s0 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s1 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s0 )//2 )*(((-3 )+s1 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s0 )//2 )+(((-3 )+s1 )//2 )+(((-3 )+s2 )//2 ),1 ),torch .float32 )

        triton_poi_fused__adaptive_avg_pool2d_2_xnumel =1 +(((-3 )+s0 )//2 )*(((-3 )+s1 )//2 )+(((-3 )+s0 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s1 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s0 )//2 )*(((-3 )+s1 )//2 )*(((-3 )+s2 )//2 )+(((-3 )+s0 )//2 )+(((-3 )+s1 )//2 )+(((-3 )+s2 )//2 )
        get_raw_stream (0 )
        triton_poi_fused__adaptive_avg_pool2d_2 [grid (triton_poi_fused__adaptive_avg_pool2d_2_xnumel )](buf3 ,buf4 ,64 ,64 ,29791 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf3 

        buf5 =torch .ops .aten ._adaptive_avg_pool2d .default (buf4 ,[1 ,10 ])
        del buf4 
        buf6 =buf5 
        del buf5 
        buf7 =empty_strided_cuda ((1 ,50 ),(50 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf6 ,(1 ,10 ),(10 ,1 ),0 ),reinterpret_tensor (primals_5 ,(10 ,50 ),(1 ,10 ),0 ),out =buf7 )
        del primals_5 
        buf8 =buf7 ;del buf7 

        get_raw_stream (0 )
        triton_poi_fused_addmm_relu_3 [grid (50 )](buf8 ,primals_6 ,50 ,XBLOCK =64 ,num_warps =1 ,num_stages =1 )
        del primals_6 
        buf9 =empty_strided_cuda ((1 ,10 ),(10 ,1 ),torch .float32 )

        extern_kernels .addmm (primals_8 ,buf8 ,reinterpret_tensor (primals_7 ,(50 ,10 ),(1 ,50 ),0 ),alpha =1 ,beta =1 ,out =buf9 )
        del primals_8 
        buf10 =empty_strided_cuda ((),(),torch .float32 )
        buf11 =buf10 ;del buf10 

        get_raw_stream (0 )
        triton_per_fused_huber_loss_4 [grid (1 )](buf11 ,buf9 ,1 ,10 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
    return (buf11 ,reinterpret_tensor (buf6 ,(1 ,10 ),(10 ,1 ),0 ),buf8 ,buf9 ,primals_7 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    primals_1 =64 
    primals_2 =64 
    primals_3 =64 
    primals_4 =rand_strided ((1 ,1 ,64 ,64 ,64 ),(262144 ,262144 ,4096 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_5 =rand_strided ((50 ,10 ),(10 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_6 =rand_strided ((50 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_7 =rand_strided ((10 ,50 ),(50 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_8 =rand_strided ((10 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ,primals_8 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
