
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
def triton_poi_fused_gelu_max_pool2d_with_indices_0 (in_out_ptr0 ,in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%ks1 )
    x2 =xindex //ks2 
    x3 =xindex 
    tmp0 =tl .load (in_ptr0 +(2 *x0 +2 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp9 =tl .load (in_ptr0 +(1 +2 *x0 +2 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp20 =tl .load (in_ptr0 +(ks4 +2 *x0 +2 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp30 =tl .load (in_ptr0 +(1 +ks4 +2 *x0 +2 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp1 =0.5 
    tmp2 =tmp0 *tmp1 
    tmp3 =0.7071067811865476 
    tmp4 =tmp0 *tmp3 
    tmp5 =libdevice .erf (tmp4 )
    tmp6 =1.0 
    tmp7 =tmp5 +tmp6 
    tmp8 =tmp2 *tmp7 
    tmp10 =tmp9 *tmp1 
    tmp11 =tmp9 *tmp3 
    tmp12 =libdevice .erf (tmp11 )
    tmp13 =tmp12 +tmp6 
    tmp14 =tmp10 *tmp13 
    tmp15 =tmp14 >tmp8 
    tmp16 =tl .full ([1 ],1 ,tl .int8 )
    tmp17 =tl .full ([1 ],0 ,tl .int8 )
    tmp18 =tl .where (tmp15 ,tmp16 ,tmp17 )
    tmp19 =triton_helpers .maximum (tmp14 ,tmp8 )
    tmp21 =tmp20 *tmp1 
    tmp22 =tmp20 *tmp3 
    tmp23 =libdevice .erf (tmp22 )
    tmp24 =tmp23 +tmp6 
    tmp25 =tmp21 *tmp24 
    tmp26 =tmp25 >tmp19 
    tmp27 =tl .full ([1 ],2 ,tl .int8 )
    tmp28 =tl .where (tmp26 ,tmp27 ,tmp18 )
    tmp29 =triton_helpers .maximum (tmp25 ,tmp19 )
    tmp31 =tmp30 *tmp1 
    tmp32 =tmp30 *tmp3 
    tmp33 =libdevice .erf (tmp32 )
    tmp34 =tmp33 +tmp6 
    tmp35 =tmp31 *tmp34 
    tmp36 =tmp35 >tmp29 
    tmp37 =tl .full ([1 ],3 ,tl .int8 )
    tmp38 =tl .where (tmp36 ,tmp37 ,tmp28 )
    tmp39 =triton_helpers .maximum (tmp35 ,tmp29 )
    tmp40 =tmp39 *tmp1 
    tmp41 =tmp39 *tmp3 
    tmp42 =libdevice .erf (tmp41 )
    tmp43 =tmp42 +tmp6 
    tmp44 =tmp40 *tmp43 
    tl .store (out_ptr0 +(x3 ),tmp38 ,xmask )
    tl .store (in_out_ptr0 +(x3 ),tmp44 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_max_unpool2d_1 (out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =0.0 
    tl .store (out_ptr0 +(x0 ),tmp0 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_gelu_max_pool2d_with_indices_max_unpool2d_2 (in_ptr0 ,out_ptr1 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,ks6 ,ks7 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%ks1 )
    x2 =xindex //ks2 
    x3 =xindex 
    tmp0 =tl .load (in_ptr0 +(2 *x0 +2 *ks3 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp1 =tl .load (in_ptr0 +(1 +2 *x0 +2 *ks3 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp7 =tl .load (in_ptr0 +(ks3 +2 *x0 +2 *ks3 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp12 =tl .load (in_ptr0 +(1 +ks3 +2 *x0 +2 *ks3 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp35 =tl .load (in_ptr0 +(2 *((x3 %ks0 ))+2 *ks3 *(((x3 //ks0 )%ks1 ))+ks3 *ks4 *(x3 //ks2 )),xmask ,eviction_policy ='evict_last')
    tmp36 =tl .load (in_ptr0 +(1 +2 *((x3 %ks0 ))+2 *ks3 *(((x3 //ks0 )%ks1 ))+ks3 *ks4 *(x3 //ks2 )),xmask ,eviction_policy ='evict_last')
    tmp38 =tl .load (in_ptr0 +(ks3 +2 *((x3 %ks0 ))+2 *ks3 *(((x3 //ks0 )%ks1 ))+ks3 *ks4 *(x3 //ks2 )),xmask ,eviction_policy ='evict_last')
    tmp40 =tl .load (in_ptr0 +(1 +ks3 +2 *((x3 %ks0 ))+2 *ks3 *(((x3 //ks0 )%ks1 ))+ks3 *ks4 *(x3 //ks2 )),xmask ,eviction_policy ='evict_last')
    tmp2 =tmp1 >tmp0 
    tmp3 =tl .full ([1 ],1 ,tl .int8 )
    tmp4 =tl .full ([1 ],0 ,tl .int8 )
    tmp5 =tl .where (tmp2 ,tmp3 ,tmp4 )
    tmp6 =triton_helpers .maximum (tmp1 ,tmp0 )
    tmp8 =tmp7 >tmp6 
    tmp9 =tl .full ([1 ],2 ,tl .int8 )
    tmp10 =tl .where (tmp8 ,tmp9 ,tmp5 )
    tmp11 =triton_helpers .maximum (tmp7 ,tmp6 )
    tmp13 =tmp12 >tmp11 
    tmp14 =tl .full ([1 ],3 ,tl .int8 )
    tmp15 =tl .where (tmp13 ,tmp14 ,tmp10 )
    triton_helpers .maximum (tmp12 ,tmp11 )
    tmp17 =tl .full ([1 ],2 ,tl .int32 )
    tmp18 =tl .where ((tmp15 <0 )!=(tmp17 <0 ),tl .where (tmp15 %tmp17 !=0 ,tmp15 //tmp17 -1 ,tmp15 //tmp17 ),tmp15 //tmp17 )
    tmp19 =tmp18 *tmp17 
    tmp20 =tmp15 -tmp19 
    tmp21 =2 *x1 
    tmp22 =tmp21 +tmp18 
    tmp23 =2 *x0 
    tmp24 =tmp23 +tmp20 
    tmp25 =ks3 
    tmp26 =tmp22 *tmp25 
    tmp27 =tmp26 +tmp24 
    tmp28 =4 *ks0 *ks1 *x2 
    tmp29 =tmp27 +tmp28 
    tmp30 =4 *ks0 *ks1 *ks5 
    tmp31 =tmp29 +tmp30 
    tmp32 =tmp29 <0 
    tmp33 =tl .where (tmp32 ,tmp31 ,tmp29 )
    tl .device_assert (((0 <=tmp33 )&(tmp33 <4 *ks5 *(ks6 //4 )*(ks7 //4 )))|~(xmask ),"index out of bounds: 0 <= tmp33 < 4*ks5*(ks6 // 4)*(ks7 // 4)")
    tmp37 =triton_helpers .maximum (tmp36 ,tmp35 )
    tmp39 =triton_helpers .maximum (tmp38 ,tmp37 )
    tmp41 =triton_helpers .maximum (tmp40 ,tmp39 )
    tl .store (out_ptr1 +(tl .broadcast_to ((tmp33 %(4 *ks0 *ks1 *ks5 )),[XBLOCK ])),tmp41 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_max_unpool2d_3 (out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =0.0 
    tl .store (out_ptr0 +(x0 ),tmp0 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_max_unpool2d_4 (in_ptr0 ,in_ptr1 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,ks6 ,ks7 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(x0 ),xmask )
    tmp19 =tl .load (in_ptr1 +(2 *ks4 *((((2 *ks4 *(((x0 //(2 *ks4 ))%(2 *ks5 )))+((x0 %(2 *ks4 ))))//(2 *ks4 ))%(2 *ks5 )))+4 *ks4 *ks5 *((((2 *ks4 *(((x0 //(2 *ks4 ))%(2 *ks5 )))+4 *ks4 *ks5 *(((x0 //(4 *ks4 *ks5 ))%ks6 ))+((x0 %(2 *ks4 ))))//(4 *ks4 *ks5 ))%ks6 ))+((((x0 %(2 *ks4 )))%(2 *ks4 )))),xmask ,eviction_policy ='evict_last')
    tmp1 =tl .full ([1 ],2 ,tl .int32 )
    tmp2 =tl .where ((tmp0 <0 )!=(tmp1 <0 ),tl .where (tmp0 %tmp1 !=0 ,tmp0 //tmp1 -1 ,tmp0 //tmp1 ),tmp0 //tmp1 )
    tmp3 =tmp2 *tmp1 
    tmp4 =tmp0 -tmp3 
    tmp5 =2 *(((x0 //ks0 )%ks1 ))
    tmp6 =tmp5 +tmp2 
    tmp7 =2 *((x0 %ks0 ))
    tmp8 =tmp7 +tmp4 
    tmp9 =ks2 
    tmp10 =tmp6 *tmp9 
    tmp11 =tmp10 +tmp8 
    tmp12 =16 *ks4 *ks5 *(x0 //ks3 )
    tmp13 =tmp11 +tmp12 
    tmp14 =16 *ks4 *ks5 *ks6 
    tmp15 =tmp13 +tmp14 
    tmp16 =tmp13 <0 
    tmp17 =tl .where (tmp16 ,tmp15 ,tmp13 )
    tl .device_assert (((0 <=tmp17 )&(tmp17 <16 *ks6 *(ks2 //4 )*(ks7 //4 )))|~(xmask ),"index out of bounds: 0 <= tmp17 < 16*ks6*(ks2 // 4)*(ks7 // 4)")
    tl .store (out_ptr0 +(tl .broadcast_to ((tmp17 %(16 *ks4 *ks5 *ks6 )),[XBLOCK ])),tmp19 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_log_sigmoid_forward_5 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%ks1 )
    x2 =xindex //ks2 
    x3 =xindex 
    tmp0 =tl .load (in_ptr0 +(x0 +4 *ks3 *((((x0 +4 *ks3 *x1 )//(4 *ks3 ))%(4 *ks4 )))+16 *ks3 *ks4 *((((x0 +4 *ks3 *x1 +16 *ks3 *ks4 *x2 )//(16 *ks3 *ks4 ))%ks5 ))),xmask ,eviction_policy ='evict_last')
    tmp3 =tl .load (in_ptr0 +(x0 +4 *ks3 *((((x0 +4 *ks3 *x1 )//ks0 )%ks1 ))+16 *ks3 *ks4 *((((x0 +4 *ks3 *x1 +16 *ks3 *ks4 *x2 )//ks2 )%ks5 ))),xmask ,eviction_policy ='evict_last')
    tmp1 =0.0 
    tmp2 =triton_helpers .minimum (tmp1 ,tmp0 )
    tmp4 =tl_math .abs (tmp3 )
    tmp5 =-tmp4 
    tmp6 =tl_math .exp (tmp5 )
    tmp7 =libdevice .log1p (tmp6 )
    tmp8 =tmp2 -tmp7 
    tl .store (out_ptr0 +(x3 ),tmp8 ,xmask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        s2 //2 
        s1 //2 
        (s1 //2 )*(s2 //2 )
        buf0 =empty_strided_cuda ((1 ,s0 ,s1 //2 ,s2 //2 ),(s0 *(s1 //2 )*(s2 //2 ),(s1 //2 )*(s2 //2 ),s2 //2 ,1 ),torch .int8 )
        buf1 =empty_strided_cuda ((1 ,s0 ,s1 //2 ,s2 //2 ),(s0 *(s1 //2 )*(s2 //2 ),(s1 //2 )*(s2 //2 ),s2 //2 ,1 ),torch .float32 )
        buf3 =buf1 ;del buf1 

        triton_poi_fused_gelu_max_pool2d_with_indices_0_xnumel =s0 *(s1 //2 )*(s2 //2 )
        get_raw_stream (0 )
        triton_poi_fused_gelu_max_pool2d_with_indices_0 [grid (triton_poi_fused_gelu_max_pool2d_with_indices_0_xnumel )](buf3 ,arg3_1 ,buf0 ,32 ,32 ,1024 ,64 ,64 ,3072 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        buf5 =empty_strided_cuda ((1 ,s0 ,2 *(s1 //4 ),2 *(s2 //4 )),(4 *s0 *(s1 //4 )*(s2 //4 ),4 *(s1 //4 )*(s2 //4 ),2 *(s2 //4 ),1 ),torch .float32 )

        triton_poi_fused_max_unpool2d_1_xnumel =4 *s0 *(s1 //4 )*(s2 //4 )
        get_raw_stream (0 )
        triton_poi_fused_max_unpool2d_1 [grid (triton_poi_fused_max_unpool2d_1_xnumel )](buf5 ,3072 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        s2 //4 
        s1 //4 
        (s1 //4 )*(s2 //4 )

        triton_poi_fused_gelu_max_pool2d_with_indices_max_unpool2d_2_xnumel =s0 *(s1 //4 )*(s2 //4 )
        get_raw_stream (0 )
        triton_poi_fused_gelu_max_pool2d_with_indices_max_unpool2d_2 [grid (triton_poi_fused_gelu_max_pool2d_with_indices_max_unpool2d_2_xnumel )](buf3 ,buf5 ,16 ,16 ,256 ,32 ,32 ,3 ,64 ,64 ,768 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf3 
        buf7 =empty_strided_cuda ((1 ,s0 ,4 *(s1 //4 ),4 *(s2 //4 )),(16 *s0 *(s1 //4 )*(s2 //4 ),16 *(s1 //4 )*(s2 //4 ),4 *(s2 //4 ),1 ),torch .float32 )

        triton_poi_fused_max_unpool2d_3_xnumel =16 *s0 *(s1 //4 )*(s2 //4 )
        get_raw_stream (0 )
        triton_poi_fused_max_unpool2d_3 [grid (triton_poi_fused_max_unpool2d_3_xnumel )](buf7 ,12288 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )

        triton_poi_fused_max_unpool2d_4_xnumel =s0 *(s1 //2 )*(s2 //2 )
        get_raw_stream (0 )
        triton_poi_fused_max_unpool2d_4 [grid (triton_poi_fused_max_unpool2d_4_xnumel )](buf0 ,buf5 ,buf7 ,32 ,32 ,64 ,1024 ,16 ,16 ,3 ,64 ,3072 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf0 
        del buf5 
        4 *(s2 //4 )
        4 *(s1 //4 )
        16 *(s1 //4 )*(s2 //4 )
        buf9 =empty_strided_cuda ((1 ,s0 ,4 *(s1 //4 ),4 *(s2 //4 )),(16 *s0 *(s1 //4 )*(s2 //4 ),16 *(s1 //4 )*(s2 //4 ),4 *(s2 //4 ),1 ),torch .float32 )

        triton_poi_fused_log_sigmoid_forward_5_xnumel =16 *s0 *(s1 //4 )*(s2 //4 )
        get_raw_stream (0 )
        triton_poi_fused_log_sigmoid_forward_5 [grid (triton_poi_fused_log_sigmoid_forward_5_xnumel )](buf7 ,buf9 ,64 ,64 ,4096 ,16 ,16 ,3 ,12288 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del buf7 
    return (buf9 ,)

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
