
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
def triton_poi_fused_reflection_pad3d_relu_0 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%ks1 )
    x2 =xindex //ks2 
    x3 =xindex 
    tmp0 =tl .load (in_ptr0 +(ks5 *(tl .where ((-1 )+ks4 +((-1 )*tl_math .abs (1 +((-1 )*ks4 )+tl_math .abs ((-1 )+(tl .where (1 +ks4 +((-1 )*tl_math .abs (1 +ks4 +((-1 )*tl_math .abs ((-2 )+x1 ))))<0 ,3 +((-1 )*tl_math .abs (1 +ks4 +((-1 )*tl_math .abs ((-2 )+x1 ))))+2 *ks4 ,1 +ks4 +((-1 )*tl_math .abs (1 +ks4 +((-1 )*tl_math .abs ((-2 )+x1 )))))))))<0 ,(-1 )+((-1 )*tl_math .abs (1 +((-1 )*ks4 )+tl_math .abs ((-1 )+(tl .where (1 +ks4 +((-1 )*tl_math .abs (1 +ks4 +((-1 )*tl_math .abs ((-2 )+x1 ))))<0 ,3 +((-1 )*tl_math .abs (1 +ks4 +((-1 )*tl_math .abs ((-2 )+x1 ))))+2 *ks4 ,1 +ks4 +((-1 )*tl_math .abs (1 +ks4 +((-1 )*tl_math .abs ((-2 )+x1 )))))))))+2 *ks4 ,(-1 )+ks4 +((-1 )*tl_math .abs (1 +((-1 )*ks4 )+tl_math .abs ((-1 )+(tl .where (1 +ks4 +((-1 )*tl_math .abs (1 +ks4 +((-1 )*tl_math .abs ((-2 )+x1 ))))<0 ,3 +((-1 )*tl_math .abs (1 +ks4 +((-1 )*tl_math .abs ((-2 )+x1 ))))+2 *ks4 ,1 +ks4 +((-1 )*tl_math .abs (1 +ks4 +((-1 )*tl_math .abs ((-2 )+x1 )))))))))))+ks4 *ks5 *(tl .where ((-1 )+ks3 +((-1 )*tl_math .abs (1 +((-1 )*ks3 )+tl_math .abs ((-1 )+(tl .where (1 +ks3 +((-1 )*tl_math .abs (1 +ks3 +((-1 )*tl_math .abs ((-2 )+x2 ))))<0 ,3 +((-1 )*tl_math .abs (1 +ks3 +((-1 )*tl_math .abs ((-2 )+x2 ))))+2 *ks3 ,1 +ks3 +((-1 )*tl_math .abs (1 +ks3 +((-1 )*tl_math .abs ((-2 )+x2 )))))))))<0 ,(-1 )+((-1 )*tl_math .abs (1 +((-1 )*ks3 )+tl_math .abs ((-1 )+(tl .where (1 +ks3 +((-1 )*tl_math .abs (1 +ks3 +((-1 )*tl_math .abs ((-2 )+x2 ))))<0 ,3 +((-1 )*tl_math .abs (1 +ks3 +((-1 )*tl_math .abs ((-2 )+x2 ))))+2 *ks3 ,1 +ks3 +((-1 )*tl_math .abs (1 +ks3 +((-1 )*tl_math .abs ((-2 )+x2 )))))))))+2 *ks3 ,(-1 )+ks3 +((-1 )*tl_math .abs (1 +((-1 )*ks3 )+tl_math .abs ((-1 )+(tl .where (1 +ks3 +((-1 )*tl_math .abs (1 +ks3 +((-1 )*tl_math .abs ((-2 )+x2 ))))<0 ,3 +((-1 )*tl_math .abs (1 +ks3 +((-1 )*tl_math .abs ((-2 )+x2 ))))+2 *ks3 ,1 +ks3 +((-1 )*tl_math .abs (1 +ks3 +((-1 )*tl_math .abs ((-2 )+x2 )))))))))))+(tl .where ((-1 )+ks5 +((-1 )*tl_math .abs (1 +((-1 )*ks5 )+tl_math .abs ((-1 )+(tl .where (1 +ks5 +((-1 )*tl_math .abs (1 +ks5 +((-1 )*tl_math .abs ((-2 )+x0 ))))<0 ,3 +((-1 )*tl_math .abs (1 +ks5 +((-1 )*tl_math .abs ((-2 )+x0 ))))+2 *ks5 ,1 +ks5 +((-1 )*tl_math .abs (1 +ks5 +((-1 )*tl_math .abs ((-2 )+x0 )))))))))<0 ,(-1 )+((-1 )*tl_math .abs (1 +((-1 )*ks5 )+tl_math .abs ((-1 )+(tl .where (1 +ks5 +((-1 )*tl_math .abs (1 +ks5 +((-1 )*tl_math .abs ((-2 )+x0 ))))<0 ,3 +((-1 )*tl_math .abs (1 +ks5 +((-1 )*tl_math .abs ((-2 )+x0 ))))+2 *ks5 ,1 +ks5 +((-1 )*tl_math .abs (1 +ks5 +((-1 )*tl_math .abs ((-2 )+x0 )))))))))+2 *ks5 ,(-1 )+ks5 +((-1 )*tl_math .abs (1 +((-1 )*ks5 )+tl_math .abs ((-1 )+(tl .where (1 +ks5 +((-1 )*tl_math .abs (1 +ks5 +((-1 )*tl_math .abs ((-2 )+x0 ))))<0 ,3 +((-1 )*tl_math .abs (1 +ks5 +((-1 )*tl_math .abs ((-2 )+x0 ))))+2 *ks5 ,1 +ks5 +((-1 )*tl_math .abs (1 +ks5 +((-1 )*tl_math .abs ((-2 )+x0 )))))))))))),xmask ,eviction_policy ='evict_last')
    tmp1 =tl .full ([1 ],0 ,tl .int32 )
    tmp2 =triton_helpers .maximum (tmp1 ,tmp0 )
    tl .store (out_ptr0 +(x3 ),tmp2 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_elu_reflection_pad3d_relu_1 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ks5 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%ks1 )
    x2 =xindex //ks2 
    x3 =xindex 
    tmp0 =tl .load (in_ptr0 +(6 *(tl .where (5 +ks4 +((-1 )*tl_math .abs (5 +ks4 +((-1 )*tl_math .abs ((-3 )+x1 ))))<0 ,11 +((-1 )*tl_math .abs (5 +ks4 +((-1 )*tl_math .abs ((-3 )+x1 ))))+2 *ks4 ,5 +ks4 +((-1 )*tl_math .abs (5 +ks4 +((-1 )*tl_math .abs ((-3 )+x1 ))))))+36 *(tl .where (5 +ks3 +((-1 )*tl_math .abs (5 +ks3 +((-1 )*tl_math .abs ((-3 )+x2 ))))<0 ,11 +((-1 )*tl_math .abs (5 +ks3 +((-1 )*tl_math .abs ((-3 )+x2 ))))+2 *ks3 ,5 +ks3 +((-1 )*tl_math .abs (5 +ks3 +((-1 )*tl_math .abs ((-3 )+x2 ))))))+ks5 *(tl .where (5 +ks4 +((-1 )*tl_math .abs (5 +ks4 +((-1 )*tl_math .abs ((-3 )+x1 ))))<0 ,11 +((-1 )*tl_math .abs (5 +ks4 +((-1 )*tl_math .abs ((-3 )+x1 ))))+2 *ks4 ,5 +ks4 +((-1 )*tl_math .abs (5 +ks4 +((-1 )*tl_math .abs ((-3 )+x1 ))))))+6 *ks4 *(tl .where (5 +ks3 +((-1 )*tl_math .abs (5 +ks3 +((-1 )*tl_math .abs ((-3 )+x2 ))))<0 ,11 +((-1 )*tl_math .abs (5 +ks3 +((-1 )*tl_math .abs ((-3 )+x2 ))))+2 *ks3 ,5 +ks3 +((-1 )*tl_math .abs (5 +ks3 +((-1 )*tl_math .abs ((-3 )+x2 ))))))+6 *ks5 *(tl .where (5 +ks3 +((-1 )*tl_math .abs (5 +ks3 +((-1 )*tl_math .abs ((-3 )+x2 ))))<0 ,11 +((-1 )*tl_math .abs (5 +ks3 +((-1 )*tl_math .abs ((-3 )+x2 ))))+2 *ks3 ,5 +ks3 +((-1 )*tl_math .abs (5 +ks3 +((-1 )*tl_math .abs ((-3 )+x2 ))))))+ks4 *ks5 *(tl .where (5 +ks3 +((-1 )*tl_math .abs (5 +ks3 +((-1 )*tl_math .abs ((-3 )+x2 ))))<0 ,11 +((-1 )*tl_math .abs (5 +ks3 +((-1 )*tl_math .abs ((-3 )+x2 ))))+2 *ks3 ,5 +ks3 +((-1 )*tl_math .abs (5 +ks3 +((-1 )*tl_math .abs ((-3 )+x2 ))))))+(tl .where (5 +ks5 +((-1 )*tl_math .abs (5 +ks5 +((-1 )*tl_math .abs ((-3 )+x0 ))))<0 ,11 +((-1 )*tl_math .abs (5 +ks5 +((-1 )*tl_math .abs ((-3 )+x0 ))))+2 *ks5 ,5 +ks5 +((-1 )*tl_math .abs (5 +ks5 +((-1 )*tl_math .abs ((-3 )+x0 ))))))),xmask ,eviction_policy ='evict_last')
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
    tmp11 =tl .full ([1 ],0 ,tl .int32 )
    tmp12 =triton_helpers .maximum (tmp11 ,tmp10 )
    tl .store (out_ptr0 +(x3 ),tmp12 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_view_2 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(12 *(((x0 //ks0 )%ks1 ))+144 *(x0 //(144 +12 *ks2 +12 *ks3 +ks2 *ks3 ))+ks3 *(((x0 //ks0 )%ks1 ))+12 *ks2 *(x0 //(144 +12 *ks2 +12 *ks3 +ks2 *ks3 ))+12 *ks3 *(x0 //(144 +12 *ks2 +12 *ks3 +ks2 *ks3 ))+ks2 *ks3 *(x0 //(144 +12 *ks2 +12 *ks3 +ks2 *ks3 ))+((x0 %ks0 ))),xmask ,eviction_policy ='evict_last')
    tl .store (out_ptr0 +(x0 ),tmp0 ,xmask )

def call (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
    args .clear ()
    s0 =arg0_1 
    s1 =arg1_1 
    s2 =arg2_1 
    assert_size_stride (arg3_1 ,(1 ,1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        6 +s2 
        6 +s1 
        36 +6 *s1 +6 *s2 +s1 *s2 
        buf0 =empty_strided_cuda ((1 ,1 ,6 +s0 ,6 +s1 ,6 +s2 ),(216 +36 *s0 +36 *s1 +36 *s2 +6 *s0 *s1 +6 *s0 *s2 +6 *s1 *s2 +s0 *s1 *s2 ,216 +36 *s0 +36 *s1 +36 *s2 +6 *s0 *s1 +6 *s0 *s2 +6 *s1 *s2 +s0 *s1 *s2 ,36 +6 *s1 +6 *s2 +s1 *s2 ,6 +s2 ,1 ),torch .float32 )

        triton_poi_fused_reflection_pad3d_relu_0_xnumel =216 +36 *s0 +36 *s1 +36 *s2 +6 *s0 *s1 +6 *s0 *s2 +6 *s1 *s2 +s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_reflection_pad3d_relu_0 [grid (triton_poi_fused_reflection_pad3d_relu_0_xnumel )](arg3_1 ,buf0 ,70 ,70 ,4900 ,64 ,64 ,64 ,343000 ,XBLOCK =512 ,num_warps =8 ,num_stages =1 )
        del arg3_1 
        12 +s2 
        12 +s1 
        144 +12 *s1 +12 *s2 +s1 *s2 
        buf1 =empty_strided_cuda ((1 ,1 ,12 +s0 ,12 +s1 ,12 +s2 ),(1728 +144 *s0 +144 *s1 +144 *s2 +12 *s0 *s1 +12 *s0 *s2 +12 *s1 *s2 +s0 *s1 *s2 ,1 ,144 +12 *s1 +12 *s2 +s1 *s2 ,12 +s2 ,1 ),torch .float32 )

        triton_poi_fused_elu_reflection_pad3d_relu_1_xnumel =1728 +144 *s0 +144 *s1 +144 *s2 +12 *s0 *s1 +12 *s0 *s2 +12 *s1 *s2 +s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_elu_reflection_pad3d_relu_1 [grid (triton_poi_fused_elu_reflection_pad3d_relu_1_xnumel )](buf0 ,buf1 ,76 ,76 ,5776 ,64 ,64 ,64 ,438976 ,XBLOCK =512 ,num_warps =8 ,num_stages =1 )
        del buf0 
        buf2 =empty_strided_cuda ((1 ,1 ,1728 +144 *s0 +144 *s1 +144 *s2 +12 *s0 *s1 +12 *s0 *s2 +12 *s1 *s2 +s0 *s1 *s2 ),(1728 +144 *s0 +144 *s1 +144 *s2 +12 *s0 *s1 +12 *s0 *s2 +12 *s1 *s2 +s0 *s1 *s2 ,1728 +144 *s0 +144 *s1 +144 *s2 +12 *s0 *s1 +12 *s0 *s2 +12 *s1 *s2 +s0 *s1 *s2 ,1 ),torch .float32 )

        triton_poi_fused_view_2_xnumel =1728 +144 *s0 +144 *s1 +144 *s2 +12 *s0 *s1 +12 *s0 *s2 +12 *s1 *s2 +s0 *s1 *s2 
        get_raw_stream (0 )
        triton_poi_fused_view_2 [grid (triton_poi_fused_view_2_xnumel )](buf1 ,buf2 ,76 ,76 ,64 ,64 ,438976 ,XBLOCK =1024 ,num_warps =4 ,num_stages =1 )
        del buf1 
    return (buf2 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    arg0_1 =64 
    arg1_1 =64 
    arg2_1 =64 
    arg3_1 =rand_strided ((1 ,1 ,64 ,64 ,64 ),(262144 ,262144 ,4096 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
