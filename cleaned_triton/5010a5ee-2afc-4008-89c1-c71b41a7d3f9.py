
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
def triton_poi_fused_avg_pool2d_softplus_0 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %ks0 )
    x1 =((xindex //ks0 )%ks1 )
    x2 =xindex //ks2 
    x3 =xindex 
    tmp0 =tl .load (in_ptr0 +(2 *x0 +2 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp1 =tl .load (in_ptr0 +(1 +2 *x0 +2 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp3 =tl .load (in_ptr0 +(ks4 +2 *x0 +2 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp5 =tl .load (in_ptr0 +(1 +ks4 +2 *x0 +2 *ks4 *x1 +ks3 *ks4 *x2 ),xmask ,eviction_policy ='evict_last')
    tmp2 =tmp1 +tmp0 
    tmp4 =tmp3 +tmp2 
    tmp6 =tmp5 +tmp4 
    tmp7 =0.25 
    tmp8 =tmp6 *tmp7 
    tmp9 =1.0 
    tmp10 =tmp8 *tmp9 
    tmp11 =20.0 
    tmp12 =tmp10 >tmp11 
    tmp13 =tl_math .exp (tmp10 )
    tmp14 =libdevice .log1p (tmp13 )
    tmp15 =tmp14 *tmp9 
    tmp16 =tl .where (tmp12 ,tmp8 ,tmp15 )
    tl .store (out_ptr0 +(x3 ),tmp16 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_col2im_1 (out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
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
def triton_poi_fused_col2im_2 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,ks3 ,ks4 ,ynumel ,xnumel ,YBLOCK :tl .constexpr ,XBLOCK :tl .constexpr ):
    xnumel =16 
    yoffset =(tl .program_id (1 )+tl .program_id (2 )*tl .num_programs (1 ))*YBLOCK 
    yindex =yoffset +tl .arange (0 ,YBLOCK )[None ,:]
    ymask =yindex <ynumel 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    x3 =(xindex %4 )
    x4 =xindex //4 
    y0 =(yindex %2 )
    y1 =((yindex //2 )%2 )
    y2 =yindex //4 
    tmp0 =tl .load (in_ptr0 +(2 *(((x3 +4 *x4 +y0 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+2 *y1 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+4 *y2 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 )))%(ks4 //4 )))+2 *ks0 *((((x3 +4 *x4 +y0 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+2 *y1 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+4 *y2 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 )))//(ks4 //4 ))%(ks3 //4 )))+ks0 *ks1 *((((x3 +4 *x4 +y0 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+2 *y1 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+4 *y2 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 )))//((ks3 //4 )*(ks4 //4 )))%ks2 ))),xmask &ymask ,eviction_policy ='evict_last')
    tmp1 =tl .load (in_ptr0 +(1 +2 *(((x3 +4 *x4 +y0 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+2 *y1 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+4 *y2 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 )))%(ks4 //4 )))+2 *ks0 *((((x3 +4 *x4 +y0 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+2 *y1 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+4 *y2 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 )))//(ks4 //4 ))%(ks3 //4 )))+ks0 *ks1 *((((x3 +4 *x4 +y0 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+2 *y1 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+4 *y2 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 )))//((ks3 //4 )*(ks4 //4 )))%ks2 ))),xmask &ymask ,eviction_policy ='evict_last')
    tmp3 =tl .load (in_ptr0 +(ks0 +2 *(((x3 +4 *x4 +y0 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+2 *y1 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+4 *y2 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 )))%(ks4 //4 )))+2 *ks0 *((((x3 +4 *x4 +y0 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+2 *y1 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+4 *y2 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 )))//(ks4 //4 ))%(ks3 //4 )))+ks0 *ks1 *((((x3 +4 *x4 +y0 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+2 *y1 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+4 *y2 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 )))//((ks3 //4 )*(ks4 //4 )))%ks2 ))),xmask &ymask ,eviction_policy ='evict_last')
    tmp5 =tl .load (in_ptr0 +(1 +ks0 +2 *(((x3 +4 *x4 +y0 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+2 *y1 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+4 *y2 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 )))%(ks4 //4 )))+2 *ks0 *((((x3 +4 *x4 +y0 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+2 *y1 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+4 *y2 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 )))//(ks4 //4 ))%(ks3 //4 )))+ks0 *ks1 *((((x3 +4 *x4 +y0 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+2 *y1 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 ))+4 *y2 *(triton_helpers .div_floor_integer ((ks3 //4 )*(ks4 //4 ),4 )))//((ks3 //4 )*(ks4 //4 )))%ks2 ))),xmask &ymask ,eviction_policy ='evict_last')
    tmp2 =tmp1 +tmp0 
    tmp4 =tmp3 +tmp2 
    tmp6 =tmp5 +tmp4 
    tmp7 =0.25 
    tmp8 =tmp6 *tmp7 
    tl .atomic_add (out_ptr0 +(y0 +2 *x3 +8 *y1 +16 *x4 +64 *y2 ),tmp8 ,xmask &ymask ,sem ='relaxed')

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
        buf0 =empty_strided_cuda ((1 ,s0 ,s1 //2 ,s2 //2 ),(s0 *(s1 //2 )*(s2 //2 ),(s1 //2 )*(s2 //2 ),s2 //2 ,1 ),torch .float32 )

        triton_poi_fused_avg_pool2d_softplus_0_xnumel =s0 *(s1 //2 )*(s2 //2 )
        get_raw_stream (0 )
        triton_poi_fused_avg_pool2d_softplus_0 [grid (triton_poi_fused_avg_pool2d_softplus_0_xnumel )](arg3_1 ,buf0 ,16 ,16 ,256 ,32 ,32 ,768 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del arg3_1 
        buf1 =empty_strided_cuda ((1 ,s0 ,8 ,8 ),(64 *s0 ,64 ,8 ,1 ),torch .float32 )

        triton_poi_fused_col2im_1_xnumel =64 *s0 
        get_raw_stream (0 )
        triton_poi_fused_col2im_1 [grid (triton_poi_fused_col2im_1_xnumel )](buf1 ,192 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )

        triton_poi_fused_col2im_2_ynumel =4 *s0 
        get_raw_stream (0 )
        triton_poi_fused_col2im_2 [grid (triton_poi_fused_col2im_2_ynumel ,16 )](buf0 ,buf1 ,16 ,16 ,3 ,32 ,32 ,12 ,16 ,XBLOCK =16 ,YBLOCK =16 ,num_warps =4 ,num_stages =1 )
        del buf0 
    return (buf1 ,)

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
