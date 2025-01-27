
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
def triton_poi_fused__to_copy_0 (out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =64 
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
def triton_per_fused_log_sigmoid_forward_native_layer_norm_native_layer_norm_backward_1 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,out_ptr2 ,out_ptr3 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    R0_BLOCK :tl .constexpr =64 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_0 =r0_index 
    tmp0 =tl .load (in_out_ptr0 +(r0_0 ),None )
    tmp21 =tl .load (in_ptr0 +(r0_0 ),None )
    tmp23 =tl .load (in_ptr1 +(r0_0 ),None )
    tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
    tmp3 =tl .broadcast_to (tmp1 ,[XBLOCK ,R0_BLOCK ])
    tmp5 =tl .sum (tmp3 ,1 )[:,None ]
    tmp6 =tl .full ([XBLOCK ,1 ],64 ,tl .int32 )
    tmp7 =tmp6 .to (tl .float32 )
    tmp8 =tmp5 /tmp7 
    tmp9 =tmp1 -tmp8 
    tmp10 =tmp9 *tmp9 
    tmp11 =tl .broadcast_to (tmp10 ,[XBLOCK ,R0_BLOCK ])
    tmp13 =tl .sum (tmp11 ,1 )[:,None ]
    tmp14 =tmp0 -tmp8 
    tmp15 =64.0 
    tmp16 =tmp13 /tmp15 
    tmp17 =1e-05 
    tmp18 =tmp16 +tmp17 
    tmp19 =libdevice .rsqrt (tmp18 )
    tmp20 =tmp14 *tmp19 
    tmp22 =tmp20 *tmp21 
    tmp24 =tmp22 +tmp23 
    tmp25 =0.0 
    tmp26 =triton_helpers .minimum (tmp25 ,tmp24 )
    tmp27 =tl_math .abs (tmp24 )
    tmp28 =-tmp27 
    tmp29 =tl_math .exp (tmp28 )
    tmp30 =libdevice .log1p (tmp29 )
    tmp31 =tmp26 -tmp30 
    tmp32 =0.015625 
    tmp33 =tmp19 *tmp32 
    tl .store (in_out_ptr0 +(tl .broadcast_to (r0_0 ,[XBLOCK ,R0_BLOCK ])),tmp20 ,None )
    tl .store (out_ptr2 +(tl .broadcast_to (r0_0 ,[XBLOCK ,R0_BLOCK ])),tmp31 ,None )
    tl .store (out_ptr3 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp33 ,None )

def call (args ):
    primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ,primals_8 ,primals_9 ,primals_10 ,primals_11 =args 
    args .clear ()
    assert_size_stride (primals_1 ,(1 ,10 ,128 ),(1280 ,128 ,1 ))
    assert_size_stride (primals_2 ,(192 ,128 ),(128 ,1 ))
    assert_size_stride (primals_3 ,(192 ,64 ),(64 ,1 ))
    assert_size_stride (primals_4 ,(192 ,),(1 ,))
    assert_size_stride (primals_5 ,(192 ,),(1 ,))
    assert_size_stride (primals_6 ,(192 ,64 ),(64 ,1 ))
    assert_size_stride (primals_7 ,(192 ,64 ),(64 ,1 ))
    assert_size_stride (primals_8 ,(192 ,),(1 ,))
    assert_size_stride (primals_9 ,(192 ,),(1 ,))
    assert_size_stride (primals_10 ,(64 ,),(1 ,))
    assert_size_stride (primals_11 ,(64 ,),(1 ,))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((1 ,64 ),(64 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused__to_copy_0 [grid (64 )](buf0 ,64 ,XBLOCK =64 ,num_warps =1 ,num_stages =1 )
        buf1 =empty_strided_cuda ((1 ,192 ),(192 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),0 ),reinterpret_tensor (primals_2 ,(128 ,192 ),(1 ,128 ),0 ),out =buf1 )
        buf2 =empty_strided_cuda ((1 ,192 ),(192 ,1 ),torch .float32 )

        extern_kernels .mm (buf0 ,reinterpret_tensor (primals_3 ,(64 ,192 ),(1 ,64 ),0 ),out =buf2 )

        buf3 =torch .ops .aten ._thnn_fused_gru_cell .default (buf1 ,buf2 ,buf0 ,primals_4 ,primals_5 )
        buf4 =buf3 [0 ]
        buf5 =buf3 [1 ]
        del buf3 
        buf6 =buf2 ;del buf2 

        extern_kernels .mm (buf4 ,reinterpret_tensor (primals_6 ,(64 ,192 ),(1 ,64 ),0 ),out =buf6 )
        buf7 =buf1 ;del buf1 

        extern_kernels .mm (buf4 ,reinterpret_tensor (primals_7 ,(64 ,192 ),(1 ,64 ),0 ),out =buf7 )

        buf8 =torch .ops .aten ._thnn_fused_gru_cell .default (buf6 ,buf7 ,buf4 ,primals_8 ,primals_9 )
        buf9 =buf8 [0 ]
        buf10 =buf8 [1 ]
        del buf8 
        buf11 =buf7 ;del buf7 

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),128 ),reinterpret_tensor (primals_2 ,(128 ,192 ),(1 ,128 ),0 ),out =buf11 )
        buf12 =buf6 ;del buf6 

        extern_kernels .mm (buf9 ,reinterpret_tensor (primals_3 ,(64 ,192 ),(1 ,64 ),0 ),out =buf12 )

        buf13 =torch .ops .aten ._thnn_fused_gru_cell .default (buf11 ,buf12 ,buf9 ,primals_4 ,primals_5 )
        buf14 =buf13 [0 ]
        buf15 =buf13 [1 ]
        del buf13 
        buf16 =buf12 ;del buf12 

        extern_kernels .mm (buf14 ,reinterpret_tensor (primals_6 ,(64 ,192 ),(1 ,64 ),0 ),out =buf16 )
        buf17 =buf11 ;del buf11 

        extern_kernels .mm (buf14 ,reinterpret_tensor (primals_7 ,(64 ,192 ),(1 ,64 ),0 ),out =buf17 )

        buf18 =torch .ops .aten ._thnn_fused_gru_cell .default (buf16 ,buf17 ,buf14 ,primals_8 ,primals_9 )
        buf19 =buf18 [0 ]
        buf20 =buf18 [1 ]
        del buf18 
        buf21 =buf17 ;del buf17 

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),256 ),reinterpret_tensor (primals_2 ,(128 ,192 ),(1 ,128 ),0 ),out =buf21 )
        buf22 =buf16 ;del buf16 

        extern_kernels .mm (buf19 ,reinterpret_tensor (primals_3 ,(64 ,192 ),(1 ,64 ),0 ),out =buf22 )

        buf23 =torch .ops .aten ._thnn_fused_gru_cell .default (buf21 ,buf22 ,buf19 ,primals_4 ,primals_5 )
        buf24 =buf23 [0 ]
        buf25 =buf23 [1 ]
        del buf23 
        buf26 =buf22 ;del buf22 

        extern_kernels .mm (buf24 ,reinterpret_tensor (primals_6 ,(64 ,192 ),(1 ,64 ),0 ),out =buf26 )
        buf27 =buf21 ;del buf21 

        extern_kernels .mm (buf24 ,reinterpret_tensor (primals_7 ,(64 ,192 ),(1 ,64 ),0 ),out =buf27 )

        buf28 =torch .ops .aten ._thnn_fused_gru_cell .default (buf26 ,buf27 ,buf24 ,primals_8 ,primals_9 )
        buf29 =buf28 [0 ]
        buf30 =buf28 [1 ]
        del buf28 
        buf31 =buf27 ;del buf27 

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),384 ),reinterpret_tensor (primals_2 ,(128 ,192 ),(1 ,128 ),0 ),out =buf31 )
        buf32 =buf26 ;del buf26 

        extern_kernels .mm (buf29 ,reinterpret_tensor (primals_3 ,(64 ,192 ),(1 ,64 ),0 ),out =buf32 )

        buf33 =torch .ops .aten ._thnn_fused_gru_cell .default (buf31 ,buf32 ,buf29 ,primals_4 ,primals_5 )
        buf34 =buf33 [0 ]
        buf35 =buf33 [1 ]
        del buf33 
        buf36 =buf32 ;del buf32 

        extern_kernels .mm (buf34 ,reinterpret_tensor (primals_6 ,(64 ,192 ),(1 ,64 ),0 ),out =buf36 )
        buf37 =buf31 ;del buf31 

        extern_kernels .mm (buf34 ,reinterpret_tensor (primals_7 ,(64 ,192 ),(1 ,64 ),0 ),out =buf37 )

        buf38 =torch .ops .aten ._thnn_fused_gru_cell .default (buf36 ,buf37 ,buf34 ,primals_8 ,primals_9 )
        buf39 =buf38 [0 ]
        buf40 =buf38 [1 ]
        del buf38 
        buf41 =buf37 ;del buf37 

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),512 ),reinterpret_tensor (primals_2 ,(128 ,192 ),(1 ,128 ),0 ),out =buf41 )
        buf42 =buf36 ;del buf36 

        extern_kernels .mm (buf39 ,reinterpret_tensor (primals_3 ,(64 ,192 ),(1 ,64 ),0 ),out =buf42 )

        buf43 =torch .ops .aten ._thnn_fused_gru_cell .default (buf41 ,buf42 ,buf39 ,primals_4 ,primals_5 )
        buf44 =buf43 [0 ]
        buf45 =buf43 [1 ]
        del buf43 
        buf46 =buf42 ;del buf42 

        extern_kernels .mm (buf44 ,reinterpret_tensor (primals_6 ,(64 ,192 ),(1 ,64 ),0 ),out =buf46 )
        buf47 =buf41 ;del buf41 

        extern_kernels .mm (buf44 ,reinterpret_tensor (primals_7 ,(64 ,192 ),(1 ,64 ),0 ),out =buf47 )

        buf48 =torch .ops .aten ._thnn_fused_gru_cell .default (buf46 ,buf47 ,buf44 ,primals_8 ,primals_9 )
        buf49 =buf48 [0 ]
        buf50 =buf48 [1 ]
        del buf48 
        buf51 =buf47 ;del buf47 

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),640 ),reinterpret_tensor (primals_2 ,(128 ,192 ),(1 ,128 ),0 ),out =buf51 )
        buf52 =buf46 ;del buf46 

        extern_kernels .mm (buf49 ,reinterpret_tensor (primals_3 ,(64 ,192 ),(1 ,64 ),0 ),out =buf52 )

        buf53 =torch .ops .aten ._thnn_fused_gru_cell .default (buf51 ,buf52 ,buf49 ,primals_4 ,primals_5 )
        buf54 =buf53 [0 ]
        buf55 =buf53 [1 ]
        del buf53 
        buf56 =buf52 ;del buf52 

        extern_kernels .mm (buf54 ,reinterpret_tensor (primals_6 ,(64 ,192 ),(1 ,64 ),0 ),out =buf56 )
        buf57 =buf51 ;del buf51 

        extern_kernels .mm (buf54 ,reinterpret_tensor (primals_7 ,(64 ,192 ),(1 ,64 ),0 ),out =buf57 )

        buf58 =torch .ops .aten ._thnn_fused_gru_cell .default (buf56 ,buf57 ,buf54 ,primals_8 ,primals_9 )
        buf59 =buf58 [0 ]
        buf60 =buf58 [1 ]
        del buf58 
        buf61 =buf57 ;del buf57 

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),768 ),reinterpret_tensor (primals_2 ,(128 ,192 ),(1 ,128 ),0 ),out =buf61 )
        buf62 =buf56 ;del buf56 

        extern_kernels .mm (buf59 ,reinterpret_tensor (primals_3 ,(64 ,192 ),(1 ,64 ),0 ),out =buf62 )

        buf63 =torch .ops .aten ._thnn_fused_gru_cell .default (buf61 ,buf62 ,buf59 ,primals_4 ,primals_5 )
        buf64 =buf63 [0 ]
        buf65 =buf63 [1 ]
        del buf63 
        buf66 =buf62 ;del buf62 

        extern_kernels .mm (buf64 ,reinterpret_tensor (primals_6 ,(64 ,192 ),(1 ,64 ),0 ),out =buf66 )
        buf67 =buf61 ;del buf61 

        extern_kernels .mm (buf64 ,reinterpret_tensor (primals_7 ,(64 ,192 ),(1 ,64 ),0 ),out =buf67 )

        buf68 =torch .ops .aten ._thnn_fused_gru_cell .default (buf66 ,buf67 ,buf64 ,primals_8 ,primals_9 )
        buf69 =buf68 [0 ]
        buf70 =buf68 [1 ]
        del buf68 
        buf71 =buf67 ;del buf67 

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),896 ),reinterpret_tensor (primals_2 ,(128 ,192 ),(1 ,128 ),0 ),out =buf71 )
        buf72 =buf66 ;del buf66 

        extern_kernels .mm (buf69 ,reinterpret_tensor (primals_3 ,(64 ,192 ),(1 ,64 ),0 ),out =buf72 )

        buf73 =torch .ops .aten ._thnn_fused_gru_cell .default (buf71 ,buf72 ,buf69 ,primals_4 ,primals_5 )
        buf74 =buf73 [0 ]
        buf75 =buf73 [1 ]
        del buf73 
        buf76 =buf72 ;del buf72 

        extern_kernels .mm (buf74 ,reinterpret_tensor (primals_6 ,(64 ,192 ),(1 ,64 ),0 ),out =buf76 )
        buf77 =buf71 ;del buf71 

        extern_kernels .mm (buf74 ,reinterpret_tensor (primals_7 ,(64 ,192 ),(1 ,64 ),0 ),out =buf77 )

        buf78 =torch .ops .aten ._thnn_fused_gru_cell .default (buf76 ,buf77 ,buf74 ,primals_8 ,primals_9 )
        buf79 =buf78 [0 ]
        buf80 =buf78 [1 ]
        del buf78 
        buf81 =buf77 ;del buf77 

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),1024 ),reinterpret_tensor (primals_2 ,(128 ,192 ),(1 ,128 ),0 ),out =buf81 )
        buf82 =buf76 ;del buf76 

        extern_kernels .mm (buf79 ,reinterpret_tensor (primals_3 ,(64 ,192 ),(1 ,64 ),0 ),out =buf82 )

        buf83 =torch .ops .aten ._thnn_fused_gru_cell .default (buf81 ,buf82 ,buf79 ,primals_4 ,primals_5 )
        buf84 =buf83 [0 ]
        buf85 =buf83 [1 ]
        del buf83 
        buf86 =buf82 ;del buf82 

        extern_kernels .mm (buf84 ,reinterpret_tensor (primals_6 ,(64 ,192 ),(1 ,64 ),0 ),out =buf86 )
        buf87 =buf81 ;del buf81 

        extern_kernels .mm (buf84 ,reinterpret_tensor (primals_7 ,(64 ,192 ),(1 ,64 ),0 ),out =buf87 )

        buf88 =torch .ops .aten ._thnn_fused_gru_cell .default (buf86 ,buf87 ,buf84 ,primals_8 ,primals_9 )
        buf89 =buf88 [0 ]
        buf90 =buf88 [1 ]
        del buf88 
        buf91 =buf87 ;del buf87 

        extern_kernels .mm (reinterpret_tensor (primals_1 ,(1 ,128 ),(128 ,1 ),1152 ),reinterpret_tensor (primals_2 ,(128 ,192 ),(1 ,128 ),0 ),out =buf91 )
        del primals_2 
        buf92 =buf86 ;del buf86 

        extern_kernels .mm (buf89 ,reinterpret_tensor (primals_3 ,(64 ,192 ),(1 ,64 ),0 ),out =buf92 )

        buf93 =torch .ops .aten ._thnn_fused_gru_cell .default (buf91 ,buf92 ,buf89 ,primals_4 ,primals_5 )
        del primals_4 
        del primals_5 
        buf94 =buf93 [0 ]
        buf95 =buf93 [1 ]
        del buf93 
        buf96 =buf92 ;del buf92 

        extern_kernels .mm (buf94 ,reinterpret_tensor (primals_6 ,(64 ,192 ),(1 ,64 ),0 ),out =buf96 )
        buf97 =buf91 ;del buf91 

        extern_kernels .mm (buf94 ,reinterpret_tensor (primals_7 ,(64 ,192 ),(1 ,64 ),0 ),out =buf97 )

        buf98 =torch .ops .aten ._thnn_fused_gru_cell .default (buf96 ,buf97 ,buf94 ,primals_8 ,primals_9 )
        del buf96 
        del buf97 
        del primals_8 
        del primals_9 
        buf99 =buf98 [0 ]
        buf100 =buf98 [1 ]
        del buf98 
        buf104 =buf99 ;del buf99 
        buf105 =empty_strided_cuda ((1 ,64 ),(64 ,1 ),torch .float32 )
        buf106 =empty_strided_cuda ((1 ,1 ),(1 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused_log_sigmoid_forward_native_layer_norm_native_layer_norm_backward_1 [grid (1 )](buf104 ,primals_10 ,primals_11 ,buf105 ,buf106 ,1 ,64 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
    return (buf105 ,primals_10 ,primals_11 ,buf0 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),0 ),buf4 ,buf5 ,buf9 ,buf10 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),128 ),buf14 ,buf15 ,buf19 ,buf20 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),256 ),buf24 ,buf25 ,buf29 ,buf30 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),384 ),buf34 ,buf35 ,buf39 ,buf40 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),512 ),buf44 ,buf45 ,buf49 ,buf50 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),640 ),buf54 ,buf55 ,buf59 ,buf60 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),768 ),buf64 ,buf65 ,buf69 ,buf70 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),896 ),buf74 ,buf75 ,buf79 ,buf80 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),1024 ),buf84 ,buf85 ,buf89 ,buf90 ,reinterpret_tensor (primals_1 ,(1 ,128 ),(1280 ,1 ),1152 ),buf94 ,buf95 ,buf100 ,buf104 ,buf106 ,primals_7 ,primals_6 ,primals_3 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    primals_1 =rand_strided ((1 ,10 ,128 ),(1280 ,128 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_2 =rand_strided ((192 ,128 ),(128 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_3 =rand_strided ((192 ,64 ),(64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_4 =rand_strided ((192 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_5 =rand_strided ((192 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_6 =rand_strided ((192 ,64 ),(64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_7 =rand_strided ((192 ,64 ),(64 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_8 =rand_strided ((192 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_9 =rand_strided ((192 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_10 =rand_strided ((64 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_11 =rand_strided ((64 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ,primals_8 ,primals_9 ,primals_10 ,primals_11 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
