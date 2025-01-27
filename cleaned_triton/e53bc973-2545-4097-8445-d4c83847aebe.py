
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
def triton_poi_fused_abs_adaptive_max_pool2d_gt_mish_mul_sign_sub_where_0 (in_out_ptr0 ,in_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =1000 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =(xindex %10 )
    x1 =xindex //10 
    x2 =xindex 
    tmp0 =tl .full ([1 ],0 ,tl .int64 )
    tmp1 =tl .full ([1 ],1 ,tl .int64 )
    tmp2 =tmp0 <tmp1 
    tmp3 =(7 *x0 )//5 
    tmp4 =(23 +14 *x0 )//10 
    tmp5 =tmp3 <tmp4 
    tmp6 =tmp2 &tmp5 
    tmp7 =(-2 )+((7 *x0 )//5 )
    tmp8 =tl .full ([1 ],0 ,tl .int64 )
    tmp9 =tmp7 >=tmp8 
    tmp10 =tl .full ([1 ],10 ,tl .int64 )
    tmp11 =tmp7 <tmp10 
    tmp12 =tmp9 &tmp11 
    tmp13 =tmp12 &tmp6 
    tmp14 =tl .load (in_ptr0 +((-2 )+10 *x1 +((7 *x0 )//5 )),tmp13 &xmask ,other =0.0 )
    tmp15 =tl .full (tmp14 .shape ,float ("-inf"),tmp14 .dtype )
    tmp16 =tl .where (tmp6 ,tmp14 ,tmp15 )
    tmp17 =1 +((7 *x0 )//5 )
    tmp18 =tmp17 <tmp4 
    tmp19 =tmp2 &tmp18 
    tmp20 =(-1 )+((7 *x0 )//5 )
    tmp21 =tl .full ([1 ],0 ,tl .int64 )
    tmp22 =tmp20 >=tmp21 
    tmp23 =tl .full ([1 ],10 ,tl .int64 )
    tmp24 =tmp20 <tmp23 
    tmp25 =tmp22 &tmp24 
    tmp26 =tmp25 &tmp19 
    tmp27 =tl .load (in_ptr0 +((-1 )+10 *x1 +((7 *x0 )//5 )),tmp26 &xmask ,other =0.0 )
    tmp28 =tl .full (tmp27 .shape ,float ("-inf"),tmp27 .dtype )
    tmp29 =tl .where (tmp19 ,tmp27 ,tmp28 )
    tmp30 =triton_helpers .maximum (tmp29 ,tmp16 )
    tmp31 =2 +((7 *x0 )//5 )
    tmp32 =tmp31 <tmp4 
    tmp33 =tmp2 &tmp32 
    tmp34 =(7 *x0 )//5 
    tmp35 =tl .full ([1 ],0 ,tl .int64 )
    tmp36 =tmp34 >=tmp35 
    tmp37 =tl .full ([1 ],10 ,tl .int64 )
    tmp38 =tmp34 <tmp37 
    tmp39 =tmp36 &tmp38 
    tmp40 =tmp39 &tmp33 
    tmp41 =tl .load (in_ptr0 +(10 *x1 +((7 *x0 )//5 )),tmp40 &xmask ,other =0.0 )
    tmp42 =tl .full (tmp41 .shape ,float ("-inf"),tmp41 .dtype )
    tmp43 =tl .where (tmp33 ,tmp41 ,tmp42 )
    tmp44 =triton_helpers .maximum (tmp43 ,tmp30 )
    tmp45 =20.0 
    tmp46 =tmp44 >tmp45 
    tmp47 =tl_math .exp (tmp44 )
    tmp48 =libdevice .log1p (tmp47 )
    tmp49 =tl .where (tmp46 ,tmp44 ,tmp48 )
    tmp50 =libdevice .tanh (tmp49 )
    tmp51 =tmp44 *tmp50 
    tmp52 =tl_math .abs (tmp51 )
    tmp53 =0.5 
    tmp54 =tmp52 >tmp53 
    tmp55 =tl .full ([1 ],0 ,tl .int32 )
    tmp56 =tmp55 <tmp51 
    tmp57 =tmp56 .to (tl .int8 )
    tmp58 =tmp51 <tmp55 
    tmp59 =tmp58 .to (tl .int8 )
    tmp60 =tmp57 -tmp59 
    tmp61 =tmp60 .to (tmp51 .dtype )
    tmp62 =tmp61 *tmp53 
    tmp63 =tmp51 -tmp62 
    tmp64 =0.0 
    tmp65 =tmp51 *tmp64 
    tmp66 =tl .where (tmp54 ,tmp63 ,tmp65 )
    tl .store (in_out_ptr0 +(x2 ),tmp66 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice ,math as tl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__to_copy_1 (out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =2000 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =0.0 
    tl .store (out_ptr0 +(x0 ),tmp0 ,xmask )

def call (args ):
    primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 =args 
    args .clear ()
    assert_size_stride (primals_1 ,(5 ,20 ,10 ),(200 ,10 ,1 ))
    assert_size_stride (primals_2 ,(80 ,10 ),(10 ,1 ))
    assert_size_stride (primals_3 ,(80 ,20 ),(20 ,1 ))
    assert_size_stride (primals_4 ,(80 ,),(1 ,))
    assert_size_stride (primals_5 ,(80 ,),(1 ,))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
        buf0 =empty_strided_cuda ((5 ,20 ,1 ,10 ),(200 ,10 ,10 ,1 ),torch .float32 )
        buf1 =reinterpret_tensor (buf0 ,(5 ,20 ,10 ),(200 ,10 ,1 ),0 );del buf0 

        get_raw_stream (0 )
        triton_poi_fused_abs_adaptive_max_pool2d_gt_mish_mul_sign_sub_where_0 [grid (1000 )](buf1 ,primals_1 ,1000 ,XBLOCK =128 ,num_warps =4 ,num_stages =1 )
        del primals_1 
        buf2 =empty_strided_cuda ((100 ,20 ),(20 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused__to_copy_1 [grid (2000 )](buf2 ,2000 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
        buf3 =empty_strided_cuda ((100 ,80 ),(80 ,1 ),torch .float32 )

        extern_kernels .mm (reinterpret_tensor (buf1 ,(100 ,10 ),(10 ,1 ),0 ),reinterpret_tensor (primals_2 ,(10 ,80 ),(1 ,10 ),0 ),out =buf3 )
        del primals_2 
        buf4 =empty_strided_cuda ((100 ,80 ),(80 ,1 ),torch .float32 )

        extern_kernels .mm (buf2 ,reinterpret_tensor (primals_3 ,(20 ,80 ),(1 ,20 ),0 ),out =buf4 )
        del primals_3 

        buf5 =torch .ops .aten ._thnn_fused_lstm_cell .default (buf3 ,buf4 ,buf2 ,primals_4 ,primals_5 )
        del buf3 
        del buf4 
        del primals_4 
        del primals_5 
        buf6 =buf5 [0 ]
        buf7 =buf5 [1 ]
        buf8 =buf5 [2 ]
        del buf5 
    return (reinterpret_tensor (buf6 ,(5 ,20 ,20 ),(400 ,20 ,1 ),0 ),reinterpret_tensor (buf1 ,(100 ,10 ),(10 ,1 ),0 ),buf2 ,buf7 ,buf8 ,)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    primals_1 =rand_strided ((5 ,20 ,10 ),(200 ,10 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_2 =rand_strided ((80 ,10 ),(10 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_3 =rand_strided ((80 ,20 ),(20 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_4 =rand_strided ((80 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_5 =rand_strided ((80 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
