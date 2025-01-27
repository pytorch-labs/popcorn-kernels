
importtorch 
importtriton 
importtriton .languageastl 
fromtorch ._inductor .runtime .triton_heuristicsimport (
grid ,
)
fromtorch ._Cimport_cuda_getCurrentRawStreamasget_raw_stream 
fromtorch ._Cimport_cuda_getCurrentRawStreamasget_raw_stream 

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

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_poi_fused__adaptive_avg_pool2d_0 (in_ptr0 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
xmask =xindex <xnumel 
x0 =xindex 
tmp0 =tl .load (in_ptr0 +(2 *x0 ),xmask ,eviction_policy ='evict_last')
tmp1 =tl .load (in_ptr0 +(1 +2 *x0 ),xmask ,eviction_policy ='evict_last')
tmp2 =tmp1 +tmp0 
tmp3 =0.5 
tmp4 =tmp2 *tmp3 
tl .store (out_ptr0 +(x0 ),tmp4 ,xmask )

defcall (args ):
    arg0_1 ,arg1_1 ,arg2_1 =args 
args .clear ()
s0 =arg0_1 
s1 =arg1_1 
assert_size_stride (arg2_1 ,(1 ,s0 ,128 ),(128 *s0 ,128 ,1 ))
withtorch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
buf0 =empty_strided_cuda ((1 ,s0 ,1 ,64 ),(64 *s0 ,64 ,64 ,1 ),torch .float32 )

triton_poi_fused__adaptive_avg_pool2d_0_xnumel =64 *s0 
stream0 =get_raw_stream (0 )
triton_poi_fused__adaptive_avg_pool2d_0 [grid (triton_poi_fused__adaptive_avg_pool2d_0_xnumel )](arg2_1 ,buf0 ,192 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
delarg2_1 
return (reinterpret_tensor (buf0 ,(1 ,64 ,s0 ),(64 *s0 ,1 ,64 ),0 ),)

defbenchmark_compiled_module (times =10 ,repeat =10 ):
    fromtorch ._dynamo .testingimportrand_strided 
fromtorch ._inductor .utilsimportprint_performance 
arg0_1 =3 
arg1_1 =128 
arg2_1 =rand_strided ((1 ,3 ,128 ),(384 ,128 ,1 ),device ='cuda:0',dtype =torch .float32 )
fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ])
returnprint_performance (fn ,times =times ,repeat =repeat )

if__name__ =="__main__":
    fromtorch ._inductor .wrapper_benchmarkimportcompiled_module_main 
compiled_module_main ('None',benchmark_compiled_module )
