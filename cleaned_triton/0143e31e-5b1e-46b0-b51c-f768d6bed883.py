
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
deftriton_poi_fused_bernoulli_0 (in_ptr0 ,out_ptr0 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =3 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
xmask =xindex <xnumel 
x0 =xindex 
tmp0 =tl .load (in_ptr0 +load_seed_offset )
tmp1 =x0 
tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
tl .store (out_ptr0 +(x0 ),tmp2 ,xmask )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_poi_fused_bernoulli_1 (in_ptr0 ,out_ptr0 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =3 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
xmask =xindex <xnumel 
x0 =xindex 
tmp0 =tl .load (in_ptr0 +load_seed_offset )
tmp1 =x0 
tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
tl .store (out_ptr0 +(x0 ),tmp2 ,xmask )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_poi_fused__to_copy_add_bernoulli_hardsigmoid_mul_relu_2 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =12288 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
xmask =tl .full ([XBLOCK ],True ,tl .int1 )
x2 =xindex 
x1 =xindex //4096 
tmp0 =tl .load (in_ptr0 +(x2 ),None )
tmp11 =tl .load (in_ptr1 +(x1 ),None ,eviction_policy ='evict_last')
tmp30 =tl .load (in_ptr2 +(x1 ),None ,eviction_policy ='evict_last')
tmp1 =3.0 
tmp2 =tmp0 +tmp1 
tmp3 =0.0 
tmp4 =triton_helpers .maximum (tmp2 ,tmp3 )
tmp5 =6.0 
tmp6 =triton_helpers .minimum (tmp4 ,tmp5 )
tmp7 =0.16666666666666666 
tmp8 =tmp6 *tmp7 
tmp9 =tl .full ([1 ],0 ,tl .int32 )
tmp10 =triton_helpers .maximum (tmp9 ,tmp8 )
tmp12 =0.5 
tmp13 =tmp11 <tmp12 
tmp14 =tmp13 .to (tl .float32 )
tmp15 =0.8864048946659319 
tmp16 =tmp14 *tmp15 
tmp17 =tmp10 *tmp16 
tmp18 =-1.0 
tmp19 =tmp14 +tmp18 
tmp20 =1.558387861036063 
tmp21 =tmp19 *tmp20 
tmp22 =0.7791939305180315 
tmp23 =tmp21 +tmp22 
tmp24 =tmp17 +tmp23 
tmp25 =tmp24 +tmp1 
tmp26 =triton_helpers .maximum (tmp25 ,tmp3 )
tmp27 =triton_helpers .minimum (tmp26 ,tmp5 )
tmp28 =tmp27 *tmp7 
tmp29 =triton_helpers .maximum (tmp9 ,tmp28 )
tmp31 =tmp30 <tmp12 
tmp32 =tmp31 .to (tl .float32 )
tmp33 =tmp32 *tmp15 
tmp34 =tmp29 *tmp33 
tmp35 =tmp32 +tmp18 
tmp36 =tmp35 *tmp20 
tmp37 =tmp36 +tmp22 
tmp38 =tmp34 +tmp37 
tl .store (in_out_ptr0 +(x2 ),tmp38 ,None )

defcall (args ):
    arg0_1 ,=args 
args .clear ()
assert_size_stride (arg0_1 ,(1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ))
withtorch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
buf0 =empty_strided_cuda ((2 ,),(1 ,),torch .int64 )

aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[2 ],out =buf0 )
buf1 =empty_strided_cuda ((1 ,3 ,1 ,1 ),(3 ,1 ,3 ,3 ),torch .float32 )

stream0 =get_raw_stream (0 )
triton_poi_fused_bernoulli_0 [grid (3 )](buf0 ,buf1 ,0 ,3 ,XBLOCK =4 ,num_warps =1 ,num_stages =1 )
buf2 =empty_strided_cuda ((1 ,3 ,1 ,1 ),(3 ,1 ,3 ,3 ),torch .float32 )

stream0 =get_raw_stream (0 )
triton_poi_fused_bernoulli_1 [grid (3 )](buf0 ,buf2 ,1 ,3 ,XBLOCK =4 ,num_warps =1 ,num_stages =1 )
delbuf0 
buf3 =empty_strided_cuda ((1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ),torch .float32 )
buf4 =buf3 ;delbuf3 

stream0 =get_raw_stream (0 )
triton_poi_fused__to_copy_add_bernoulli_hardsigmoid_mul_relu_2 [grid (12288 )](buf4 ,arg0_1 ,buf1 ,buf2 ,12288 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
delarg0_1 
delbuf1 
delbuf2 
return (reinterpret_tensor (buf4 ,(1 ,12288 ),(12288 ,1 ),0 ),)

defbenchmark_compiled_module (times =10 ,repeat =10 ):
    fromtorch ._dynamo .testingimportrand_strided 
fromtorch ._inductor .utilsimportprint_performance 
arg0_1 =rand_strided ((1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
fn =lambda :call ([arg0_1 ])
returnprint_performance (fn ,times =times ,repeat =repeat )

if__name__ =="__main__":
    fromtorch ._inductor .wrapper_benchmarkimportcompiled_module_main 
compiled_module_main ('None',benchmark_compiled_module )
