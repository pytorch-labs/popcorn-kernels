
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
fromtorch ._inductor .runtime .triton_helpersimportmathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_red_fused__log_softmax_0 (in_ptr0 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =2 
rnumel =r0_numel 
RBLOCK :tl .constexpr =R0_BLOCK 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
xmask =xindex <xnumel 
r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
rbase =r0_base 
x0 =xindex 
_tmp41 =tl .full ([XBLOCK ,R0_BLOCK ],float ("-inf"),tl .float32 )
forr0_offsetinrange (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
r0_mask =r0_index <r0_numel 
roffset =r0_offset 
rindex =r0_index 
r0_1 =r0_index 
tmp0 =r0_1 +x0 *((1 +ks0 *ks1 *ks2 )//2 )
tmp1 =ks0 *ks1 *ks2 
tmp2 =tmp0 <tmp1 
tmp3 =tl .load (in_ptr0 +(((r0_1 +x0 *((1 +ks0 *ks1 *ks2 )//2 ))%(ks0 *ks1 *ks2 ))),r0_mask &tmp2 &xmask ,eviction_policy ='evict_last',other =0.0 )
tmp4 =0.0 
tmp5 =tmp3 >tmp4 
tmp6 =0.01 
tmp7 =tmp3 *tmp6 
tmp8 =tl .where (tmp5 ,tmp3 ,tmp7 )
tmp9 =tl_math .abs (tmp8 )
tmp10 =0.5 
tmp11 =tmp9 >tmp10 
tmp12 =tl .full ([1 ,1 ],0 ,tl .int32 )
tmp13 =tmp12 <tmp8 
tmp14 =tmp13 .to (tl .int8 )
tmp15 =tmp8 <tmp12 
tmp16 =tmp15 .to (tl .int8 )
tmp17 =tmp14 -tmp16 
tmp18 =tmp17 .to (tmp8 .dtype )
tmp19 =tmp18 *tmp10 
tmp20 =tmp8 -tmp19 
tmp21 =tmp8 *tmp4 
tmp22 =tl .where (tmp11 ,tmp20 ,tmp21 )
tmp23 =tmp22 >tmp4 
tmp24 =tmp22 *tmp6 
tmp25 =tl .where (tmp23 ,tmp22 ,tmp24 )
tmp26 =tl_math .abs (tmp25 )
tmp27 =tmp26 >tmp10 
tmp28 =tmp12 <tmp25 
tmp29 =tmp28 .to (tl .int8 )
tmp30 =tmp25 <tmp12 
tmp31 =tmp30 .to (tl .int8 )
tmp32 =tmp29 -tmp31 
tmp33 =tmp32 .to (tmp25 .dtype )
tmp34 =tmp33 *tmp10 
tmp35 =tmp25 -tmp34 
tmp36 =tmp25 *tmp4 
tmp37 =tl .where (tmp27 ,tmp35 ,tmp36 )
tmp38 =tl .full (tmp37 .shape ,float ("-inf"),tmp37 .dtype )
tmp39 =tl .where (tmp2 ,tmp37 ,tmp38 )
tmp40 =tl .broadcast_to (tmp39 ,[XBLOCK ,R0_BLOCK ])
tmp42 =triton_helpers .maximum (_tmp41 ,tmp40 )
_tmp41 =tl .where (r0_mask &xmask ,tmp42 ,_tmp41 )
tmp41 =triton_helpers .max2 (_tmp41 ,1 )[:,None ]
tl .store (out_ptr0 +(x0 ),tmp41 ,xmask )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
fromtorch ._inductor .runtime .triton_helpersimportmathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_per_fused__log_softmax_1 (in_ptr0 ,out_ptr0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    xnumel =1 
r0_numel =2 
R0_BLOCK :tl .constexpr =2 
rnumel =r0_numel 
RBLOCK :tl .constexpr =R0_BLOCK 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
xmask =tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
r0_offset =0 
r0_mask =tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
roffset =r0_offset 
rindex =r0_index 
r0_0 =r0_index 
tmp0 =tl .load (in_ptr0 +(r0_0 ),None )
tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
tmp3 =triton_helpers .max2 (tmp1 ,1 )[:,None ]
tl .store (out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp3 ,None )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
fromtorch ._inductor .runtime .triton_helpersimportmathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_red_fused__log_softmax_2 (in_ptr0 ,in_ptr1 ,out_ptr0 ,ks0 ,ks1 ,ks2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =2 
rnumel =r0_numel 
RBLOCK :tl .constexpr =R0_BLOCK 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
xmask =xindex <xnumel 
r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
rbase =r0_base 
x0 =xindex 
_tmp44 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
forr0_offsetinrange (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
r0_mask =r0_index <r0_numel 
roffset =r0_offset 
rindex =r0_index 
r0_1 =r0_index 
tmp0 =r0_1 +x0 *((1 +ks0 *ks1 *ks2 )//2 )
tmp1 =ks0 *ks1 *ks2 
tmp2 =tmp0 <tmp1 
tmp3 =tl .load (in_ptr0 +(((r0_1 +x0 *((1 +ks0 *ks1 *ks2 )//2 ))%(ks0 *ks1 *ks2 ))),r0_mask &tmp2 &xmask ,eviction_policy ='evict_last',other =0.0 )
tmp4 =0.0 
tmp5 =tmp3 >tmp4 
tmp6 =0.01 
tmp7 =tmp3 *tmp6 
tmp8 =tl .where (tmp5 ,tmp3 ,tmp7 )
tmp9 =tl_math .abs (tmp8 )
tmp10 =0.5 
tmp11 =tmp9 >tmp10 
tmp12 =tl .full ([1 ,1 ],0 ,tl .int32 )
tmp13 =tmp12 <tmp8 
tmp14 =tmp13 .to (tl .int8 )
tmp15 =tmp8 <tmp12 
tmp16 =tmp15 .to (tl .int8 )
tmp17 =tmp14 -tmp16 
tmp18 =tmp17 .to (tmp8 .dtype )
tmp19 =tmp18 *tmp10 
tmp20 =tmp8 -tmp19 
tmp21 =tmp8 *tmp4 
tmp22 =tl .where (tmp11 ,tmp20 ,tmp21 )
tmp23 =tmp22 >tmp4 
tmp24 =tmp22 *tmp6 
tmp25 =tl .where (tmp23 ,tmp22 ,tmp24 )
tmp26 =tl_math .abs (tmp25 )
tmp27 =tmp26 >tmp10 
tmp28 =tmp12 <tmp25 
tmp29 =tmp28 .to (tl .int8 )
tmp30 =tmp25 <tmp12 
tmp31 =tmp30 .to (tl .int8 )
tmp32 =tmp29 -tmp31 
tmp33 =tmp32 .to (tmp25 .dtype )
tmp34 =tmp33 *tmp10 
tmp35 =tmp25 -tmp34 
tmp36 =tmp25 *tmp4 
tmp37 =tl .where (tmp27 ,tmp35 ,tmp36 )
tmp38 =tl .load (in_ptr1 +(tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .int32 )),tmp2 ,eviction_policy ='evict_last',other =0.0 )
tmp39 =tmp37 -tmp38 
tmp40 =tl_math .exp (tmp39 )
tmp41 =tl .full (tmp40 .shape ,0 ,tmp40 .dtype )
tmp42 =tl .where (tmp2 ,tmp40 ,tmp41 )
tmp43 =tl .broadcast_to (tmp42 ,[XBLOCK ,R0_BLOCK ])
tmp45 =_tmp44 +tmp43 
_tmp44 =tl .where (r0_mask &xmask ,tmp45 ,_tmp44 )
tmp44 =tl .sum (_tmp44 ,1 )[:,None ]
tl .store (out_ptr0 +(x0 ),tmp44 ,xmask )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
fromtorch ._inductor .runtime .triton_helpersimportmathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_per_fused__log_softmax_nll_loss_forward_randint_3 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,load_seed_offset ,ks1 ,ks2 ,ks3 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    xnumel =1 
r0_numel =2 
R0_BLOCK :tl .constexpr =2 
rnumel =r0_numel 
RBLOCK :tl .constexpr =R0_BLOCK 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
xmask =tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
r0_offset =0 
r0_mask =tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
roffset =r0_offset 
rindex =r0_index 
r0_0 =r0_index 
tmp0 =tl .load (in_ptr0 +(r0_0 ),None )
tmp51 =tl .load (in_out_ptr0 +(0 ))
tmp52 =tl .broadcast_to (tmp51 ,[XBLOCK ,1 ])
tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
tmp3 =tl .sum (tmp1 ,1 )[:,None ]
tmp4 =tl .load (in_ptr1 +load_seed_offset )
tmp5 =tl .full ([1 ,1 ],0 ,tl .int32 )
tmp6 =tl .full ([1 ,1 ],0 ,tl .int64 )
tmp7 =tl .full ([1 ,1 ],10 ,tl .int64 )
tmp8 =triton_helpers .randint64 (tmp4 ,(tmp5 ).to (tl .uint32 ),tmp6 ,tmp7 )
tmp9 =tl .full ([1 ,1 ],-100 ,tl .int64 )
tmp10 =tmp8 !=tmp9 
tmp11 =tl .where (tmp10 ,tmp8 ,tmp6 )
tmp12 =ks1 *ks2 *ks3 
tmp13 =tmp11 +tmp12 
tmp14 =tmp11 <0 
tmp15 =tl .where (tmp14 ,tmp13 ,tmp11 )
tl .device_assert ((0 <=tmp15 )&(tmp15 <ks1 *ks2 *ks3 ),"index out of bounds: 0 <= tmp15 < ks1*ks2*ks3")
tmp17 =tl .load (in_ptr2 +((tmp15 %(ks1 *ks2 *ks3 ))),None ,eviction_policy ='evict_last')
tmp18 =0.0 
tmp19 =tmp17 >tmp18 
tmp20 =0.01 
tmp21 =tmp17 *tmp20 
tmp22 =tl .where (tmp19 ,tmp17 ,tmp21 )
tmp23 =tl_math .abs (tmp22 )
tmp24 =0.5 
tmp25 =tmp23 >tmp24 
tmp26 =tmp5 <tmp22 
tmp27 =tmp26 .to (tl .int8 )
tmp28 =tmp22 <tmp5 
tmp29 =tmp28 .to (tl .int8 )
tmp30 =tmp27 -tmp29 
tmp31 =tmp30 .to (tmp22 .dtype )
tmp32 =tmp31 *tmp24 
tmp33 =tmp22 -tmp32 
tmp34 =tmp22 *tmp18 
tmp35 =tl .where (tmp25 ,tmp33 ,tmp34 )
tmp36 =tmp35 >tmp18 
tmp37 =tmp35 *tmp20 
tmp38 =tl .where (tmp36 ,tmp35 ,tmp37 )
tmp39 =tl_math .abs (tmp38 )
tmp40 =tmp39 >tmp24 
tmp41 =tmp5 <tmp38 
tmp42 =tmp41 .to (tl .int8 )
tmp43 =tmp38 <tmp5 
tmp44 =tmp43 .to (tl .int8 )
tmp45 =tmp42 -tmp44 
tmp46 =tmp45 .to (tmp38 .dtype )
tmp47 =tmp46 *tmp24 
tmp48 =tmp38 -tmp47 
tmp49 =tmp38 *tmp18 
tmp50 =tl .where (tmp40 ,tmp48 ,tmp49 )
tmp53 =tmp50 -tmp52 
tmp54 =tl_math .log (tmp3 )
tmp55 =tmp53 -tmp54 
tmp56 =-tmp55 
tmp57 =tl .where (tmp10 ,tmp56 ,tmp18 )
tmp58 =tmp10 .to (tl .int32 )
tmp59 =tmp58 .to (tl .float32 )
tmp60 =tmp57 /tmp59 
tl .debug_barrier ()
tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp60 ,None )

defcall (args ):
    arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 =args 
args .clear ()
s0 =arg0_1 
s1 =arg1_1 
s2 =arg2_1 
assert_size_stride (arg3_1 ,(1 ,s0 ,s1 ,s2 ),(s0 *s1 *s2 ,s1 *s2 ,s2 ,1 ))
withtorch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
buf0 =empty_strided_cuda ((1 ,),(1 ,),torch .int64 )

aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[1 ],out =buf0 )
buf1 =empty_strided_cuda ((1 ,1 ,2 ),(2 ,2 ,1 ),torch .float32 )

triton_red_fused__log_softmax_0_r0_numel =(1 +s0 *s1 *s2 )//2 
stream0 =get_raw_stream (0 )
triton_red_fused__log_softmax_0 [grid (2 )](arg3_1 ,buf1 ,3 ,64 ,64 ,2 ,6144 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
buf2 =empty_strided_cuda ((1 ,1 ),(1 ,1 ),torch .float32 )

stream0 =get_raw_stream (0 )
triton_per_fused__log_softmax_1 [grid (1 )](buf1 ,buf2 ,1 ,2 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
buf3 =buf1 ;delbuf1 

triton_red_fused__log_softmax_2_r0_numel =(1 +s0 *s1 *s2 )//2 
stream0 =get_raw_stream (0 )
triton_red_fused__log_softmax_2 [grid (2 )](arg3_1 ,buf2 ,buf3 ,3 ,64 ,64 ,2 ,6144 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
buf5 =reinterpret_tensor (buf2 ,(),(),0 );delbuf2 

stream0 =get_raw_stream (0 )
triton_per_fused__log_softmax_nll_loss_forward_randint_3 [grid (1 )](buf5 ,buf3 ,buf0 ,arg3_1 ,0 ,3 ,64 ,64 ,1 ,2 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
delarg3_1 
delbuf0 
delbuf3 
return (buf5 ,)

defbenchmark_compiled_module (times =10 ,repeat =10 ):
    fromtorch ._dynamo .testingimportrand_strided 
fromtorch ._inductor .utilsimportprint_performance 
arg0_1 =3 
arg1_1 =64 
arg2_1 =64 
arg3_1 =rand_strided ((1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
fn =lambda :call ([arg0_1 ,arg1_1 ,arg2_1 ,arg3_1 ])
returnprint_performance (fn ,times =times ,repeat =repeat )

if__name__ =="__main__":
    fromtorch ._inductor .wrapper_benchmarkimportcompiled_module_main 
compiled_module_main ('None',benchmark_compiled_module )
