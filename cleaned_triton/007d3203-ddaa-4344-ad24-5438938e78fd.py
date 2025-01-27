
importtorch 
fromtorch ._inductor .select_algorithmimportextern_kernels 
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
fromtorch ._inductor .runtime .triton_helpersimportlibdevice ,mathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_poi_fused_constant_pad_nd_0 (in_ptr0 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =13068 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
xmask =xindex <xnumel 
x1 =((xindex //66 )%66 )
x0 =(xindex %66 )
x2 =xindex //4356 
x4 =xindex 
tmp0 =(-1 )+x1 
tmp1 =tl .full ([1 ],0 ,tl .int64 )
tmp2 =tmp0 >=tmp1 
tmp3 =tl .full ([1 ],64 ,tl .int64 )
tmp4 =tmp0 <tmp3 
tmp5 =(-1 )+x0 
tmp6 =tmp5 >=tmp1 
tmp7 =tmp5 <tmp3 
tmp8 =tmp2 &tmp4 
tmp9 =tmp8 &tmp6 
tmp10 =tmp9 &tmp7 
tmp11 =tl .load (in_ptr0 +((-65 )+x0 +64 *x1 +4096 *x2 ),tmp10 &xmask ,other =0.5 )
tl .store (out_ptr0 +(x4 ),tmp11 ,xmask )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
fromtorch ._inductor .runtime .triton_helpersimportlibdevice ,mathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_red_fused__native_batch_norm_legit_functional_convolution_mish_1 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,in_ptr3 ,in_ptr4 ,out_ptr0 ,out_ptr2 ,out_ptr4 ,out_ptr6 ,out_ptr8 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =32 
r0_numel =4354 
rnumel =r0_numel 
RBLOCK :tl .constexpr =R0_BLOCK 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
xmask =xindex <xnumel 
r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
rbase =r0_base 
x0 =xindex 
tmp1 =tl .load (in_ptr0 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp4_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
tmp4_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
tmp4_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
forr0_offsetinrange (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
r0_mask =r0_index <r0_numel 
roffset =r0_offset 
rindex =r0_index 
r0_1 =r0_index 
tmp0 =tl .load (in_out_ptr0 +(r0_1 +4354 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
tmp2 =tmp0 +tmp1 
tmp3 =tl .broadcast_to (tmp2 ,[XBLOCK ,R0_BLOCK ])
tmp4_mean_next ,tmp4_m2_next ,tmp4_weight_next =triton_helpers .welford_reduce (
tmp3 ,tmp4_mean ,tmp4_m2 ,tmp4_weight ,roffset ==0 
)
tmp4_mean =tl .where (r0_mask &xmask ,tmp4_mean_next ,tmp4_mean )
tmp4_m2 =tl .where (r0_mask &xmask ,tmp4_m2_next ,tmp4_m2 )
tmp4_weight =tl .where (r0_mask &xmask ,tmp4_weight_next ,tmp4_weight )
tl .store (in_out_ptr0 +(r0_1 +4354 *x0 ),tmp2 ,r0_mask &xmask )
tmp7 ,tmp8 ,tmp9 =triton_helpers .welford (tmp4_mean ,tmp4_m2 ,tmp4_weight ,1 )
tmp4 =tmp7 [:,None ]
tmp5 =tmp8 [:,None ]
tmp6 =tmp9 [:,None ]
tl .store (out_ptr0 +(x0 ),tmp4 ,xmask )
tmp19 =tl .load (in_ptr1 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp24 =tl .load (in_ptr2 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp10 =4354.0 
tmp11 =tmp5 /tmp10 
tmp12 =1e-05 
tmp13 =tmp11 +tmp12 
tmp14 =libdevice .rsqrt (tmp13 )
tmp15 =1.000229726625316 
tmp16 =tmp11 *tmp15 
tmp17 =0.1 
tmp18 =tmp16 *tmp17 
tmp20 =0.9 
tmp21 =tmp19 *tmp20 
tmp22 =tmp18 +tmp21 
tmp23 =tmp4 *tmp17 
tmp25 =tmp24 *tmp20 
tmp26 =tmp23 +tmp25 
tl .store (out_ptr2 +(x0 ),tmp14 ,xmask )
tl .store (out_ptr4 +(x0 ),tmp22 ,xmask )
tl .store (out_ptr6 +(x0 ),tmp26 ,xmask )
tmp30 =tl .load (in_ptr3 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp32 =tl .load (in_ptr4 +(x0 ),xmask ,eviction_policy ='evict_last')
forr0_offsetinrange (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
r0_mask =r0_index <r0_numel 
roffset =r0_offset 
rindex =r0_index 
r0_1 =r0_index 
tmp27 =tl .load (in_out_ptr0 +(r0_1 +4354 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
tmp28 =tmp27 -tmp4 
tmp29 =tmp28 *tmp14 
tmp31 =tmp29 *tmp30 
tmp33 =tmp31 +tmp32 
tmp34 =20.0 
tmp35 =tmp33 >tmp34 
tmp36 =tl_math .exp (tmp33 )
tmp37 =libdevice .log1p (tmp36 )
tmp38 =tl .where (tmp35 ,tmp33 ,tmp37 )
tmp39 =libdevice .tanh (tmp38 )
tmp40 =tmp33 *tmp39 
tl .store (out_ptr8 +(r0_1 +4354 *x0 ),tmp40 ,r0_mask &xmask )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
fromtorch ._inductor .runtime .triton_helpersimportlibdevice ,mathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_red_fused__native_batch_norm_legit_functional_convolution_hardswish_2 (in_out_ptr0 ,in_out_ptr1 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,in_ptr3 ,in_ptr4 ,out_ptr0 ,out_ptr2 ,out_ptr4 ,out_ptr6 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =64 
r0_numel =4352 
rnumel =r0_numel 
RBLOCK :tl .constexpr =R0_BLOCK 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
xmask =xindex <xnumel 
r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
rbase =r0_base 
x0 =xindex 
tmp1 =tl .load (in_ptr0 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp4_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
tmp4_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
tmp4_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
forr0_offsetinrange (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
r0_mask =r0_index <r0_numel 
roffset =r0_offset 
rindex =r0_index 
r0_1 =r0_index 
tmp0 =tl .load (in_out_ptr0 +(r0_1 +4352 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
tmp2 =tmp0 +tmp1 
tmp3 =tl .broadcast_to (tmp2 ,[XBLOCK ,R0_BLOCK ])
tmp4_mean_next ,tmp4_m2_next ,tmp4_weight_next =triton_helpers .welford_reduce (
tmp3 ,tmp4_mean ,tmp4_m2 ,tmp4_weight ,roffset ==0 
)
tmp4_mean =tl .where (r0_mask &xmask ,tmp4_mean_next ,tmp4_mean )
tmp4_m2 =tl .where (r0_mask &xmask ,tmp4_m2_next ,tmp4_m2 )
tmp4_weight =tl .where (r0_mask &xmask ,tmp4_weight_next ,tmp4_weight )
tl .store (in_out_ptr0 +(r0_1 +4352 *x0 ),tmp2 ,r0_mask &xmask )
tmp7 ,tmp8 ,tmp9 =triton_helpers .welford (tmp4_mean ,tmp4_m2 ,tmp4_weight ,1 )
tmp4 =tmp7 [:,None ]
tmp5 =tmp8 [:,None ]
tmp6 =tmp9 [:,None ]
tl .store (out_ptr0 +(x0 ),tmp4 ,xmask )
tmp19 =tl .load (in_ptr1 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp24 =tl .load (in_ptr2 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp10 =4352.0 
tmp11 =tmp5 /tmp10 
tmp12 =1e-05 
tmp13 =tmp11 +tmp12 
tmp14 =libdevice .rsqrt (tmp13 )
tmp15 =1.0002298322224776 
tmp16 =tmp11 *tmp15 
tmp17 =0.1 
tmp18 =tmp16 *tmp17 
tmp20 =0.9 
tmp21 =tmp19 *tmp20 
tmp22 =tmp18 +tmp21 
tmp23 =tmp4 *tmp17 
tmp25 =tmp24 *tmp20 
tmp26 =tmp23 +tmp25 
tl .store (out_ptr2 +(x0 ),tmp14 ,xmask )
tl .store (out_ptr4 +(x0 ),tmp22 ,xmask )
tl .store (out_ptr6 +(x0 ),tmp26 ,xmask )
tmp30 =tl .load (in_ptr3 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp32 =tl .load (in_ptr4 +(x0 ),xmask ,eviction_policy ='evict_last')
forr0_offsetinrange (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
r0_mask =r0_index <r0_numel 
roffset =r0_offset 
rindex =r0_index 
r0_1 =r0_index 
tmp27 =tl .load (in_out_ptr0 +(r0_1 +4352 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
tmp28 =tmp27 -tmp4 
tmp29 =tmp28 *tmp14 
tmp31 =tmp29 *tmp30 
tmp33 =tmp31 +tmp32 
tmp34 =3.0 
tmp35 =tmp33 +tmp34 
tmp36 =0.0 
tmp37 =triton_helpers .maximum (tmp35 ,tmp36 )
tmp38 =6.0 
tmp39 =triton_helpers .minimum (tmp37 ,tmp38 )
tmp40 =tmp33 *tmp39 
tmp41 =0.16666666666666666 
tmp42 =tmp40 *tmp41 
tl .store (in_out_ptr1 +(r0_1 +4352 *x0 ),tmp42 ,r0_mask &xmask )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
fromtorch ._inductor .runtime .triton_helpersimportlibdevice ,mathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_red_fused__native_batch_norm_legit_functional_convolution_mish_3 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,in_ptr3 ,in_ptr4 ,out_ptr0 ,out_ptr2 ,out_ptr4 ,out_ptr6 ,out_ptr8 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =128 
r0_numel =4350 
rnumel =r0_numel 
RBLOCK :tl .constexpr =R0_BLOCK 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
xmask =xindex <xnumel 
r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
rbase =r0_base 
x0 =xindex 
tmp1 =tl .load (in_ptr0 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp4_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
tmp4_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
tmp4_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
forr0_offsetinrange (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
r0_mask =r0_index <r0_numel 
roffset =r0_offset 
rindex =r0_index 
r0_1 =r0_index 
tmp0 =tl .load (in_out_ptr0 +(r0_1 +4350 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
tmp2 =tmp0 +tmp1 
tmp3 =tl .broadcast_to (tmp2 ,[XBLOCK ,R0_BLOCK ])
tmp4_mean_next ,tmp4_m2_next ,tmp4_weight_next =triton_helpers .welford_reduce (
tmp3 ,tmp4_mean ,tmp4_m2 ,tmp4_weight ,roffset ==0 
)
tmp4_mean =tl .where (r0_mask &xmask ,tmp4_mean_next ,tmp4_mean )
tmp4_m2 =tl .where (r0_mask &xmask ,tmp4_m2_next ,tmp4_m2 )
tmp4_weight =tl .where (r0_mask &xmask ,tmp4_weight_next ,tmp4_weight )
tl .store (in_out_ptr0 +(r0_1 +4350 *x0 ),tmp2 ,r0_mask &xmask )
tmp7 ,tmp8 ,tmp9 =triton_helpers .welford (tmp4_mean ,tmp4_m2 ,tmp4_weight ,1 )
tmp4 =tmp7 [:,None ]
tmp5 =tmp8 [:,None ]
tmp6 =tmp9 [:,None ]
tl .store (out_ptr0 +(x0 ),tmp4 ,xmask )
tmp19 =tl .load (in_ptr1 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp24 =tl .load (in_ptr2 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp10 =4350.0 
tmp11 =tmp5 /tmp10 
tmp12 =1e-05 
tmp13 =tmp11 +tmp12 
tmp14 =libdevice .rsqrt (tmp13 )
tmp15 =1.0002299379167625 
tmp16 =tmp11 *tmp15 
tmp17 =0.1 
tmp18 =tmp16 *tmp17 
tmp20 =0.9 
tmp21 =tmp19 *tmp20 
tmp22 =tmp18 +tmp21 
tmp23 =tmp4 *tmp17 
tmp25 =tmp24 *tmp20 
tmp26 =tmp23 +tmp25 
tl .store (out_ptr2 +(x0 ),tmp14 ,xmask )
tl .store (out_ptr4 +(x0 ),tmp22 ,xmask )
tl .store (out_ptr6 +(x0 ),tmp26 ,xmask )
tmp30 =tl .load (in_ptr3 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp32 =tl .load (in_ptr4 +(x0 ),xmask ,eviction_policy ='evict_last')
forr0_offsetinrange (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
r0_mask =r0_index <r0_numel 
roffset =r0_offset 
rindex =r0_index 
r0_1 =r0_index 
tmp27 =tl .load (in_out_ptr0 +(r0_1 +4350 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
tmp28 =tmp27 -tmp4 
tmp29 =tmp28 *tmp14 
tmp31 =tmp29 *tmp30 
tmp33 =tmp31 +tmp32 
tmp34 =20.0 
tmp35 =tmp33 >tmp34 
tmp36 =tl_math .exp (tmp33 )
tmp37 =libdevice .log1p (tmp36 )
tmp38 =tl .where (tmp35 ,tmp33 ,tmp37 )
tmp39 =libdevice .tanh (tmp38 )
tmp40 =tmp33 *tmp39 
tl .store (out_ptr8 +(r0_1 +4350 *x0 ),tmp40 ,r0_mask &xmask )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
fromtorch ._inductor .runtime .triton_helpersimportlibdevice ,mathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_red_fused__native_batch_norm_legit_functional_convolution_hardswish_4 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,in_ptr3 ,in_ptr4 ,out_ptr0 ,out_ptr2 ,out_ptr4 ,out_ptr6 ,out_ptr8 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =256 
r0_numel =4348 
rnumel =r0_numel 
RBLOCK :tl .constexpr =R0_BLOCK 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
xmask =xindex <xnumel 
r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
rbase =r0_base 
x0 =xindex 
tmp1 =tl .load (in_ptr0 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp4_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
tmp4_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
tmp4_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
forr0_offsetinrange (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
r0_mask =r0_index <r0_numel 
roffset =r0_offset 
rindex =r0_index 
r0_1 =r0_index 
tmp0 =tl .load (in_out_ptr0 +(r0_1 +4348 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
tmp2 =tmp0 +tmp1 
tmp3 =tl .broadcast_to (tmp2 ,[XBLOCK ,R0_BLOCK ])
tmp4_mean_next ,tmp4_m2_next ,tmp4_weight_next =triton_helpers .welford_reduce (
tmp3 ,tmp4_mean ,tmp4_m2 ,tmp4_weight ,roffset ==0 
)
tmp4_mean =tl .where (r0_mask &xmask ,tmp4_mean_next ,tmp4_mean )
tmp4_m2 =tl .where (r0_mask &xmask ,tmp4_m2_next ,tmp4_m2 )
tmp4_weight =tl .where (r0_mask &xmask ,tmp4_weight_next ,tmp4_weight )
tl .store (in_out_ptr0 +(r0_1 +4348 *x0 ),tmp2 ,r0_mask &xmask )
tmp7 ,tmp8 ,tmp9 =triton_helpers .welford (tmp4_mean ,tmp4_m2 ,tmp4_weight ,1 )
tmp4 =tmp7 [:,None ]
tmp5 =tmp8 [:,None ]
tmp6 =tmp9 [:,None ]
tl .store (out_ptr0 +(x0 ),tmp4 ,xmask )
tmp19 =tl .load (in_ptr1 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp24 =tl .load (in_ptr2 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp10 =4348.0 
tmp11 =tmp5 /tmp10 
tmp12 =1e-05 
tmp13 =tmp11 +tmp12 
tmp14 =libdevice .rsqrt (tmp13 )
tmp15 =1.0002300437083045 
tmp16 =tmp11 *tmp15 
tmp17 =0.1 
tmp18 =tmp16 *tmp17 
tmp20 =0.9 
tmp21 =tmp19 *tmp20 
tmp22 =tmp18 +tmp21 
tmp23 =tmp4 *tmp17 
tmp25 =tmp24 *tmp20 
tmp26 =tmp23 +tmp25 
tl .store (out_ptr2 +(x0 ),tmp14 ,xmask )
tl .store (out_ptr4 +(x0 ),tmp22 ,xmask )
tl .store (out_ptr6 +(x0 ),tmp26 ,xmask )
tmp30 =tl .load (in_ptr3 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp32 =tl .load (in_ptr4 +(x0 ),xmask ,eviction_policy ='evict_last')
forr0_offsetinrange (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
r0_mask =r0_index <r0_numel 
roffset =r0_offset 
rindex =r0_index 
r0_1 =r0_index 
tmp27 =tl .load (in_out_ptr0 +(r0_1 +4348 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
tmp28 =tmp27 -tmp4 
tmp29 =tmp28 *tmp14 
tmp31 =tmp29 *tmp30 
tmp33 =tmp31 +tmp32 
tmp34 =3.0 
tmp35 =tmp33 +tmp34 
tmp36 =0.0 
tmp37 =triton_helpers .maximum (tmp35 ,tmp36 )
tmp38 =6.0 
tmp39 =triton_helpers .minimum (tmp37 ,tmp38 )
tmp40 =tmp33 *tmp39 
tmp41 =0.16666666666666666 
tmp42 =tmp40 *tmp41 
tl .store (out_ptr8 +(r0_1 +4348 *x0 ),tmp42 ,r0_mask &xmask )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
fromtorch ._inductor .runtime .triton_helpersimportlibdevice ,mathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_red_fused__native_batch_norm_legit_functional_convolution_log_sigmoid_forward_5 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,in_ptr2 ,in_ptr3 ,in_ptr4 ,out_ptr0 ,out_ptr2 ,out_ptr4 ,out_ptr6 ,out_ptr8 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =512 
r0_numel =4346 
rnumel =r0_numel 
RBLOCK :tl .constexpr =R0_BLOCK 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
xmask =xindex <xnumel 
r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
rbase =r0_base 
x0 =xindex 
tmp1 =tl .load (in_ptr0 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp4_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
tmp4_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
tmp4_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
forr0_offsetinrange (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
r0_mask =r0_index <r0_numel 
roffset =r0_offset 
rindex =r0_index 
r0_1 =r0_index 
tmp0 =tl .load (in_out_ptr0 +(r0_1 +4346 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
tmp2 =tmp0 +tmp1 
tmp3 =tl .broadcast_to (tmp2 ,[XBLOCK ,R0_BLOCK ])
tmp4_mean_next ,tmp4_m2_next ,tmp4_weight_next =triton_helpers .welford_reduce (
tmp3 ,tmp4_mean ,tmp4_m2 ,tmp4_weight ,roffset ==0 
)
tmp4_mean =tl .where (r0_mask &xmask ,tmp4_mean_next ,tmp4_mean )
tmp4_m2 =tl .where (r0_mask &xmask ,tmp4_m2_next ,tmp4_m2 )
tmp4_weight =tl .where (r0_mask &xmask ,tmp4_weight_next ,tmp4_weight )
tl .store (in_out_ptr0 +(r0_1 +4346 *x0 ),tmp2 ,r0_mask &xmask )
tmp7 ,tmp8 ,tmp9 =triton_helpers .welford (tmp4_mean ,tmp4_m2 ,tmp4_weight ,1 )
tmp4 =tmp7 [:,None ]
tmp5 =tmp8 [:,None ]
tmp6 =tmp9 [:,None ]
tl .store (out_ptr0 +(x0 ),tmp4 ,xmask )
tmp19 =tl .load (in_ptr1 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp24 =tl .load (in_ptr2 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp10 =4346.0 
tmp11 =tmp5 /tmp10 
tmp12 =1e-05 
tmp13 =tmp11 +tmp12 
tmp14 =libdevice .rsqrt (tmp13 )
tmp15 =1.000230149597238 
tmp16 =tmp11 *tmp15 
tmp17 =0.1 
tmp18 =tmp16 *tmp17 
tmp20 =0.9 
tmp21 =tmp19 *tmp20 
tmp22 =tmp18 +tmp21 
tmp23 =tmp4 *tmp17 
tmp25 =tmp24 *tmp20 
tmp26 =tmp23 +tmp25 
tl .store (out_ptr2 +(x0 ),tmp14 ,xmask )
tl .store (out_ptr4 +(x0 ),tmp22 ,xmask )
tl .store (out_ptr6 +(x0 ),tmp26 ,xmask )
tmp30 =tl .load (in_ptr3 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp32 =tl .load (in_ptr4 +(x0 ),xmask ,eviction_policy ='evict_last')
forr0_offsetinrange (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
r0_mask =r0_index <r0_numel 
roffset =r0_offset 
rindex =r0_index 
r0_1 =r0_index 
tmp27 =tl .load (in_out_ptr0 +(r0_1 +4346 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
tmp28 =tmp27 -tmp4 
tmp29 =tmp28 *tmp14 
tmp31 =tmp29 *tmp30 
tmp33 =tmp31 +tmp32 
tmp34 =0.0 
tmp35 =triton_helpers .minimum (tmp34 ,tmp33 )
tmp36 =tl_math .abs (tmp33 )
tmp37 =-tmp36 
tmp38 =tl_math .exp (tmp37 )
tmp39 =libdevice .log1p (tmp38 )
tmp40 =tmp35 -tmp39 
tl .store (out_ptr8 +(r0_1 +4346 *x0 ),tmp40 ,r0_mask &xmask )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
fromtorch ._inductor .runtime .triton_helpersimportlibdevice ,mathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_poi_fused_add_6 (in_ptr0 ,out_ptr1 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =1 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
xmask =tl .full ([XBLOCK ],True ,tl .int1 )
tmp0 =tl .load (in_ptr0 +(0 ))
tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ])
tmp2 =tl .full ([1 ],1 ,tl .int64 )
tmp3 =tmp1 +tmp2 
tl .store (out_ptr1 +(tl .full ([XBLOCK ],0 ,tl .int32 )),tmp3 ,None )

defcall (args ):
    primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ,primals_8 ,primals_9 ,primals_10 ,primals_11 ,primals_12 ,primals_13 ,primals_14 ,primals_15 ,primals_16 ,primals_17 ,primals_18 ,primals_19 ,primals_20 ,primals_21 ,primals_22 ,primals_23 ,primals_24 ,primals_25 ,primals_26 ,primals_27 ,primals_28 ,primals_29 ,primals_30 ,primals_31 ,primals_32 ,primals_33 ,primals_34 ,primals_35 ,primals_36 =args 
args .clear ()
assert_size_stride (primals_1 ,(1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ))
assert_size_stride (primals_2 ,(32 ,3 ,3 ),(9 ,3 ,1 ))
assert_size_stride (primals_3 ,(32 ,),(1 ,))
assert_size_stride (primals_4 ,(),())
assert_size_stride (primals_5 ,(32 ,),(1 ,))
assert_size_stride (primals_6 ,(32 ,),(1 ,))
assert_size_stride (primals_7 ,(32 ,),(1 ,))
assert_size_stride (primals_8 ,(32 ,),(1 ,))
assert_size_stride (primals_9 ,(64 ,32 ,3 ),(96 ,3 ,1 ))
assert_size_stride (primals_10 ,(64 ,),(1 ,))
assert_size_stride (primals_11 ,(),())
assert_size_stride (primals_12 ,(64 ,),(1 ,))
assert_size_stride (primals_13 ,(64 ,),(1 ,))
assert_size_stride (primals_14 ,(64 ,),(1 ,))
assert_size_stride (primals_15 ,(64 ,),(1 ,))
assert_size_stride (primals_16 ,(128 ,64 ,3 ),(192 ,3 ,1 ))
assert_size_stride (primals_17 ,(128 ,),(1 ,))
assert_size_stride (primals_18 ,(),())
assert_size_stride (primals_19 ,(128 ,),(1 ,))
assert_size_stride (primals_20 ,(128 ,),(1 ,))
assert_size_stride (primals_21 ,(128 ,),(1 ,))
assert_size_stride (primals_22 ,(128 ,),(1 ,))
assert_size_stride (primals_23 ,(256 ,128 ,3 ),(384 ,3 ,1 ))
assert_size_stride (primals_24 ,(256 ,),(1 ,))
assert_size_stride (primals_25 ,(),())
assert_size_stride (primals_26 ,(256 ,),(1 ,))
assert_size_stride (primals_27 ,(256 ,),(1 ,))
assert_size_stride (primals_28 ,(256 ,),(1 ,))
assert_size_stride (primals_29 ,(256 ,),(1 ,))
assert_size_stride (primals_30 ,(512 ,256 ,3 ),(768 ,3 ,1 ))
assert_size_stride (primals_31 ,(512 ,),(1 ,))
assert_size_stride (primals_32 ,(),())
assert_size_stride (primals_33 ,(512 ,),(1 ,))
assert_size_stride (primals_34 ,(512 ,),(1 ,))
assert_size_stride (primals_35 ,(512 ,),(1 ,))
assert_size_stride (primals_36 ,(512 ,),(1 ,))
withtorch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
buf0 =empty_strided_cuda ((1 ,3 ,66 ,66 ),(13068 ,4356 ,66 ,1 ),torch .float32 )

stream0 =get_raw_stream (0 )
triton_poi_fused_constant_pad_nd_0 [grid (13068 )](primals_1 ,buf0 ,13068 ,XBLOCK =256 ,num_warps =4 ,num_stages =1 )
delprimals_1 

buf1 =extern_kernels .convolution (reinterpret_tensor (buf0 ,(1 ,3 ,4356 ),(0 ,4356 ,1 ),0 ),primals_2 ,stride =(1 ,),padding =(0 ,),dilation =(1 ,),transposed =False ,output_padding =(0 ,),groups =1 ,bias =None )
assert_size_stride (buf1 ,(1 ,32 ,4354 ),(139328 ,4354 ,1 ))
buf2 =buf1 ;delbuf1 
buf3 =empty_strided_cuda ((1 ,32 ,1 ),(32 ,1 ,1 ),torch .float32 )
buf6 =empty_strided_cuda ((1 ,32 ,1 ),(32 ,1 ,1 ),torch .float32 )
buf8 =empty_strided_cuda ((1 ,32 ,4354 ),(139328 ,4354 ,1 ),torch .float32 )

stream0 =get_raw_stream (0 )
triton_red_fused__native_batch_norm_legit_functional_convolution_mish_1 [grid (32 )](buf2 ,primals_3 ,primals_6 ,primals_5 ,primals_7 ,primals_8 ,buf3 ,buf6 ,primals_6 ,primals_5 ,buf8 ,32 ,4354 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
delprimals_3 
delprimals_5 
delprimals_6 

buf9 =extern_kernels .convolution (buf8 ,primals_9 ,stride =(1 ,),padding =(0 ,),dilation =(1 ,),transposed =False ,output_padding =(0 ,),groups =1 ,bias =None )
assert_size_stride (buf9 ,(1 ,64 ,4352 ),(278528 ,4352 ,1 ))
buf10 =buf9 ;delbuf9 
buf11 =empty_strided_cuda ((1 ,64 ,1 ),(64 ,1 ,1 ),torch .float32 )
buf14 =empty_strided_cuda ((1 ,64 ,1 ),(64 ,1 ,1 ),torch .float32 )
buf15 =empty_strided_cuda ((1 ,64 ,4352 ),(278528 ,4352 ,1 ),torch .float32 )
buf16 =buf15 ;delbuf15 

stream0 =get_raw_stream (0 )
triton_red_fused__native_batch_norm_legit_functional_convolution_hardswish_2 [grid (64 )](buf10 ,buf16 ,primals_10 ,primals_13 ,primals_12 ,primals_14 ,primals_15 ,buf11 ,buf14 ,primals_13 ,primals_12 ,64 ,4352 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
delprimals_10 
delprimals_12 
delprimals_13 

buf17 =extern_kernels .convolution (buf16 ,primals_16 ,stride =(1 ,),padding =(0 ,),dilation =(1 ,),transposed =False ,output_padding =(0 ,),groups =1 ,bias =None )
assert_size_stride (buf17 ,(1 ,128 ,4350 ),(556800 ,4350 ,1 ))
buf18 =buf17 ;delbuf17 
buf19 =empty_strided_cuda ((1 ,128 ,1 ),(128 ,1 ,1 ),torch .float32 )
buf22 =empty_strided_cuda ((1 ,128 ,1 ),(128 ,1 ,1 ),torch .float32 )
buf24 =empty_strided_cuda ((1 ,128 ,4350 ),(556800 ,4350 ,1 ),torch .float32 )

stream0 =get_raw_stream (0 )
triton_red_fused__native_batch_norm_legit_functional_convolution_mish_3 [grid (128 )](buf18 ,primals_17 ,primals_20 ,primals_19 ,primals_21 ,primals_22 ,buf19 ,buf22 ,primals_20 ,primals_19 ,buf24 ,128 ,4350 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
delprimals_17 
delprimals_19 
delprimals_20 

buf25 =extern_kernels .convolution (buf24 ,primals_23 ,stride =(1 ,),padding =(0 ,),dilation =(1 ,),transposed =False ,output_padding =(0 ,),groups =1 ,bias =None )
assert_size_stride (buf25 ,(1 ,256 ,4348 ),(1113088 ,4348 ,1 ))
buf26 =buf25 ;delbuf25 
buf27 =empty_strided_cuda ((1 ,256 ,1 ),(256 ,1 ,1 ),torch .float32 )
buf30 =empty_strided_cuda ((1 ,256 ,1 ),(256 ,1 ,1 ),torch .float32 )
buf32 =empty_strided_cuda ((1 ,256 ,4348 ),(1113088 ,4348 ,1 ),torch .float32 )

stream0 =get_raw_stream (0 )
triton_red_fused__native_batch_norm_legit_functional_convolution_hardswish_4 [grid (256 )](buf26 ,primals_24 ,primals_27 ,primals_26 ,primals_28 ,primals_29 ,buf27 ,buf30 ,primals_27 ,primals_26 ,buf32 ,256 ,4348 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
delprimals_24 
delprimals_26 
delprimals_27 

buf33 =extern_kernels .convolution (buf32 ,primals_30 ,stride =(1 ,),padding =(0 ,),dilation =(1 ,),transposed =False ,output_padding =(0 ,),groups =1 ,bias =None )
assert_size_stride (buf33 ,(1 ,512 ,4346 ),(2225152 ,4346 ,1 ))
buf34 =buf33 ;delbuf33 
buf35 =empty_strided_cuda ((1 ,512 ,1 ),(512 ,1 ,1 ),torch .float32 )
buf38 =empty_strided_cuda ((1 ,512 ,1 ),(512 ,1 ,1 ),torch .float32 )
buf40 =empty_strided_cuda ((1 ,512 ,4346 ),(2225152 ,4346 ,1 ),torch .float32 )

stream0 =get_raw_stream (0 )
triton_red_fused__native_batch_norm_legit_functional_convolution_log_sigmoid_forward_5 [grid (512 )](buf34 ,primals_31 ,primals_34 ,primals_33 ,primals_35 ,primals_36 ,buf35 ,buf38 ,primals_34 ,primals_33 ,buf40 ,512 ,4346 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
delprimals_31 
delprimals_33 
delprimals_34 

stream0 =get_raw_stream (0 )
triton_poi_fused_add_6 [grid (1 )](primals_4 ,primals_4 ,1 ,XBLOCK =1 ,num_warps =1 ,num_stages =1 )
delprimals_4 

stream0 =get_raw_stream (0 )
triton_poi_fused_add_6 [grid (1 )](primals_11 ,primals_11 ,1 ,XBLOCK =1 ,num_warps =1 ,num_stages =1 )
delprimals_11 

stream0 =get_raw_stream (0 )
triton_poi_fused_add_6 [grid (1 )](primals_18 ,primals_18 ,1 ,XBLOCK =1 ,num_warps =1 ,num_stages =1 )
delprimals_18 

stream0 =get_raw_stream (0 )
triton_poi_fused_add_6 [grid (1 )](primals_25 ,primals_25 ,1 ,XBLOCK =1 ,num_warps =1 ,num_stages =1 )
delprimals_25 

stream0 =get_raw_stream (0 )
triton_poi_fused_add_6 [grid (1 )](primals_32 ,primals_32 ,1 ,XBLOCK =1 ,num_warps =1 ,num_stages =1 )
delprimals_32 
return (buf40 ,primals_2 ,primals_7 ,primals_8 ,primals_9 ,primals_14 ,primals_15 ,primals_16 ,primals_21 ,primals_22 ,primals_23 ,primals_28 ,primals_29 ,primals_30 ,primals_35 ,primals_36 ,reinterpret_tensor (buf0 ,(1 ,3 ,4356 ),(13068 ,4356 ,1 ),0 ),buf2 ,buf3 ,buf6 ,buf8 ,buf10 ,buf11 ,buf14 ,buf16 ,buf18 ,buf19 ,buf22 ,buf24 ,buf26 ,buf27 ,buf30 ,buf32 ,buf34 ,buf35 ,buf38 ,)

defbenchmark_compiled_module (times =10 ,repeat =10 ):
    fromtorch ._dynamo .testingimportrand_strided 
fromtorch ._inductor .utilsimportprint_performance 
primals_1 =rand_strided ((1 ,3 ,64 ,64 ),(12288 ,4096 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
primals_2 =rand_strided ((32 ,3 ,3 ),(9 ,3 ,1 ),device ='cuda:0',dtype =torch .float32 )
primals_3 =rand_strided ((32 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_4 =rand_strided ((),(),device ='cuda:0',dtype =torch .int64 )
primals_5 =rand_strided ((32 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_6 =rand_strided ((32 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_7 =rand_strided ((32 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_8 =rand_strided ((32 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_9 =rand_strided ((64 ,32 ,3 ),(96 ,3 ,1 ),device ='cuda:0',dtype =torch .float32 )
primals_10 =rand_strided ((64 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_11 =rand_strided ((),(),device ='cuda:0',dtype =torch .int64 )
primals_12 =rand_strided ((64 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_13 =rand_strided ((64 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_14 =rand_strided ((64 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_15 =rand_strided ((64 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_16 =rand_strided ((128 ,64 ,3 ),(192 ,3 ,1 ),device ='cuda:0',dtype =torch .float32 )
primals_17 =rand_strided ((128 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_18 =rand_strided ((),(),device ='cuda:0',dtype =torch .int64 )
primals_19 =rand_strided ((128 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_20 =rand_strided ((128 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_21 =rand_strided ((128 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_22 =rand_strided ((128 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_23 =rand_strided ((256 ,128 ,3 ),(384 ,3 ,1 ),device ='cuda:0',dtype =torch .float32 )
primals_24 =rand_strided ((256 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_25 =rand_strided ((),(),device ='cuda:0',dtype =torch .int64 )
primals_26 =rand_strided ((256 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_27 =rand_strided ((256 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_28 =rand_strided ((256 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_29 =rand_strided ((256 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_30 =rand_strided ((512 ,256 ,3 ),(768 ,3 ,1 ),device ='cuda:0',dtype =torch .float32 )
primals_31 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_32 =rand_strided ((),(),device ='cuda:0',dtype =torch .int64 )
primals_33 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_34 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_35 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_36 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
fn =lambda :call ([primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ,primals_8 ,primals_9 ,primals_10 ,primals_11 ,primals_12 ,primals_13 ,primals_14 ,primals_15 ,primals_16 ,primals_17 ,primals_18 ,primals_19 ,primals_20 ,primals_21 ,primals_22 ,primals_23 ,primals_24 ,primals_25 ,primals_26 ,primals_27 ,primals_28 ,primals_29 ,primals_30 ,primals_31 ,primals_32 ,primals_33 ,primals_34 ,primals_35 ,primals_36 ])
returnprint_performance (fn ,times =times ,repeat =repeat )

if__name__ =="__main__":
    fromtorch ._inductor .wrapper_benchmarkimportcompiled_module_main 
compiled_module_main ('None',benchmark_compiled_module )
