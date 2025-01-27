
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
deftriton_red_fused__native_batch_norm_legit_functional_0 (in_ptr0 ,out_ptr0 ,out_ptr1 ,out_ptr2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =64 
r0_numel =8192 
rnumel =r0_numel 
RBLOCK :tl .constexpr =R0_BLOCK 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
xmask =xindex <xnumel 
r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
rbase =r0_base 
x0 =xindex 
tmp2_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
tmp2_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
tmp2_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
forr0_offsetinrange (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
r0_mask =r0_index <r0_numel 
roffset =r0_offset 
rindex =r0_index 
r0_1 =r0_index 
tmp0 =tl .load (in_ptr0 +(r0_1 +8192 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
tmp1 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
tmp2_mean_next ,tmp2_m2_next ,tmp2_weight_next =triton_helpers .welford_reduce (
tmp1 ,tmp2_mean ,tmp2_m2 ,tmp2_weight ,roffset ==0 
)
tmp2_mean =tl .where (r0_mask &xmask ,tmp2_mean_next ,tmp2_mean )
tmp2_m2 =tl .where (r0_mask &xmask ,tmp2_m2_next ,tmp2_m2 )
tmp2_weight =tl .where (r0_mask &xmask ,tmp2_weight_next ,tmp2_weight )
tmp5 ,tmp6 ,tmp7 =triton_helpers .welford (tmp2_mean ,tmp2_m2 ,tmp2_weight ,1 )
tmp2 =tmp5 [:,None ]
tmp3 =tmp6 [:,None ]
tmp4 =tmp7 [:,None ]
tl .store (out_ptr0 +(x0 ),tmp2 ,xmask )
tl .store (out_ptr1 +(x0 ),tmp3 ,xmask )
tl .store (out_ptr2 +(x0 ),tmp4 ,xmask )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
fromtorch ._inductor .runtime .triton_helpersimportlibdevice ,mathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_per_fused__native_batch_norm_legit_functional_1 (in_ptr0 ,in_ptr1 ,in_ptr2 ,in_ptr3 ,in_ptr4 ,out_ptr0 ,out_ptr1 ,out_ptr2 ,out_ptr4 ,out_ptr6 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    xnumel =16 
r0_numel =4 
R0_BLOCK :tl .constexpr =4 
rnumel =r0_numel 
RBLOCK :tl .constexpr =R0_BLOCK 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
xmask =xindex <xnumel 
r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
r0_offset =0 
r0_mask =tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
roffset =r0_offset 
rindex =r0_index 
r0_1 =r0_index 
x0 =xindex 
tmp0 =tl .load (in_ptr0 +(r0_1 +4 *x0 ),xmask ,other =0.0 )
tmp1 =tl .load (in_ptr1 +(r0_1 +4 *x0 ),xmask ,other =0.0 )
tmp2 =tl .load (in_ptr2 +(r0_1 +4 *x0 ),xmask ,other =0.0 )
tmp25 =tl .load (in_ptr3 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp30 =tl .load (in_ptr4 +(x0 ),xmask ,eviction_policy ='evict_last')
tmp3 =tl .broadcast_to (tmp0 ,[XBLOCK ,R0_BLOCK ])
tmp4 =tl .broadcast_to (tmp1 ,[XBLOCK ,R0_BLOCK ])
tmp5 =tl .broadcast_to (tmp2 ,[XBLOCK ,R0_BLOCK ])
tmp7 =tl .where (xmask ,tmp3 ,0 )
tmp8 =tl .where (xmask ,tmp4 ,0 )
tmp9 =tl .where (xmask ,tmp5 ,0 )
tmp10 ,tmp11 ,tmp12 =triton_helpers .welford (tmp7 ,tmp8 ,tmp9 ,1 )
tmp13 =tmp10 [:,None ]
tmp14 =tmp11 [:,None ]
tmp15 =tmp12 [:,None ]
tmp16 =32768.0 
tmp17 =tmp14 /tmp16 
tmp18 =1e-05 
tmp19 =tmp17 +tmp18 
tmp20 =libdevice .rsqrt (tmp19 )
tmp21 =1.000030518509476 
tmp22 =tmp17 *tmp21 
tmp23 =0.1 
tmp24 =tmp22 *tmp23 
tmp26 =0.9 
tmp27 =tmp25 *tmp26 
tmp28 =tmp24 +tmp27 
tmp29 =tmp13 *tmp23 
tmp31 =tmp30 *tmp26 
tmp32 =tmp29 +tmp31 
tl .store (out_ptr2 +(x0 ),tmp20 ,xmask )
tl .store (out_ptr4 +(x0 ),tmp28 ,xmask )
tl .store (out_ptr6 +(x0 ),tmp32 ,xmask )
tl .store (out_ptr0 +(x0 ),tmp13 ,xmask )
tl .store (out_ptr1 +(x0 ),tmp14 ,xmask )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
fromtorch ._inductor .runtime .triton_helpersimportlibdevice ,mathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_poi_fused_pixel_shuffle_2 (in_ptr0 ,in_ptr1 ,in_ptr2 ,in_ptr3 ,in_ptr4 ,out_ptr0 ,ynumel ,xnumel ,YBLOCK :tl .constexpr ,XBLOCK :tl .constexpr ):
    ynumel =262144 
xnumel =2 
yoffset =(tl .program_id (1 )+tl .program_id (2 )*tl .num_programs (1 ))*YBLOCK 
yindex =yoffset +tl .arange (0 ,YBLOCK )[None ,:]
ymask =yindex <ynumel 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
xmask =xindex <xnumel 
x5 =xindex 
y0 =(yindex %32 )
y1 =((yindex //32 )%2 )
y2 =((yindex //64 )%32 )
y6 =yindex //2048 
y3 =((yindex //2048 )%16 )
y4 =yindex //32768 
y7 =yindex 
tmp0 =tl .load (in_ptr0 +(y0 +32 *y2 +1024 *x5 +2048 *y1 +4096 *y6 +4096 *((y0 +32 *y2 +1024 *x5 +2048 *y1 )//4096 )),xmask &ymask ,eviction_policy ='evict_last')
tmp1 =tl .load (in_ptr1 +(2 *y4 +((y0 +32 *y2 +1024 *x5 +2048 *y1 +4096 *y3 )//32768 )),xmask &ymask ,eviction_policy ='evict_last')
tmp3 =tl .load (in_ptr2 +(2 *y4 +((y0 +32 *y2 +1024 *x5 +2048 *y1 +4096 *y3 )//32768 )),xmask &ymask ,eviction_policy ='evict_last')
tmp10 =tl .load (in_ptr3 +(2 *y4 +((y0 +32 *y2 +1024 *x5 +2048 *y1 +4096 *y3 )//32768 )),xmask &ymask ,eviction_policy ='evict_last')
tmp12 =tl .load (in_ptr4 +(2 *y4 +((y0 +32 *y2 +1024 *x5 +2048 *y1 +4096 *y3 )//32768 )),xmask &ymask ,eviction_policy ='evict_last')
tmp2 =tmp0 -tmp1 
tmp4 =32768.0 
tmp5 =tmp3 /tmp4 
tmp6 =1e-05 
tmp7 =tmp5 +tmp6 
tmp8 =libdevice .rsqrt (tmp7 )
tmp9 =tmp2 *tmp8 
tmp11 =tmp9 *tmp10 
tmp13 =tmp11 +tmp12 
tl .store (out_ptr0 +(x5 +2 *y7 ),tmp13 ,xmask &ymask )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
fromtorch ._inductor .runtime .triton_helpersimportlibdevice ,mathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_poi_fused__to_copy_3 (out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =1048576 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
xmask =tl .full ([XBLOCK ],True ,tl .int1 )
x0 =xindex 
tmp0 =0.0 
tl .store (out_ptr0 +(x0 ),tmp0 ,None )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
fromtorch ._inductor .runtime .triton_helpersimportlibdevice ,mathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_poi_fused__to_copy_4 (out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =524288 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
xmask =tl .full ([XBLOCK ],True ,tl .int1 )
x0 =xindex 
tmp0 =0.0 
tl .store (out_ptr0 +(x0 ),tmp0 ,None )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
fromtorch ._inductor .runtime .triton_helpersimportlibdevice ,mathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_red_fused__prelu_kernel_huber_loss_huber_loss_backward_log_sigmoid_forward_sub_tanh_5 (in_ptr0 ,in_ptr1 ,out_ptr0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =64 
r0_numel =8192 
rnumel =r0_numel 
RBLOCK :tl .constexpr =R0_BLOCK 
xoffset =tl .program_id (0 )*XBLOCK 
xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
xmask =xindex <xnumel 
r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
rbase =r0_base 
x0 =xindex 
tmp3 =tl .load (in_ptr1 +(0 ))
tmp4 =tl .broadcast_to (tmp3 ,[XBLOCK ,R0_BLOCK ])
_tmp39 =tl .full ([XBLOCK ,R0_BLOCK ],0 ,tl .float32 )
forr0_offsetinrange (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
r0_mask =r0_index <r0_numel 
roffset =r0_offset 
rindex =r0_index 
r0_1 =r0_index 
tmp0 =tl .load (in_ptr0 +(r0_1 +8192 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
tmp1 =0.0 
tmp2 =tmp0 >tmp1 
tmp5 =tmp4 *tmp0 
tmp6 =tl .where (tmp2 ,tmp0 ,tmp5 )
tmp7 =triton_helpers .minimum (tmp1 ,tmp6 )
tmp8 =tl_math .abs (tmp6 )
tmp9 =-tmp8 
tmp10 =tl_math .exp (tmp9 )
tmp11 =libdevice .log1p (tmp10 )
tmp12 =tmp7 -tmp11 
tmp13 =libdevice .tanh (tmp12 )
tmp14 =tmp12 -tmp13 
tmp15 =triton_helpers .minimum (tmp1 ,tmp14 )
tmp16 =tl_math .abs (tmp14 )
tmp17 =-tmp16 
tmp18 =tl_math .exp (tmp17 )
tmp19 =libdevice .log1p (tmp18 )
tmp20 =tmp15 -tmp19 
tmp21 =libdevice .tanh (tmp20 )
tmp22 =tmp20 -tmp21 
tmp23 =triton_helpers .minimum (tmp1 ,tmp22 )
tmp24 =tl_math .abs (tmp22 )
tmp25 =-tmp24 
tmp26 =tl_math .exp (tmp25 )
tmp27 =libdevice .log1p (tmp26 )
tmp28 =tmp23 -tmp27 
tmp29 =tl_math .abs (tmp28 )
tmp30 =1.0 
tmp31 =tmp29 <tmp30 
tmp32 =0.5 
tmp33 =tmp29 *tmp32 
tmp34 =tmp33 *tmp29 
tmp35 =tmp29 -tmp32 
tmp36 =tmp35 *tmp30 
tmp37 =tl .where (tmp31 ,tmp34 ,tmp36 )
tmp38 =tl .broadcast_to (tmp37 ,[XBLOCK ,R0_BLOCK ])
tmp40 =_tmp39 +tmp38 
_tmp39 =tl .where (r0_mask &xmask ,tmp40 ,_tmp39 )
tmp39 =tl .sum (_tmp39 ,1 )[:,None ]
tl .store (out_ptr0 +(x0 ),tmp39 ,xmask )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
fromtorch ._inductor .runtime .triton_helpersimportlibdevice ,mathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_per_fused__prelu_kernel_huber_loss_huber_loss_backward_log_sigmoid_forward_sub_tanh_6 (in_out_ptr0 ,in_ptr0 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    xnumel =1 
r0_numel =64 
R0_BLOCK :tl .constexpr =64 
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
tmp3 =tl .sum (tmp1 ,1 )[:,None ]
tmp4 =524288.0 
tmp5 =tmp3 /tmp4 
tl .debug_barrier ()
tl .store (in_out_ptr0 +(tl .full ([XBLOCK ,1 ],0 ,tl .int32 )),tmp5 ,None )

importtriton 
importtriton .languageastl 

fromtorch ._inductor .runtimeimporttriton_helpers 
fromtorch ._inductor .runtime .triton_helpersimportlibdevice ,mathastl_math 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
deftriton_poi_fused_add_7 (in_ptr0 ,out_ptr1 ,xnumel ,XBLOCK :tl .constexpr ):
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
    primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ,primals_8 ,primals_9 ,primals_10 ,primals_11 ,primals_12 ,primals_13 ,primals_14 ,primals_15 =args 
args .clear ()
assert_size_stride (primals_1 ,(1 ,16 ,8 ,64 ,64 ),(524288 ,32768 ,4096 ,64 ,1 ))
assert_size_stride (primals_2 ,(),())
assert_size_stride (primals_3 ,(16 ,),(1 ,))
assert_size_stride (primals_4 ,(16 ,),(1 ,))
assert_size_stride (primals_5 ,(16 ,),(1 ,))
assert_size_stride (primals_6 ,(16 ,),(1 ,))
assert_size_stride (primals_7 ,(512 ,64 ),(64 ,1 ))
assert_size_stride (primals_8 ,(512 ,128 ),(128 ,1 ))
assert_size_stride (primals_9 ,(512 ,),(1 ,))
assert_size_stride (primals_10 ,(512 ,),(1 ,))
assert_size_stride (primals_11 ,(256 ,128 ),(128 ,1 ))
assert_size_stride (primals_12 ,(256 ,64 ),(64 ,1 ))
assert_size_stride (primals_13 ,(256 ,),(1 ,))
assert_size_stride (primals_14 ,(256 ,),(1 ,))
assert_size_stride (primals_15 ,(1 ,),(1 ,))
withtorch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )
buf0 =empty_strided_cuda ((1 ,16 ,1 ,1 ,1 ,4 ),(64 ,4 ,64 ,64 ,64 ,1 ),torch .float32 )
buf1 =empty_strided_cuda ((1 ,16 ,1 ,1 ,1 ,4 ),(64 ,4 ,64 ,64 ,64 ,1 ),torch .float32 )
buf2 =empty_strided_cuda ((1 ,16 ,1 ,1 ,1 ,4 ),(64 ,4 ,64 ,64 ,64 ,1 ),torch .float32 )

stream0 =get_raw_stream (0 )
triton_red_fused__native_batch_norm_legit_functional_0 [grid (64 )](primals_1 ,buf0 ,buf1 ,buf2 ,64 ,8192 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
buf3 =empty_strided_cuda ((1 ,16 ,1 ,1 ,1 ),(16 ,1 ,16 ,16 ,16 ),torch .float32 )
buf4 =empty_strided_cuda ((1 ,16 ,1 ,1 ,1 ),(16 ,1 ,16 ,16 ,16 ),torch .float32 )
buf6 =empty_strided_cuda ((1 ,16 ,1 ,1 ,1 ),(16 ,1 ,16 ,16 ,16 ),torch .float32 )

stream0 =get_raw_stream (0 )
triton_per_fused__native_batch_norm_legit_functional_1 [grid (16 )](buf0 ,buf1 ,buf2 ,primals_4 ,primals_3 ,buf3 ,buf4 ,buf6 ,primals_4 ,primals_3 ,16 ,4 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
delbuf0 
delbuf1 
delprimals_3 
delprimals_4 
buf7 =empty_strided_cuda ((8 ,16 ,32 ,2 ,32 ,2 ),(65536 ,4096 ,128 ,64 ,2 ,1 ),torch .float32 )

stream0 =get_raw_stream (0 )
triton_poi_fused_pixel_shuffle_2 [grid (262144 ,2 )](primals_1 ,buf3 ,buf4 ,primals_5 ,primals_6 ,buf7 ,262144 ,2 ,XBLOCK =2 ,YBLOCK =1024 ,num_warps =8 ,num_stages =1 )
delbuf4 
delprimals_5 
delprimals_6 
buf8 =empty_strided_cuda ((8192 ,128 ),(128 ,1 ),torch .float32 )

stream0 =get_raw_stream (0 )
triton_poi_fused__to_copy_3 [grid (1048576 )](buf8 ,1048576 ,XBLOCK =1024 ,num_warps =4 ,num_stages =1 )
buf9 =empty_strided_cuda ((8192 ,512 ),(512 ,1 ),torch .float32 )

extern_kernels .mm (reinterpret_tensor (buf7 ,(8192 ,64 ),(64 ,1 ),0 ),reinterpret_tensor (primals_7 ,(64 ,512 ),(1 ,64 ),0 ),out =buf9 )
buf10 =empty_strided_cuda ((8192 ,512 ),(512 ,1 ),torch .float32 )

extern_kernels .mm (buf8 ,reinterpret_tensor (primals_8 ,(128 ,512 ),(1 ,128 ),0 ),out =buf10 )
delprimals_8 

buf11 =torch .ops .aten ._thnn_fused_lstm_cell .default (buf9 ,buf10 ,buf8 ,primals_9 ,primals_10 )
delbuf10 
delbuf9 
delprimals_10 
delprimals_9 
buf12 =buf11 [0 ]
buf13 =buf11 [1 ]
buf14 =buf11 [2 ]
delbuf11 
buf15 =empty_strided_cuda ((8192 ,64 ),(64 ,1 ),torch .float32 )

stream0 =get_raw_stream (0 )
triton_poi_fused__to_copy_4 [grid (524288 )](buf15 ,524288 ,XBLOCK =1024 ,num_warps =4 ,num_stages =1 )
buf16 =empty_strided_cuda ((8192 ,256 ),(256 ,1 ),torch .float32 )

extern_kernels .mm (buf12 ,reinterpret_tensor (primals_11 ,(128 ,256 ),(1 ,128 ),0 ),out =buf16 )
buf17 =empty_strided_cuda ((8192 ,256 ),(256 ,1 ),torch .float32 )

extern_kernels .mm (buf15 ,reinterpret_tensor (primals_12 ,(64 ,256 ),(1 ,64 ),0 ),out =buf17 )
delprimals_12 

buf18 =torch .ops .aten ._thnn_fused_lstm_cell .default (buf16 ,buf17 ,buf15 ,primals_13 ,primals_14 )
delbuf16 
delbuf17 
delprimals_13 
delprimals_14 
buf19 =buf18 [0 ]
buf20 =buf18 [1 ]
buf21 =buf18 [2 ]
delbuf18 
buf22 =reinterpret_tensor (buf2 ,(64 ,),(1 ,),0 );delbuf2 

stream0 =get_raw_stream (0 )
triton_red_fused__prelu_kernel_huber_loss_huber_loss_backward_log_sigmoid_forward_sub_tanh_5 [grid (64 )](buf19 ,primals_15 ,buf22 ,64 ,8192 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
buf23 =empty_strided_cuda ((),(),torch .float32 )
buf32 =buf23 ;delbuf23 

stream0 =get_raw_stream (0 )
triton_per_fused__prelu_kernel_huber_loss_huber_loss_backward_log_sigmoid_forward_sub_tanh_6 [grid (1 )](buf32 ,buf22 ,1 ,64 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
delbuf22 

stream0 =get_raw_stream (0 )
triton_poi_fused_add_7 [grid (1 )](primals_2 ,primals_2 ,1 ,XBLOCK =1 ,num_warps =1 ,num_stages =1 )
delprimals_2 
return (buf32 ,primals_1 ,primals_15 ,reinterpret_tensor (buf6 ,(16 ,),(1 ,),0 ),reinterpret_tensor (buf7 ,(8192 ,64 ),(64 ,1 ),0 ),buf8 ,buf12 ,buf13 ,buf14 ,buf15 ,buf19 ,buf20 ,buf21 ,primals_11 ,primals_7 ,reinterpret_tensor (buf3 ,(1 ,16 ,1 ,1 ,1 ),(16 ,1 ,1 ,1 ,1 ),0 ),)

defbenchmark_compiled_module (times =10 ,repeat =10 ):
    fromtorch ._dynamo .testingimportrand_strided 
fromtorch ._inductor .utilsimportprint_performance 
primals_1 =rand_strided ((1 ,16 ,8 ,64 ,64 ),(524288 ,32768 ,4096 ,64 ,1 ),device ='cuda:0',dtype =torch .float32 )
primals_2 =rand_strided ((),(),device ='cuda:0',dtype =torch .int64 )
primals_3 =rand_strided ((16 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_4 =rand_strided ((16 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_5 =rand_strided ((16 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_6 =rand_strided ((16 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_7 =rand_strided ((512 ,64 ),(64 ,1 ),device ='cuda:0',dtype =torch .float32 )
primals_8 =rand_strided ((512 ,128 ),(128 ,1 ),device ='cuda:0',dtype =torch .float32 )
primals_9 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_10 =rand_strided ((512 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_11 =rand_strided ((256 ,128 ),(128 ,1 ),device ='cuda:0',dtype =torch .float32 )
primals_12 =rand_strided ((256 ,64 ),(64 ,1 ),device ='cuda:0',dtype =torch .float32 )
primals_13 =rand_strided ((256 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_14 =rand_strided ((256 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
primals_15 =rand_strided ((1 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
fn =lambda :call ([primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ,primals_8 ,primals_9 ,primals_10 ,primals_11 ,primals_12 ,primals_13 ,primals_14 ,primals_15 ])
returnprint_performance (fn ,times =times ,repeat =repeat )

if__name__ =="__main__":
    fromtorch ._inductor .wrapper_benchmarkimportcompiled_module_main 
compiled_module_main ('None',benchmark_compiled_module )
