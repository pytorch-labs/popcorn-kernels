
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
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused__native_batch_norm_legit_convolution_0 (in_out_ptr0 ,in_ptr0 ,out_ptr0 ,out_ptr1 ,out_ptr2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =64 
    r0_numel =8192 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x3 =xindex 
    x1 =xindex //4 
    tmp1 =tl .load (in_ptr0 +(x1 ),xmask ,eviction_policy ='evict_last')
    tmp4_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp4_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp4_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        roffset =r0_offset 
        r0_2 =r0_index 
        tmp0 =tl .load (in_out_ptr0 +(r0_2 +8192 *x3 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp2 =tmp0 +tmp1 
        tmp3 =tl .broadcast_to (tmp2 ,[XBLOCK ,R0_BLOCK ])
        tmp4_mean_next ,tmp4_m2_next ,tmp4_weight_next =triton_helpers .welford_reduce (
        tmp3 ,tmp4_mean ,tmp4_m2 ,tmp4_weight ,roffset ==0 
        )
        tmp4_mean =tl .where (r0_mask &xmask ,tmp4_mean_next ,tmp4_mean )
        tmp4_m2 =tl .where (r0_mask &xmask ,tmp4_m2_next ,tmp4_m2 )
        tmp4_weight =tl .where (r0_mask &xmask ,tmp4_weight_next ,tmp4_weight )
        tl .store (in_out_ptr0 +(r0_2 +8192 *x3 ),tmp2 ,r0_mask &xmask )
    tmp7 ,tmp8 ,tmp9 =triton_helpers .welford (tmp4_mean ,tmp4_m2 ,tmp4_weight ,1 )
    tmp4 =tmp7 [:,None ]
    tmp5 =tmp8 [:,None ]
    tmp6 =tmp9 [:,None ]
    tl .store (out_ptr0 +(x3 ),tmp4 ,xmask )
    tl .store (out_ptr1 +(x3 ),tmp5 ,xmask )
    tl .store (out_ptr2 +(x3 ),tmp6 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused__native_batch_norm_legit_1 (in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr0 ,out_ptr1 ,out_ptr2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    xnumel =16 
    R0_BLOCK :tl .constexpr =4 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_1 =r0_index 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(r0_1 +4 *x0 ),xmask ,other =0.0 )
    tmp1 =tl .load (in_ptr1 +(r0_1 +4 *x0 ),xmask ,other =0.0 )
    tmp2 =tl .load (in_ptr2 +(r0_1 +4 *x0 ),xmask ,other =0.0 )
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
    tl .store (out_ptr2 +(x0 ),tmp20 ,xmask )
    tl .store (out_ptr0 +(x0 ),tmp13 ,xmask )
    tl .store (out_ptr1 +(x0 ),tmp14 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__native_batch_norm_legit_unsqueeze_2 (in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr1 ,ynumel ,xnumel ,YBLOCK :tl .constexpr ,XBLOCK :tl .constexpr ):
    xnumel =32 
    yoffset =tl .program_id (1 )*YBLOCK 
    yindex =yoffset +tl .arange (0 ,YBLOCK )[None ,:]
    tl .full ([XBLOCK ,YBLOCK ],True ,tl .int1 )
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    x2 =xindex 
    y3 =yindex 
    y1 =yindex //1024 
    tmp0 =tl .load (in_ptr0 +(x2 +32 *y3 ),xmask ,eviction_policy ='evict_last')
    tmp1 =tl .load (in_ptr1 +(y1 ),None ,eviction_policy ='evict_last')
    tmp3 =tl .load (in_ptr2 +(y1 ),None ,eviction_policy ='evict_last')
    tmp2 =tmp0 -tmp1 
    tmp4 =32768.0 
    tmp5 =tmp3 /tmp4 
    tmp6 =1e-05 
    tmp7 =tmp5 +tmp6 
    tmp8 =libdevice .rsqrt (tmp7 )
    tmp9 =tmp2 *tmp8 
    tl .store (out_ptr1 +(y3 +16384 *x2 ),tmp9 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_max_pool2d_with_indices_3 (in_ptr0 ,out_ptr0 ,out_ptr1 ,ynumel ,xnumel ,YBLOCK :tl .constexpr ,XBLOCK :tl .constexpr ):
    ynumel =16 
    yoffset =tl .program_id (1 )*YBLOCK 
    yindex =yoffset +tl .arange (0 ,YBLOCK )[None ,:]
    ymask =yindex <ynumel 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,YBLOCK ],True ,tl .int1 )
    x1 =xindex 
    y0 =yindex 
    tmp0 =tl .load (in_ptr0 +(x1 +32768 *y0 ),ymask ,eviction_policy ='evict_last')
    tmp1 =tl .load (in_ptr0 +(16384 +x1 +32768 *y0 ),ymask ,eviction_policy ='evict_last')
    tmp2 =tmp1 >tmp0 
    tmp3 =tl .full ([1 ,1 ],1 ,tl .int8 )
    tmp4 =tl .full ([1 ,1 ],0 ,tl .int8 )
    tmp5 =tl .where (tmp2 ,tmp3 ,tmp4 )
    tmp6 =triton_helpers .maximum (tmp1 ,tmp0 )
    tl .store (out_ptr0 +(x1 +16384 *y0 ),tmp5 ,ymask )
    tl .store (out_ptr1 +(y0 +16 *x1 ),tmp6 ,ymask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused__native_batch_norm_legit_convolution_4 (in_out_ptr0 ,in_ptr0 ,out_ptr0 ,out_ptr1 ,out_ptr2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =64 
    r0_numel =8192 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x3 =xindex 
    x1 =xindex //2 
    tmp1 =tl .load (in_ptr0 +(x1 ),xmask ,eviction_policy ='evict_last')
    tmp4_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp4_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp4_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        roffset =r0_offset 
        r0_2 =r0_index 
        tmp0 =tl .load (in_out_ptr0 +(r0_2 +8192 *x3 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp2 =tmp0 +tmp1 
        tmp3 =tl .broadcast_to (tmp2 ,[XBLOCK ,R0_BLOCK ])
        tmp4_mean_next ,tmp4_m2_next ,tmp4_weight_next =triton_helpers .welford_reduce (
        tmp3 ,tmp4_mean ,tmp4_m2 ,tmp4_weight ,roffset ==0 
        )
        tmp4_mean =tl .where (r0_mask &xmask ,tmp4_mean_next ,tmp4_mean )
        tmp4_m2 =tl .where (r0_mask &xmask ,tmp4_m2_next ,tmp4_m2 )
        tmp4_weight =tl .where (r0_mask &xmask ,tmp4_weight_next ,tmp4_weight )
        tl .store (in_out_ptr0 +(r0_2 +8192 *x3 ),tmp2 ,r0_mask &xmask )
    tmp7 ,tmp8 ,tmp9 =triton_helpers .welford (tmp4_mean ,tmp4_m2 ,tmp4_weight ,1 )
    tmp4 =tmp7 [:,None ]
    tmp5 =tmp8 [:,None ]
    tmp6 =tmp9 [:,None ]
    tl .store (out_ptr0 +(x3 ),tmp4 ,xmask )
    tl .store (out_ptr1 +(x3 ),tmp5 ,xmask )
    tl .store (out_ptr2 +(x3 ),tmp6 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused__native_batch_norm_legit_5 (in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr0 ,out_ptr1 ,out_ptr2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    xnumel =32 
    R0_BLOCK :tl .constexpr =2 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_1 =r0_index 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(r0_1 +2 *x0 ),xmask ,other =0.0 )
    tmp1 =tl .load (in_ptr1 +(r0_1 +2 *x0 ),xmask ,other =0.0 )
    tmp2 =tl .load (in_ptr2 +(r0_1 +2 *x0 ),xmask ,other =0.0 )
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
    tmp16 =16384.0 
    tmp17 =tmp14 /tmp16 
    tmp18 =1e-05 
    tmp19 =tmp17 +tmp18 
    tmp20 =libdevice .rsqrt (tmp19 )
    tl .store (out_ptr2 +(x0 ),tmp20 ,xmask )
    tl .store (out_ptr0 +(x0 ),tmp13 ,xmask )
    tl .store (out_ptr1 +(x0 ),tmp14 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_bernoulli_6 (in_ptr0 ,out_ptr1 ,load_seed_offset ,xnumel ,XBLOCK :tl .constexpr ):
    xnumel =32 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    xmask =xindex <xnumel 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =x0 
    tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tmp3 =0.5 
    tmp4 =tmp2 <tmp3 
    tl .store (out_ptr1 +(x0 ),tmp4 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__native_batch_norm_legit__to_copy_add_mul_7 (in_ptr0 ,in_ptr1 ,in_ptr2 ,in_ptr3 ,out_ptr0 ,xnumel ,XBLOCK :tl .constexpr ):
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:]
    tl .full ([XBLOCK ],True ,tl .int1 )
    x2 =xindex 
    x1 =xindex //16384 
    tmp0 =tl .load (in_ptr0 +(x2 ),None )
    tmp1 =tl .load (in_ptr1 +(x1 ),None ,eviction_policy ='evict_last')
    tmp3 =tl .load (in_ptr2 +(x1 ),None ,eviction_policy ='evict_last')
    tmp10 =tl .load (in_ptr3 +(x1 ),None ,eviction_policy ='evict_last').to (tl .int1 )
    tmp2 =tmp0 -tmp1 
    tmp4 =16384.0 
    tmp5 =tmp3 /tmp4 
    tmp6 =1e-05 
    tmp7 =tmp5 +tmp6 
    tmp8 =libdevice .rsqrt (tmp7 )
    tmp9 =tmp2 *tmp8 
    tmp11 =tmp10 .to (tl .float32 )
    tmp12 =0.8864048946659319 
    tmp13 =tmp11 *tmp12 
    tmp14 =tmp9 *tmp13 
    tmp15 =-1.0 
    tmp16 =tmp11 +tmp15 
    tmp17 =1.558387861036063 
    tmp18 =tmp16 *tmp17 
    tmp19 =0.7791939305180315 
    tmp20 =tmp18 +tmp19 
    tmp21 =tmp14 +tmp20 
    tl .store (out_ptr0 +(x2 ),tmp21 ,None )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused__native_batch_norm_legit_convolution_8 (in_out_ptr0 ,in_ptr0 ,out_ptr0 ,out_ptr1 ,out_ptr2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =128 
    r0_numel =8192 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x3 =xindex 
    x1 =xindex //2 
    tmp1 =tl .load (in_ptr0 +(x1 ),xmask ,eviction_policy ='evict_last')
    tmp4_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp4_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp4_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        roffset =r0_offset 
        r0_2 =r0_index 
        tmp0 =tl .load (in_out_ptr0 +(r0_2 +8192 *x3 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp2 =tmp0 +tmp1 
        tmp3 =tl .broadcast_to (tmp2 ,[XBLOCK ,R0_BLOCK ])
        tmp4_mean_next ,tmp4_m2_next ,tmp4_weight_next =triton_helpers .welford_reduce (
        tmp3 ,tmp4_mean ,tmp4_m2 ,tmp4_weight ,roffset ==0 
        )
        tmp4_mean =tl .where (r0_mask &xmask ,tmp4_mean_next ,tmp4_mean )
        tmp4_m2 =tl .where (r0_mask &xmask ,tmp4_m2_next ,tmp4_m2 )
        tmp4_weight =tl .where (r0_mask &xmask ,tmp4_weight_next ,tmp4_weight )
        tl .store (in_out_ptr0 +(r0_2 +8192 *x3 ),tmp2 ,r0_mask &xmask )
    tmp7 ,tmp8 ,tmp9 =triton_helpers .welford (tmp4_mean ,tmp4_m2 ,tmp4_weight ,1 )
    tmp4 =tmp7 [:,None ]
    tmp5 =tmp8 [:,None ]
    tmp6 =tmp9 [:,None ]
    tl .store (out_ptr0 +(x3 ),tmp4 ,xmask )
    tl .store (out_ptr1 +(x3 ),tmp5 ,xmask )
    tl .store (out_ptr2 +(x3 ),tmp6 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_per_fused__native_batch_norm_legit_9 (in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr0 ,out_ptr1 ,out_ptr2 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ):
    xnumel =64 
    R0_BLOCK :tl .constexpr =2 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_index =tl .arange (0 ,R0_BLOCK )[None ,:]
    tl .full ([XBLOCK ,R0_BLOCK ],True ,tl .int1 )
    r0_1 =r0_index 
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +(r0_1 +2 *x0 ),xmask ,other =0.0 )
    tmp1 =tl .load (in_ptr1 +(r0_1 +2 *x0 ),xmask ,other =0.0 )
    tmp2 =tl .load (in_ptr2 +(r0_1 +2 *x0 ),xmask ,other =0.0 )
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
    tmp16 =16384.0 
    tmp17 =tmp14 /tmp16 
    tmp18 =1e-05 
    tmp19 =tmp17 +tmp18 
    tmp20 =libdevice .rsqrt (tmp19 )
    tl .store (out_ptr2 +(x0 ),tmp20 ,xmask )
    tl .store (out_ptr0 +(x0 ),tmp13 ,xmask )
    tl .store (out_ptr1 +(x0 ),tmp14 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused__native_batch_norm_legit_unsqueeze_10 (in_ptr0 ,in_ptr1 ,in_ptr2 ,out_ptr1 ,ynumel ,xnumel ,YBLOCK :tl .constexpr ,XBLOCK :tl .constexpr ):
    ynumel =65536 
    xnumel =16 
    yoffset =(tl .program_id (1 )+tl .program_id (2 )*tl .num_programs (1 ))*YBLOCK 
    yindex =yoffset +tl .arange (0 ,YBLOCK )[None ,:]
    ymask =yindex <ynumel 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    x2 =xindex 
    y3 =yindex 
    y1 =yindex //1024 
    tmp0 =tl .load (in_ptr0 +(x2 +16 *y3 ),xmask &ymask ,eviction_policy ='evict_last')
    tmp1 =tl .load (in_ptr1 +(y1 ),ymask ,eviction_policy ='evict_last')
    tmp3 =tl .load (in_ptr2 +(y1 ),ymask ,eviction_policy ='evict_last')
    tmp2 =tmp0 -tmp1 
    tmp4 =16384.0 
    tmp5 =tmp3 /tmp4 
    tmp6 =1e-05 
    tmp7 =tmp5 +tmp6 
    tmp8 =libdevice .rsqrt (tmp7 )
    tmp9 =tmp2 *tmp8 
    tl .store (out_ptr1 +(y3 +65536 *x2 ),tmp9 ,xmask &ymask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_poi_fused_max_pool2d_with_indices_11 (in_ptr0 ,out_ptr0 ,out_ptr1 ,ynumel ,xnumel ,YBLOCK :tl .constexpr ,XBLOCK :tl .constexpr ):
    ynumel =8 
    yoffset =tl .program_id (1 )*YBLOCK 
    yindex =yoffset +tl .arange (0 ,YBLOCK )[None ,:]
    ymask =yindex <ynumel 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    tl .full ([XBLOCK ,YBLOCK ],True ,tl .int1 )
    x1 =xindex 
    y0 =yindex 
    tmp0 =tl .load (in_ptr0 +(x1 +131072 *y0 ),ymask ,eviction_policy ='evict_last')
    tmp1 =tl .load (in_ptr0 +(65536 +x1 +131072 *y0 ),ymask ,eviction_policy ='evict_last')
    tmp2 =tmp1 >tmp0 
    tmp3 =tl .full ([1 ,1 ],1 ,tl .int8 )
    tmp4 =tl .full ([1 ,1 ],0 ,tl .int8 )
    tmp5 =tl .where (tmp2 ,tmp3 ,tmp4 )
    tmp6 =triton_helpers .maximum (tmp1 ,tmp0 )
    tl .store (out_ptr0 +(x1 +65536 *y0 ),tmp5 ,ymask )
    tl .store (out_ptr1 +(y0 +8 *x1 ),tmp6 ,ymask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused__native_batch_norm_legit__to_copy_add_bernoulli_convolution_mul_12 (in_out_ptr0 ,in_ptr0 ,in_ptr1 ,out_ptr1 ,out_ptr2 ,out_ptr4 ,out_ptr5 ,load_seed_offset ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =128 
    r0_numel =8192 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    tmp0 =tl .load (in_ptr0 +load_seed_offset )
    tmp1 =x0 
    tmp2 =tl .rand (tmp0 ,(tmp1 ).to (tl .uint32 ))
    tmp3 =0.5 
    tmp4 =tmp2 <tmp3 
    tl .store (out_ptr1 +(x0 ),tmp4 ,xmask )
    tmp6 =tl .load (in_ptr1 +(x0 ),xmask ,eviction_policy ='evict_last')
    tmp9_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp9_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp9_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        roffset =r0_offset 
        r0_1 =r0_index 
        tmp5 =tl .load (in_out_ptr0 +(r0_1 +8192 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp7 =tmp5 +tmp6 
        tmp8 =tl .broadcast_to (tmp7 ,[XBLOCK ,R0_BLOCK ])
        tmp9_mean_next ,tmp9_m2_next ,tmp9_weight_next =triton_helpers .welford_reduce (
        tmp8 ,tmp9_mean ,tmp9_m2 ,tmp9_weight ,roffset ==0 
        )
        tmp9_mean =tl .where (r0_mask &xmask ,tmp9_mean_next ,tmp9_mean )
        tmp9_m2 =tl .where (r0_mask &xmask ,tmp9_m2_next ,tmp9_m2 )
        tmp9_weight =tl .where (r0_mask &xmask ,tmp9_weight_next ,tmp9_weight )
        tl .store (in_out_ptr0 +(r0_1 +8192 *x0 ),tmp7 ,r0_mask &xmask )
    tmp12 ,tmp13 ,tmp14 =triton_helpers .welford (tmp9_mean ,tmp9_m2 ,tmp9_weight ,1 )
    tmp9 =tmp12 [:,None ]
    tmp10 =tmp13 [:,None ]
    tmp11 =tmp14 [:,None ]
    tl .store (out_ptr2 +(x0 ),tmp9 ,xmask )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        roffset =r0_offset 
        r0_1 =r0_index 
        tmp15 =tl .load (in_out_ptr0 +(r0_1 +8192 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp16 =tmp15 -tmp9 
        tmp17 =8192.0 
        tmp18 =tmp10 /tmp17 
        tmp19 =1e-05 
        tmp20 =tmp18 +tmp19 
        tmp21 =libdevice .rsqrt (tmp20 )
        tmp22 =tmp16 *tmp21 
        tmp23 =tmp4 .to (tl .float32 )
        tmp24 =0.8864048946659319 
        tmp25 =tmp23 *tmp24 
        tmp26 =tmp22 *tmp25 
        tmp27 =-1.0 
        tmp28 =tmp23 +tmp27 
        tmp29 =1.558387861036063 
        tmp30 =tmp28 *tmp29 
        tmp31 =0.7791939305180315 
        tmp32 =tmp30 +tmp31 
        tmp33 =tmp26 +tmp32 
        tl .store (out_ptr4 +(r0_1 +8192 *x0 ),tmp33 ,r0_mask &xmask )
    tmp34 =8192.0 
    tmp35 =tmp10 /tmp34 
    tmp36 =1e-05 
    tmp37 =tmp35 +tmp36 
    tmp38 =libdevice .rsqrt (tmp37 )
    tl .store (out_ptr5 +(x0 ),tmp38 ,xmask )

import triton 
import triton .language as tl 

from torch ._inductor .runtime import triton_helpers 
from torch ._inductor .runtime .triton_helpers import libdevice 
triton_helpers .set_driver_to_gpu ()

@triton .jit 
def triton_red_fused__native_batch_norm_legit_convolution_13 (in_out_ptr0 ,in_ptr0 ,out_ptr0 ,out_ptr2 ,out_ptr3 ,xnumel ,r0_numel ,XBLOCK :tl .constexpr ,R0_BLOCK :tl .constexpr ):
    xnumel =256 
    r0_numel =8192 
    RBLOCK :tl .constexpr =R0_BLOCK 
    xoffset =tl .program_id (0 )*XBLOCK 
    xindex =xoffset +tl .arange (0 ,XBLOCK )[:,None ]
    xmask =xindex <xnumel 
    r0_base =tl .arange (0 ,R0_BLOCK )[None ,:]
    x0 =xindex 
    tmp1 =tl .load (in_ptr0 +(x0 ),xmask ,eviction_policy ='evict_last')
    tmp4_mean =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp4_m2 =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    tmp4_weight =tl .zeros ([XBLOCK ,R0_BLOCK ],tl .float32 )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        roffset =r0_offset 
        r0_1 =r0_index 
        tmp0 =tl .load (in_out_ptr0 +(r0_1 +8192 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp2 =tmp0 +tmp1 
        tmp3 =tl .broadcast_to (tmp2 ,[XBLOCK ,R0_BLOCK ])
        tmp4_mean_next ,tmp4_m2_next ,tmp4_weight_next =triton_helpers .welford_reduce (
        tmp3 ,tmp4_mean ,tmp4_m2 ,tmp4_weight ,roffset ==0 
        )
        tmp4_mean =tl .where (r0_mask &xmask ,tmp4_mean_next ,tmp4_mean )
        tmp4_m2 =tl .where (r0_mask &xmask ,tmp4_m2_next ,tmp4_m2 )
        tmp4_weight =tl .where (r0_mask &xmask ,tmp4_weight_next ,tmp4_weight )
        tl .store (in_out_ptr0 +(r0_1 +8192 *x0 ),tmp2 ,r0_mask &xmask )
    tmp7 ,tmp8 ,tmp9 =triton_helpers .welford (tmp4_mean ,tmp4_m2 ,tmp4_weight ,1 )
    tmp4 =tmp7 [:,None ]
    tmp5 =tmp8 [:,None ]
    tmp6 =tmp9 [:,None ]
    tl .store (out_ptr0 +(x0 ),tmp4 ,xmask )
    for r0_offset in range (0 ,r0_numel ,R0_BLOCK ):
        r0_index =r0_offset +r0_base 
        r0_mask =r0_index <r0_numel 
        roffset =r0_offset 
        r0_1 =r0_index 
        tmp10 =tl .load (in_out_ptr0 +(r0_1 +8192 *x0 ),r0_mask &xmask ,eviction_policy ='evict_first',other =0.0 )
        tmp11 =tmp10 -tmp4 
        tmp12 =8192.0 
        tmp13 =tmp5 /tmp12 
        tmp14 =1e-05 
        tmp15 =tmp13 +tmp14 
        tmp16 =libdevice .rsqrt (tmp15 )
        tmp17 =tmp11 *tmp16 
        tl .store (out_ptr2 +(r0_1 +8192 *x0 ),tmp17 ,r0_mask &xmask )
    tmp18 =8192.0 
    tmp19 =tmp5 /tmp18 
    tmp20 =1e-05 
    tmp21 =tmp19 +tmp20 
    tmp22 =libdevice .rsqrt (tmp21 )
    tl .store (out_ptr3 +(x0 ),tmp22 ,xmask )

def call (args ):
    primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ,primals_8 ,primals_9 ,primals_10 ,primals_11 =args 
    args .clear ()
    assert_size_stride (primals_1 ,(16 ,1 ,3 ,3 ,3 ),(27 ,27 ,9 ,3 ,1 ))
    assert_size_stride (primals_2 ,(16 ,),(1 ,))
    assert_size_stride (primals_3 ,(1 ,1 ,32 ,32 ,32 ),(32768 ,32768 ,1024 ,32 ,1 ))
    assert_size_stride (primals_4 ,(32 ,16 ,3 ,3 ,3 ),(432 ,27 ,9 ,3 ,1 ))
    assert_size_stride (primals_5 ,(32 ,),(1 ,))
    assert_size_stride (primals_6 ,(64 ,32 ,3 ,3 ,3 ),(864 ,27 ,9 ,3 ,1 ))
    assert_size_stride (primals_7 ,(64 ,),(1 ,))
    assert_size_stride (primals_8 ,(128 ,64 ,3 ,3 ,3 ),(1728 ,27 ,9 ,3 ,1 ))
    assert_size_stride (primals_9 ,(128 ,),(1 ,))
    assert_size_stride (primals_10 ,(256 ,128 ,3 ,3 ,3 ),(3456 ,27 ,9 ,3 ,1 ))
    assert_size_stride (primals_11 ,(256 ,),(1 ,))
    with torch .cuda ._DeviceGuard (0 ):
        torch .cuda .set_device (0 )

        buf0 =extern_kernels .convolution (primals_3 ,primals_1 ,stride =(1 ,1 ,1 ),padding =(1 ,1 ,1 ),dilation =(1 ,1 ,1 ),transposed =False ,output_padding =(0 ,0 ,0 ),groups =1 ,bias =None )
        assert_size_stride (buf0 ,(1 ,16 ,32 ,32 ,32 ),(524288 ,32768 ,1024 ,32 ,1 ))
        buf1 =buf0 ;del buf0 
        buf2 =empty_strided_cuda ((1 ,16 ,1 ,1 ,1 ,4 ),(64 ,4 ,64 ,64 ,64 ,1 ),torch .float32 )
        buf3 =empty_strided_cuda ((1 ,16 ,1 ,1 ,1 ,4 ),(64 ,4 ,64 ,64 ,64 ,1 ),torch .float32 )
        buf4 =empty_strided_cuda ((1 ,16 ,1 ,1 ,1 ,4 ),(64 ,4 ,64 ,64 ,64 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_red_fused__native_batch_norm_legit_convolution_0 [grid (64 )](buf1 ,primals_2 ,buf2 ,buf3 ,buf4 ,64 ,8192 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del primals_2 
        buf5 =empty_strided_cuda ((1 ,16 ,1 ,1 ,1 ),(16 ,1 ,16 ,16 ,16 ),torch .float32 )
        buf6 =empty_strided_cuda ((1 ,16 ,1 ,1 ,1 ),(16 ,1 ,16 ,16 ,16 ),torch .float32 )
        buf8 =empty_strided_cuda ((1 ,16 ,1 ,1 ,1 ),(16 ,1 ,16 ,16 ,16 ),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused__native_batch_norm_legit_1 [grid (16 )](buf2 ,buf3 ,buf4 ,buf5 ,buf6 ,buf8 ,16 ,4 ,XBLOCK =8 ,num_warps =2 ,num_stages =1 )
        buf10 =empty_strided_cuda ((1 ,16384 ,1 ,32 ),(524288 ,1 ,524288 ,16384 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused__native_batch_norm_legit_unsqueeze_2 [grid (16384 ,32 )](buf1 ,buf5 ,buf6 ,buf10 ,16384 ,32 ,XBLOCK =16 ,YBLOCK =256 ,num_warps =8 ,num_stages =1 )
        del buf6 
        buf11 =empty_strided_cuda ((1 ,16384 ,1 ,16 ),(262144 ,1 ,262144 ,16384 ),torch .int8 )
        buf12 =empty_strided_cuda ((1 ,16384 ,1 ,16 ),(262144 ,16 ,16 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_max_pool2d_with_indices_3 [grid (16 ,16384 )](buf10 ,buf11 ,buf12 ,16 ,16384 ,XBLOCK =64 ,YBLOCK =16 ,num_warps =4 ,num_stages =1 )

        buf13 =extern_kernels .convolution (reinterpret_tensor (buf12 ,(1 ,16 ,32 ,32 ,16 ),(0 ,16384 ,512 ,16 ,1 ),0 ),primals_4 ,stride =(1 ,1 ,1 ),padding =(1 ,1 ,1 ),dilation =(1 ,1 ,1 ),transposed =False ,output_padding =(0 ,0 ,0 ),groups =1 ,bias =None )
        assert_size_stride (buf13 ,(1 ,32 ,32 ,32 ,16 ),(524288 ,16384 ,512 ,16 ,1 ))
        buf14 =buf13 ;del buf13 
        buf15 =reinterpret_tensor (buf4 ,(1 ,32 ,1 ,1 ,1 ,2 ),(64 ,2 ,64 ,64 ,64 ,1 ),0 );del buf4 
        buf16 =reinterpret_tensor (buf3 ,(1 ,32 ,1 ,1 ,1 ,2 ),(64 ,2 ,64 ,64 ,64 ,1 ),0 );del buf3 
        buf17 =reinterpret_tensor (buf2 ,(1 ,32 ,1 ,1 ,1 ,2 ),(64 ,2 ,64 ,64 ,64 ,1 ),0 );del buf2 

        get_raw_stream (0 )
        triton_red_fused__native_batch_norm_legit_convolution_4 [grid (64 )](buf14 ,primals_5 ,buf15 ,buf16 ,buf17 ,64 ,8192 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del primals_5 
        buf18 =empty_strided_cuda ((1 ,32 ,1 ,1 ,1 ),(32 ,1 ,32 ,32 ,32 ),torch .float32 )
        buf19 =empty_strided_cuda ((1 ,32 ,1 ,1 ,1 ),(32 ,1 ,32 ,32 ,32 ),torch .float32 )
        buf21 =empty_strided_cuda ((1 ,32 ,1 ,1 ,1 ),(32 ,1 ,32 ,32 ,32 ),torch .float32 )

        get_raw_stream (0 )
        triton_per_fused__native_batch_norm_legit_5 [grid (32 )](buf15 ,buf16 ,buf17 ,buf18 ,buf19 ,buf21 ,32 ,2 ,XBLOCK =8 ,num_warps =2 ,num_stages =1 )
        buf22 =empty_strided_cuda ((2 ,),(1 ,),torch .int64 )

        aten .randint .low_out (-9223372036854775808 ,9223372036854775807 ,[2 ],out =buf22 )
        buf24 =empty_strided_cuda ((1 ,32 ,1 ,1 ,1 ),(32 ,1 ,1 ,1 ,1 ),torch .bool )

        get_raw_stream (0 )
        triton_poi_fused_bernoulli_6 [grid (32 )](buf22 ,buf24 ,0 ,32 ,XBLOCK =32 ,num_warps =1 ,num_stages =1 )
        buf25 =empty_strided_cuda ((1 ,32 ,32 ,32 ,16 ),(524288 ,16384 ,512 ,16 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused__native_batch_norm_legit__to_copy_add_mul_7 [grid (524288 )](buf14 ,buf18 ,buf19 ,buf24 ,buf25 ,524288 ,XBLOCK =1024 ,num_warps =4 ,num_stages =1 )
        del buf19 

        buf26 =extern_kernels .convolution (buf25 ,primals_6 ,stride =(1 ,1 ,1 ),padding =(1 ,1 ,1 ),dilation =(1 ,1 ,1 ),transposed =False ,output_padding =(0 ,0 ,0 ),groups =1 ,bias =None )
        assert_size_stride (buf26 ,(1 ,64 ,32 ,32 ,16 ),(1048576 ,16384 ,512 ,16 ,1 ))
        buf27 =buf26 ;del buf26 
        buf28 =empty_strided_cuda ((1 ,64 ,1 ,1 ,1 ,2 ),(128 ,2 ,128 ,128 ,128 ,1 ),torch .float32 )
        buf29 =empty_strided_cuda ((1 ,64 ,1 ,1 ,1 ,2 ),(128 ,2 ,128 ,128 ,128 ,1 ),torch .float32 )
        buf30 =empty_strided_cuda ((1 ,64 ,1 ,1 ,1 ,2 ),(128 ,2 ,128 ,128 ,128 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_red_fused__native_batch_norm_legit_convolution_8 [grid (128 )](buf27 ,primals_7 ,buf28 ,buf29 ,buf30 ,128 ,8192 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del primals_7 
        buf31 =reinterpret_tensor (buf17 ,(1 ,64 ,1 ,1 ,1 ),(64 ,1 ,64 ,64 ,64 ),0 );del buf17 
        buf32 =reinterpret_tensor (buf16 ,(1 ,64 ,1 ,1 ,1 ),(64 ,1 ,64 ,64 ,64 ),0 );del buf16 
        buf34 =reinterpret_tensor (buf15 ,(1 ,64 ,1 ,1 ,1 ),(64 ,1 ,64 ,64 ,64 ),0 );del buf15 

        get_raw_stream (0 )
        triton_per_fused__native_batch_norm_legit_9 [grid (64 )](buf28 ,buf29 ,buf30 ,buf31 ,buf32 ,buf34 ,64 ,2 ,XBLOCK =1 ,num_warps =2 ,num_stages =1 )
        del buf28 
        buf36 =empty_strided_cuda ((1 ,65536 ,1 ,16 ),(1048576 ,1 ,1048576 ,65536 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused__native_batch_norm_legit_unsqueeze_10 [grid (65536 ,16 )](buf27 ,buf31 ,buf32 ,buf36 ,65536 ,16 ,XBLOCK =16 ,YBLOCK =256 ,num_warps =8 ,num_stages =1 )
        del buf32 
        buf37 =empty_strided_cuda ((1 ,65536 ,1 ,8 ),(524288 ,1 ,524288 ,65536 ),torch .int8 )
        buf38 =empty_strided_cuda ((1 ,65536 ,1 ,8 ),(524288 ,8 ,8 ,1 ),torch .float32 )

        get_raw_stream (0 )
        triton_poi_fused_max_pool2d_with_indices_11 [grid (8 ,65536 )](buf36 ,buf37 ,buf38 ,8 ,65536 ,XBLOCK =128 ,YBLOCK =8 ,num_warps =4 ,num_stages =1 )

        buf39 =extern_kernels .convolution (reinterpret_tensor (buf38 ,(1 ,64 ,32 ,32 ,8 ),(0 ,8192 ,256 ,8 ,1 ),0 ),primals_8 ,stride =(1 ,1 ,1 ),padding =(1 ,1 ,1 ),dilation =(1 ,1 ,1 ),transposed =False ,output_padding =(0 ,0 ,0 ),groups =1 ,bias =None )
        assert_size_stride (buf39 ,(1 ,128 ,32 ,32 ,8 ),(1048576 ,8192 ,256 ,8 ,1 ))
        buf46 =empty_strided_cuda ((1 ,128 ,1 ,1 ,1 ),(128 ,1 ,1 ,1 ,1 ),torch .bool )
        buf40 =buf39 ;del buf39 
        buf41 =reinterpret_tensor (buf30 ,(1 ,128 ,1 ,1 ,1 ),(128 ,1 ,128 ,128 ,128 ),0 );del buf30 
        buf47 =empty_strided_cuda ((1 ,128 ,32 ,32 ,8 ),(1048576 ,8192 ,256 ,8 ,1 ),torch .float32 )
        buf44 =reinterpret_tensor (buf29 ,(1 ,128 ,1 ,1 ,1 ),(128 ,1 ,128 ,128 ,128 ),0 );del buf29 

        get_raw_stream (0 )
        triton_red_fused__native_batch_norm_legit__to_copy_add_bernoulli_convolution_mul_12 [grid (128 )](buf40 ,buf22 ,primals_9 ,buf46 ,buf41 ,buf47 ,buf44 ,1 ,128 ,8192 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del buf22 
        del primals_9 

        buf48 =extern_kernels .convolution (buf47 ,primals_10 ,stride =(1 ,1 ,1 ),padding =(1 ,1 ,1 ),dilation =(1 ,1 ,1 ),transposed =False ,output_padding =(0 ,0 ,0 ),groups =1 ,bias =None )
        assert_size_stride (buf48 ,(1 ,256 ,32 ,32 ,8 ),(2097152 ,8192 ,256 ,8 ,1 ))
        buf49 =buf48 ;del buf48 
        buf50 =empty_strided_cuda ((1 ,256 ,1 ,1 ,1 ),(256 ,1 ,256 ,256 ,256 ),torch .float32 )
        buf53 =empty_strided_cuda ((1 ,256 ,32 ,32 ,8 ),(2097152 ,8192 ,256 ,8 ,1 ),torch .float32 )
        buf54 =empty_strided_cuda ((1 ,256 ,1 ,1 ,1 ),(256 ,1 ,256 ,256 ,256 ),torch .float32 )

        get_raw_stream (0 )
        triton_red_fused__native_batch_norm_legit_convolution_13 [grid (256 )](buf49 ,primals_11 ,buf50 ,buf53 ,buf54 ,256 ,8192 ,XBLOCK =1 ,R0_BLOCK =2048 ,num_warps =16 ,num_stages =1 )
        del primals_11 
    return (buf53 ,primals_1 ,primals_3 ,primals_4 ,primals_6 ,primals_8 ,primals_10 ,buf1 ,reinterpret_tensor (buf8 ,(16 ,),(1 ,),0 ),buf10 ,buf11 ,reinterpret_tensor (buf12 ,(1 ,16 ,32 ,32 ,16 ),(262144 ,16384 ,512 ,16 ,1 ),0 ),buf14 ,reinterpret_tensor (buf21 ,(32 ,),(1 ,),0 ),buf24 ,buf25 ,buf27 ,reinterpret_tensor (buf34 ,(64 ,),(1 ,),0 ),buf36 ,buf37 ,reinterpret_tensor (buf38 ,(1 ,64 ,32 ,32 ,8 ),(524288 ,8192 ,256 ,8 ,1 ),0 ),buf40 ,reinterpret_tensor (buf44 ,(128 ,),(1 ,),0 ),buf46 ,buf47 ,buf49 ,reinterpret_tensor (buf54 ,(256 ,),(1 ,),0 ),reinterpret_tensor (buf50 ,(1 ,256 ,1 ,1 ,1 ),(256 ,1 ,1 ,1 ,1 ),0 ),reinterpret_tensor (buf41 ,(1 ,128 ,1 ,1 ,1 ),(128 ,1 ,1 ,1 ,1 ),0 ),reinterpret_tensor (buf31 ,(1 ,64 ,1 ,1 ,1 ),(64 ,1 ,1 ,1 ,1 ),0 ),reinterpret_tensor (buf18 ,(1 ,32 ,1 ,1 ,1 ),(32 ,1 ,1 ,1 ,1 ),0 ),reinterpret_tensor (buf5 ,(1 ,16 ,1 ,1 ,1 ),(16 ,1 ,1 ,1 ,1 ),0 ),)

def benchmark_compiled_module (times =10 ,repeat =10 ):
    from torch ._dynamo .testing import rand_strided 
    from torch ._inductor .utils import print_performance 
    primals_1 =rand_strided ((16 ,1 ,3 ,3 ,3 ),(27 ,27 ,9 ,3 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_2 =rand_strided ((16 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_3 =rand_strided ((1 ,1 ,32 ,32 ,32 ),(32768 ,32768 ,1024 ,32 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_4 =rand_strided ((32 ,16 ,3 ,3 ,3 ),(432 ,27 ,9 ,3 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_5 =rand_strided ((32 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_6 =rand_strided ((64 ,32 ,3 ,3 ,3 ),(864 ,27 ,9 ,3 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_7 =rand_strided ((64 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_8 =rand_strided ((128 ,64 ,3 ,3 ,3 ),(1728 ,27 ,9 ,3 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_9 =rand_strided ((128 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    primals_10 =rand_strided ((256 ,128 ,3 ,3 ,3 ),(3456 ,27 ,9 ,3 ,1 ),device ='cuda:0',dtype =torch .float32 )
    primals_11 =rand_strided ((256 ,),(1 ,),device ='cuda:0',dtype =torch .float32 )
    fn =lambda :call ([primals_1 ,primals_2 ,primals_3 ,primals_4 ,primals_5 ,primals_6 ,primals_7 ,primals_8 ,primals_9 ,primals_10 ,primals_11 ])
    return print_performance (fn ,times =times ,repeat =repeat )

if __name__ =="__main__":
    from torch ._inductor .wrapper_benchmark import compiled_module_main 
    compiled_module_main ('None',benchmark_compiled_module )
