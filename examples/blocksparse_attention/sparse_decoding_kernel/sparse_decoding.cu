#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void main_kernel_1(half_t* __restrict__ Output, float* __restrict__ Output_partial, float* __restrict__ glse, int num_split);
extern "C" __global__ void main_kernel(half_t* __restrict__ K, float* __restrict__ Output_partial, half_t* __restrict__ Q, half_t* __restrict__ V, int* __restrict__ block_indices, int* __restrict__ cache_seqlens, float* __restrict__ glse, int max_cache_seqlen, int max_selected_blocks, int num_split);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel_1(half_t* __restrict__ Output, float* __restrict__ Output_partial, float* __restrict__ glse, int num_split) {
  float lse_logsum_local[1];
  float o_accum_local[4];
  float lse_max_local[1];
  float lse_local_split[1];
  int max_split[1];
  float po_local[4];
  float scale_local[1];
  lse_logsum_local[0] = 0.000000e+00f;
  *(float4*)(o_accum_local + 0) = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
  lse_max_local[0] = -CUDART_INF_F;
  for (int k = 0; k < num_split; ++k) {
    lse_local_split[0] = glse[((((int)blockIdx.x) * num_split) + k)];
    if (lse_local_split[0] != 0.000000e+00f) {
      max_split[0] = k;
      lse_max_local[0] = max(lse_max_local[0], glse[((((int)blockIdx.x) * num_split) + k)]);
    }
  }
  for (int k_1 = 0; k_1 < (num_split - 1); ++k_1) {
    if (k_1 <= max_split[0]) {
      lse_local_split[0] = glse[((((int)blockIdx.x) * num_split) + k_1)];
    }
    if (k_1 <= max_split[0]) {
      lse_logsum_local[0] = (lse_logsum_local[0] + exp2f((lse_local_split[0] - lse_max_local[0])));
    }
  }
  if (1 <= num_split) {
    if (num_split <= (max_split[0] + 1)) {
      lse_local_split[0] = glse[((num_split * (((int)blockIdx.x) + 1)) - 1)];
    }
  }
  if (1 <= num_split) {
    if (num_split <= (max_split[0] + 1)) {
      lse_logsum_local[0] = (lse_logsum_local[0] + exp2f((lse_local_split[0] - lse_max_local[0])));
    }
  }
  lse_logsum_local[0] = (__log2f(lse_logsum_local[0]) + lse_max_local[0]);
  for (int k_2 = 0; k_2 < num_split; ++k_2) {
    if (k_2 <= max_split[0]) {
      if (((int)threadIdx.x) < 32) {
        *(float4*)(po_local + 0) = *(float4*)(Output_partial + ((((((int)blockIdx.x) * num_split) * 128) + (k_2 * 128)) + (((int)threadIdx.x) * 4)));
      }
      lse_local_split[0] = glse[((((int)blockIdx.x) * num_split) + k_2)];
      scale_local[0] = exp2f((lse_local_split[0] - lse_logsum_local[0]));
      if (((int)threadIdx.x) < 32) {
        float4 __1;
          float4 v_ = *(float4*)(o_accum_local + 0);
          float4 __2;
            float4 v__1 = *(float4*)(po_local + 0);
            float4 v__2 = make_float4(scale_local[0], scale_local[0], scale_local[0], scale_local[0]);
            __2.x = (v__1.x*v__2.x);
            __2.y = (v__1.y*v__2.y);
            __2.z = (v__1.z*v__2.z);
            __2.w = (v__1.w*v__2.w);
          __1.x = (v_.x+__2.x);
          __1.y = (v_.y+__2.y);
          __1.z = (v_.z+__2.z);
          __1.w = (v_.w+__2.w);
        *(float4*)(o_accum_local + 0) = __1;
      }
    }
  }
  if (((int)threadIdx.x) < 32) {
    uint2 __3;
    float4 v__3 = *(float4*)(o_accum_local + 0);
    ((half2*)(&(__3.x)))->x = (half_t)(v__3.x);
    ((half2*)(&(__3.x)))->y = (half_t)(v__3.y);
    ((half2*)(&(__3.y)))->x = (half_t)(v__3.z);
    ((half2*)(&(__3.y)))->y = (half_t)(v__3.w);
    *(uint2*)(Output + ((((int)blockIdx.x) * 128) + (((int)threadIdx.x) * 4))) = __3;
  }
}

extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(half_t* __restrict__ K, float* __restrict__ Output_partial, half_t* __restrict__ Q, half_t* __restrict__ V, int* __restrict__ block_indices, int* __restrict__ cache_seqlens, float* __restrict__ glse, int max_cache_seqlen, int max_selected_blocks, int num_split) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[64];
  float logsum[2];
  float scores_max[2];
  signed char has_valid_block = (signed char)0;
  float acc_s[16];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  half_t acc_s_cast[16];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint4 condval;
    if (((((i * 2) + (((((int)blockIdx.y) * 7) + (((int)threadIdx.x) >> 4)) >> 2)) < 7) && (((i * 2) + (((((int)blockIdx.y) * 7) + (((int)threadIdx.x) >> 4)) >> 2)) < 7))) {
      condval = *(uint4*)(Q + (((i * 1024) + (((int)blockIdx.y) * 896)) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_half2(half_t(0.000000e+00f), half_t(0.000000e+00f)), __pack_half2(half_t(0.000000e+00f), half_t(0.000000e+00f)), __pack_half2(half_t(0.000000e+00f), half_t(0.000000e+00f)), __pack_half2(half_t(0.000000e+00f), half_t(0.000000e+00f)));
    }
    *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    *(float2*)(acc_o + (i_1 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    logsum[i_2] = 0.000000e+00f;
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    scores_max[i_3] = -CUDART_INF_F;
  }
  has_valid_block = (signed char)0;
  int condval_1;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_1 = 1;
  } else {
    condval_1 = 0;
  }
  if (0 < (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_1)) {
    if (0 <= block_indices[((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z)))]) {
      #pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+(((((block_indices[((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z)))] * 16384) + (i_4 * 4096)) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)), ((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) < max_selected_blocks) && ((((block_indices[((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z)))] * 32) + (i_4 * 8)) + (((int)threadIdx.x) >> 4)) < max_cache_seqlen)) && ((((block_indices[((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z)))] * 32) + (i_4 * 8)) + (((int)threadIdx.x) >> 4)) < max_cache_seqlen)));
      }
    }
    tl::cp_async_commit();
  }
  int condval_2;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_2 = 1;
  } else {
    condval_2 = 0;
  }
  if (0 < (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_2)) {
    if (0 <= block_indices[((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z)))]) {
      #pragma unroll
      for (int i_5 = 0; i_5 < 4; ++i_5) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), V+(((((block_indices[((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z)))] * 16384) + (i_5 * 4096)) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)), ((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) < max_selected_blocks) && ((((block_indices[((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z)))] * 32) + (i_5 * 8)) + (((int)threadIdx.x) >> 4)) < max_cache_seqlen)) && ((((block_indices[((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z)))] * 32) + (i_5 * 8)) + (((int)threadIdx.x) >> 4)) < max_cache_seqlen)));
      }
    }
    tl::cp_async_commit();
  }
  int condval_3;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_3 = 1;
  } else {
    condval_3 = 0;
  }
  if (1 < (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_3)) {
    if (0 <= block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + 1)]) {
      #pragma unroll
      for (int i_6 = 0; i_6 < 4; ++i_6) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_6 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), K+(((((block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + 1)] * 16384) + (i_6 * 4096)) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)), (((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + 1) < max_selected_blocks) && ((((block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + 1)] * 32) + (i_6 * 8)) + (((int)threadIdx.x) >> 4)) < max_cache_seqlen)) && ((((block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + 1)] * 32) + (i_6 * 8)) + (((int)threadIdx.x) >> 4)) < max_cache_seqlen)));
      }
    }
    tl::cp_async_commit();
  }
  int condval_4;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_4 = 1;
  } else {
    condval_4 = 0;
  }
  if (1 < (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_4)) {
    if (0 <= block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + 1)]) {
      #pragma unroll
      for (int i_7 = 0; i_7 < 4; ++i_7) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_7 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 40960), V+(((((block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + 1)] * 16384) + (i_7 * 4096)) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)), (((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + 1) < max_selected_blocks) && ((((block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + 1)] * 32) + (i_7 * 8)) + (((int)threadIdx.x) >> 4)) < max_cache_seqlen)) && ((((block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + 1)] * 32) + (i_7 * 8)) + (((int)threadIdx.x) >> 4)) < max_cache_seqlen)));
      }
    }
    tl::cp_async_commit();
  }
  int condval_5;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_5 = 1;
  } else {
    condval_5 = 0;
  }
  for (int k = 0; k < ((((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_5) - 2); ++k) {
    if (0 <= block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)]) {
      has_valid_block = (signed char)1;
    }
    if (0 <= block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)]) {
      #pragma unroll
      for (int i_8 = 0; i_8 < 8; ++i_8) {
        *(float2*)(acc_s + (i_8 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
      }
    }
    tl::cp_async_wait<3>();
    __syncthreads();
    if (0 <= block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)]) {
      tl::gemm_ss<64, 32, 128, 4, 1, 0, 1, 0>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 4096) + 8192)])), (&(acc_s[0])));
    }
    __syncthreads();
    if (0 <= block_indices[((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k) + 2)]) {
      #pragma unroll
      for (int i_9 = 0; i_9 < 4; ++i_9) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((k & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_9 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+(((((block_indices[((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k) + 2)] * 16384) + (i_9 * 4096)) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)), ((((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k) + 2) < max_selected_blocks) && ((((block_indices[((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k) + 2)] * 32) + (i_9 * 8)) + (((int)threadIdx.x) >> 4)) < max_cache_seqlen)) && ((((block_indices[((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k) + 2)] * 32) + (i_9 * 8)) + (((int)threadIdx.x) >> 4)) < max_cache_seqlen)));
      }
    }
    tl::cp_async_commit();
    if (0 <= block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)]) {
      if (k == 0) {
        #pragma unroll
        for (int i_10 = 0; i_10 < 4; ++i_10) {
          if (cache_seqlens[0] <= (((block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)] * 32) + (i_10 * 8)) + ((((int)threadIdx.x) & 3) * 2))) {
            *(float4*)(acc_s + (i_10 * 4)) = make_float4(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F);
          }
          if ((((block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)] * 32) + (i_10 * 8)) + ((((int)threadIdx.x) & 3) * 2)) < cache_seqlens[0]) {
            for (int vec = 0; vec < 4; ++vec) {
              float condval_6;
              if ((cache_seqlens[0] <= ((((block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)] * 32) + (i_10 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (vec & 1)))) {
                condval_6 = -CUDART_INF_F;
              } else {
                condval_6 = acc_s[((i_10 * 4) + vec)];
              }
              acc_s[((i_10 * 4) + vec)] = condval_6;
            }
          }
        }
      }
    }
    if (0 <= block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)]) {
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        scores_max_prev[i_11] = scores_max[i_11];
      }
    }
    if (0 <= block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)]) {
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        scores_max[i_12] = -CUDART_INF_F;
      }
    }
    if (0 <= block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)]) {
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        #pragma unroll
        for (int rv = 0; rv < 8; ++rv) {
          scores_max[i_13] = max(scores_max[i_13], acc_s[((((rv & 3) * 4) + (i_13 * 2)) + (rv >> 2))]);
        }
        scores_max[i_13] = tl::AllReduce<tl::MaxOp, 4, 1>::run(scores_max[i_13]);
      }
    }
    if (0 <= block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)]) {
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        float condval_7;
        if ((scores_max_prev[i_14] < scores_max[i_14])) {
          condval_7 = scores_max[i_14];
        } else {
          condval_7 = scores_max_prev[i_14];
        }
        scores_max[i_14] = condval_7;
        scores_scale[i_14] = exp2f(((scores_max_prev[i_14] * 1.275174e-01f) - (scores_max[i_14] * 1.275174e-01f)));
      }
    }
    if (0 <= block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)]) {
      #pragma unroll
      for (int i_15 = 0; i_15 < 8; ++i_15) {
        float2 __1;
        float2 __2;
          float2 __3;
            float2 v_ = *(float2*)(acc_s + (i_15 * 2));
            float2 v__1 = make_float2(1.275174e-01f, 1.275174e-01f);
            __3.x = (v_.x*v__1.x);
            __3.y = (v_.y*v__1.y);
          float2 v__2 = make_float2((scores_max[(i_15 & 1)] * 1.275174e-01f), (scores_max[(i_15 & 1)] * 1.275174e-01f));
          __2.x = (__3.x-v__2.x);
          __2.y = (__3.y-v__2.y);
        __1.x = exp2f(__2.x);
        __1.y = exp2f(__2.y);
        *(float2*)(acc_s + (i_15 * 2)) = __1;
      }
    }
    if (0 <= block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)]) {
      #pragma unroll
      for (int i_16 = 0; i_16 < 2; ++i_16) {
        scores_sum[i_16] = 0.000000e+00f;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
          scores_sum[i_16] = (scores_sum[i_16] + acc_s[((((rv_1 & 3) * 4) + (i_16 * 2)) + (rv_1 >> 2))]);
        }
        scores_sum[i_16] = tl::AllReduce<tl::SumOp, 4, 1>::run(scores_sum[i_16]);
      }
    }
    if (0 <= block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)]) {
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        logsum[i_17] = ((logsum[i_17] * scores_scale[i_17]) + scores_sum[i_17]);
      }
    }
    if (0 <= block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)]) {
      #pragma unroll
      for (int i_18 = 0; i_18 < 8; ++i_18) {
        uint1 __4;
        float2 v__3 = *(float2*)(acc_s + (i_18 * 2));
        ((half2*)(&(__4.x)))->x = (half_t)(v__3.x);
        ((half2*)(&(__4.x)))->y = (half_t)(v__3.y);
        *(uint1*)(acc_s_cast + (i_18 * 2)) = __4;
      }
    }
    if (0 <= block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)]) {
      #pragma unroll
      for (int i_19 = 0; i_19 < 32; ++i_19) {
        float2 __5;
          float2 v__4 = *(float2*)(acc_o + (i_19 * 2));
          float2 v__5 = make_float2(scores_scale[(i_19 & 1)], scores_scale[(i_19 & 1)]);
          __5.x = (v__4.x*v__5.x);
          __5.y = (v__4.y*v__5.y);
        *(float2*)(acc_o + (i_19 * 2)) = __5;
      }
    }
    tl::cp_async_wait<3>();
    __syncthreads();
    if (0 <= block_indices[(((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k)]) {
      tl::gemm_rs<64, 128, 32, 4, 1, 0, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 4096) + 16384)])), (&(acc_o[0])));
    }
    __syncthreads();
    if (0 <= block_indices[((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k) + 2)]) {
      #pragma unroll
      for (int i_20 = 0; i_20 < 4; ++i_20) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((k & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_20 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), V+(((((block_indices[((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k) + 2)] * 16384) + (i_20 * 4096)) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)), ((((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k) + 2) < max_selected_blocks) && ((((block_indices[((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k) + 2)] * 32) + (i_20 * 8)) + (((int)threadIdx.x) >> 4)) < max_cache_seqlen)) && ((((block_indices[((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + k) + 2)] * 32) + (i_20 * 8)) + (((int)threadIdx.x) >> 4)) < max_cache_seqlen)));
      }
    }
    tl::cp_async_commit();
  }
  int condval_8;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_8 = 1;
  } else {
    condval_8 = 0;
  }
  if (2 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_8)) {
    int condval_9;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_9 = 1;
    } else {
      condval_9 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_9) - 2)]) {
      has_valid_block = (signed char)1;
    }
  }
  int condval_10;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_10 = 1;
  } else {
    condval_10 = 0;
  }
  if (2 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_10)) {
    int condval_11;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_11 = 1;
    } else {
      condval_11 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_11) - 2)]) {
      #pragma unroll
      for (int i_21 = 0; i_21 < 8; ++i_21) {
        *(float2*)(acc_s + (i_21 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
      }
    }
  }
  int condval_12;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_12 = 1;
  } else {
    condval_12 = 0;
  }
  if (2 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_12)) {
    tl::cp_async_wait<3>();
    __syncthreads();
    int condval_13;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_13 = 1;
    } else {
      condval_13 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_13) - 2)]) {
      int condval_14;
      if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
        condval_14 = 1;
      } else {
        condval_14 = 0;
      }
      tl::gemm_ss<64, 32, 128, 4, 1, 0, 1, 0>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[((((((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_14) & 1) * 4096) + 8192)])), (&(acc_s[0])));
    }
  }
  int condval_15;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_15 = 1;
  } else {
    condval_15 = 0;
  }
  if (2 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_15)) {
    int condval_16;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_16 = 1;
    } else {
      condval_16 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_16) - 2)]) {
      int condval_17;
      if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
        condval_17 = 1;
      } else {
        condval_17 = 0;
      }
      if ((((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_17) == 2) {
        #pragma unroll
        for (int i_22 = 0; i_22 < 4; ++i_22) {
          int condval_18;
          if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
            condval_18 = 1;
          } else {
            condval_18 = 0;
          }
          if (cache_seqlens[0] <= (((block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_18) - 2)] * 32) + (i_22 * 8)) + ((((int)threadIdx.x) & 3) * 2))) {
            *(float4*)(acc_s + (i_22 * 4)) = make_float4(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F);
          }
          int condval_19;
          if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
            condval_19 = 1;
          } else {
            condval_19 = 0;
          }
          if ((((block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_19) - 2)] * 32) + (i_22 * 8)) + ((((int)threadIdx.x) & 3) * 2)) < cache_seqlens[0]) {
            for (int vec_1 = 0; vec_1 < 4; ++vec_1) {
              int condval_21;
              if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
                condval_21 = 1;
              } else {
                condval_21 = 0;
              }
              float condval_20;
              if ((cache_seqlens[0] <= ((((block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_21) - 2)] * 32) + (i_22 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (vec_1 & 1)))) {
                condval_20 = -CUDART_INF_F;
              } else {
                condval_20 = acc_s[((i_22 * 4) + vec_1)];
              }
              acc_s[((i_22 * 4) + vec_1)] = condval_20;
            }
          }
        }
      }
    }
  }
  int condval_22;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_22 = 1;
  } else {
    condval_22 = 0;
  }
  if (2 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_22)) {
    int condval_23;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_23 = 1;
    } else {
      condval_23 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_23) - 2)]) {
      #pragma unroll
      for (int i_23 = 0; i_23 < 2; ++i_23) {
        scores_max_prev[i_23] = scores_max[i_23];
      }
    }
  }
  int condval_24;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_24 = 1;
  } else {
    condval_24 = 0;
  }
  if (2 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_24)) {
    int condval_25;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_25 = 1;
    } else {
      condval_25 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_25) - 2)]) {
      #pragma unroll
      for (int i_24 = 0; i_24 < 2; ++i_24) {
        scores_max[i_24] = -CUDART_INF_F;
      }
    }
  }
  int condval_26;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_26 = 1;
  } else {
    condval_26 = 0;
  }
  if (2 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_26)) {
    int condval_27;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_27 = 1;
    } else {
      condval_27 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_27) - 2)]) {
      #pragma unroll
      for (int i_25 = 0; i_25 < 2; ++i_25) {
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 8; ++rv_2) {
          scores_max[i_25] = max(scores_max[i_25], acc_s[((((rv_2 & 3) * 4) + (i_25 * 2)) + (rv_2 >> 2))]);
        }
        scores_max[i_25] = tl::AllReduce<tl::MaxOp, 4, 1>::run(scores_max[i_25]);
      }
    }
  }
  int condval_28;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_28 = 1;
  } else {
    condval_28 = 0;
  }
  if (2 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_28)) {
    int condval_29;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_29 = 1;
    } else {
      condval_29 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_29) - 2)]) {
      #pragma unroll
      for (int i_26 = 0; i_26 < 2; ++i_26) {
        float condval_30;
        if ((scores_max_prev[i_26] < scores_max[i_26])) {
          condval_30 = scores_max[i_26];
        } else {
          condval_30 = scores_max_prev[i_26];
        }
        scores_max[i_26] = condval_30;
        scores_scale[i_26] = exp2f(((scores_max_prev[i_26] * 1.275174e-01f) - (scores_max[i_26] * 1.275174e-01f)));
      }
    }
  }
  int condval_31;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_31 = 1;
  } else {
    condval_31 = 0;
  }
  if (2 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_31)) {
    int condval_32;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_32 = 1;
    } else {
      condval_32 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_32) - 2)]) {
      #pragma unroll
      for (int i_27 = 0; i_27 < 8; ++i_27) {
        float2 __6;
        float2 __7;
          float2 __8;
            float2 v__6 = *(float2*)(acc_s + (i_27 * 2));
            float2 v__7 = make_float2(1.275174e-01f, 1.275174e-01f);
            __8.x = (v__6.x*v__7.x);
            __8.y = (v__6.y*v__7.y);
          float2 v__8 = make_float2((scores_max[(i_27 & 1)] * 1.275174e-01f), (scores_max[(i_27 & 1)] * 1.275174e-01f));
          __7.x = (__8.x-v__8.x);
          __7.y = (__8.y-v__8.y);
        __6.x = exp2f(__7.x);
        __6.y = exp2f(__7.y);
        *(float2*)(acc_s + (i_27 * 2)) = __6;
      }
    }
  }
  int condval_33;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_33 = 1;
  } else {
    condval_33 = 0;
  }
  if (2 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_33)) {
    int condval_34;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_34 = 1;
    } else {
      condval_34 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_34) - 2)]) {
      #pragma unroll
      for (int i_28 = 0; i_28 < 2; ++i_28) {
        scores_sum[i_28] = 0.000000e+00f;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
          scores_sum[i_28] = (scores_sum[i_28] + acc_s[((((rv_3 & 3) * 4) + (i_28 * 2)) + (rv_3 >> 2))]);
        }
        scores_sum[i_28] = tl::AllReduce<tl::SumOp, 4, 1>::run(scores_sum[i_28]);
      }
    }
  }
  int condval_35;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_35 = 1;
  } else {
    condval_35 = 0;
  }
  if (2 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_35)) {
    int condval_36;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_36 = 1;
    } else {
      condval_36 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_36) - 2)]) {
      #pragma unroll
      for (int i_29 = 0; i_29 < 2; ++i_29) {
        logsum[i_29] = ((logsum[i_29] * scores_scale[i_29]) + scores_sum[i_29]);
      }
    }
  }
  int condval_37;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_37 = 1;
  } else {
    condval_37 = 0;
  }
  if (2 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_37)) {
    int condval_38;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_38 = 1;
    } else {
      condval_38 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_38) - 2)]) {
      #pragma unroll
      for (int i_30 = 0; i_30 < 8; ++i_30) {
        uint1 __9;
        float2 v__9 = *(float2*)(acc_s + (i_30 * 2));
        ((half2*)(&(__9.x)))->x = (half_t)(v__9.x);
        ((half2*)(&(__9.x)))->y = (half_t)(v__9.y);
        *(uint1*)(acc_s_cast + (i_30 * 2)) = __9;
      }
    }
  }
  int condval_39;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_39 = 1;
  } else {
    condval_39 = 0;
  }
  if (2 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_39)) {
    int condval_40;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_40 = 1;
    } else {
      condval_40 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_40) - 2)]) {
      #pragma unroll
      for (int i_31 = 0; i_31 < 32; ++i_31) {
        float2 __10;
          float2 v__10 = *(float2*)(acc_o + (i_31 * 2));
          float2 v__11 = make_float2(scores_scale[(i_31 & 1)], scores_scale[(i_31 & 1)]);
          __10.x = (v__10.x*v__11.x);
          __10.y = (v__10.y*v__11.y);
        *(float2*)(acc_o + (i_31 * 2)) = __10;
      }
    }
  }
  int condval_41;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_41 = 1;
  } else {
    condval_41 = 0;
  }
  if (2 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_41)) {
    tl::cp_async_wait<2>();
    __syncthreads();
    int condval_42;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_42 = 1;
    } else {
      condval_42 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_42) - 2)]) {
      int condval_43;
      if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
        condval_43 = 1;
      } else {
        condval_43 = 0;
      }
      tl::gemm_rs<64, 128, 32, 4, 1, 0, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[((((((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_43) & 1) * 4096) + 16384)])), (&(acc_o[0])));
    }
  }
  int condval_44;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_44 = 1;
  } else {
    condval_44 = 0;
  }
  if (1 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_44)) {
    int condval_45;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_45 = 1;
    } else {
      condval_45 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_45) - 1)]) {
      has_valid_block = (signed char)1;
    }
  }
  int condval_46;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_46 = 1;
  } else {
    condval_46 = 0;
  }
  if (1 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_46)) {
    int condval_47;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_47 = 1;
    } else {
      condval_47 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_47) - 1)]) {
      #pragma unroll
      for (int i_32 = 0; i_32 < 8; ++i_32) {
        *(float2*)(acc_s + (i_32 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
      }
    }
  }
  int condval_48;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_48 = 1;
  } else {
    condval_48 = 0;
  }
  if (1 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_48)) {
    tl::cp_async_wait<1>();
    __syncthreads();
    int condval_49;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_49 = 1;
    } else {
      condval_49 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_49) - 1)]) {
      int condval_50;
      if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
        condval_50 = 1;
      } else {
        condval_50 = 0;
      }
      tl::gemm_ss<64, 32, 128, 4, 1, 0, 1, 0>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[(((((((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_50) + 1) & 1) * 4096) + 8192)])), (&(acc_s[0])));
    }
  }
  int condval_51;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_51 = 1;
  } else {
    condval_51 = 0;
  }
  if (1 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_51)) {
    int condval_52;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_52 = 1;
    } else {
      condval_52 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_52) - 1)]) {
      int condval_53;
      if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
        condval_53 = 1;
      } else {
        condval_53 = 0;
      }
      if ((((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_53) == 1) {
        #pragma unroll
        for (int i_33 = 0; i_33 < 4; ++i_33) {
          int condval_54;
          if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
            condval_54 = 1;
          } else {
            condval_54 = 0;
          }
          if (cache_seqlens[0] <= (((block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_54) - 1)] * 32) + (i_33 * 8)) + ((((int)threadIdx.x) & 3) * 2))) {
            *(float4*)(acc_s + (i_33 * 4)) = make_float4(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F);
          }
          int condval_55;
          if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
            condval_55 = 1;
          } else {
            condval_55 = 0;
          }
          if ((((block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_55) - 1)] * 32) + (i_33 * 8)) + ((((int)threadIdx.x) & 3) * 2)) < cache_seqlens[0]) {
            for (int vec_2 = 0; vec_2 < 4; ++vec_2) {
              int condval_57;
              if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
                condval_57 = 1;
              } else {
                condval_57 = 0;
              }
              float condval_56;
              if ((cache_seqlens[0] <= ((((block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_57) - 1)] * 32) + (i_33 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (vec_2 & 1)))) {
                condval_56 = -CUDART_INF_F;
              } else {
                condval_56 = acc_s[((i_33 * 4) + vec_2)];
              }
              acc_s[((i_33 * 4) + vec_2)] = condval_56;
            }
          }
        }
      }
    }
  }
  int condval_58;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_58 = 1;
  } else {
    condval_58 = 0;
  }
  if (1 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_58)) {
    int condval_59;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_59 = 1;
    } else {
      condval_59 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_59) - 1)]) {
      #pragma unroll
      for (int i_34 = 0; i_34 < 2; ++i_34) {
        scores_max_prev[i_34] = scores_max[i_34];
      }
    }
  }
  int condval_60;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_60 = 1;
  } else {
    condval_60 = 0;
  }
  if (1 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_60)) {
    int condval_61;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_61 = 1;
    } else {
      condval_61 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_61) - 1)]) {
      #pragma unroll
      for (int i_35 = 0; i_35 < 2; ++i_35) {
        scores_max[i_35] = -CUDART_INF_F;
      }
    }
  }
  int condval_62;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_62 = 1;
  } else {
    condval_62 = 0;
  }
  if (1 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_62)) {
    int condval_63;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_63 = 1;
    } else {
      condval_63 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_63) - 1)]) {
      #pragma unroll
      for (int i_36 = 0; i_36 < 2; ++i_36) {
        #pragma unroll
        for (int rv_4 = 0; rv_4 < 8; ++rv_4) {
          scores_max[i_36] = max(scores_max[i_36], acc_s[((((rv_4 & 3) * 4) + (i_36 * 2)) + (rv_4 >> 2))]);
        }
        scores_max[i_36] = tl::AllReduce<tl::MaxOp, 4, 1>::run(scores_max[i_36]);
      }
    }
  }
  int condval_64;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_64 = 1;
  } else {
    condval_64 = 0;
  }
  if (1 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_64)) {
    int condval_65;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_65 = 1;
    } else {
      condval_65 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_65) - 1)]) {
      #pragma unroll
      for (int i_37 = 0; i_37 < 2; ++i_37) {
        float condval_66;
        if ((scores_max_prev[i_37] < scores_max[i_37])) {
          condval_66 = scores_max[i_37];
        } else {
          condval_66 = scores_max_prev[i_37];
        }
        scores_max[i_37] = condval_66;
        scores_scale[i_37] = exp2f(((scores_max_prev[i_37] * 1.275174e-01f) - (scores_max[i_37] * 1.275174e-01f)));
      }
    }
  }
  int condval_67;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_67 = 1;
  } else {
    condval_67 = 0;
  }
  if (1 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_67)) {
    int condval_68;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_68 = 1;
    } else {
      condval_68 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_68) - 1)]) {
      #pragma unroll
      for (int i_38 = 0; i_38 < 8; ++i_38) {
        float2 __11;
        float2 __12;
          float2 __13;
            float2 v__12 = *(float2*)(acc_s + (i_38 * 2));
            float2 v__13 = make_float2(1.275174e-01f, 1.275174e-01f);
            __13.x = (v__12.x*v__13.x);
            __13.y = (v__12.y*v__13.y);
          float2 v__14 = make_float2((scores_max[(i_38 & 1)] * 1.275174e-01f), (scores_max[(i_38 & 1)] * 1.275174e-01f));
          __12.x = (__13.x-v__14.x);
          __12.y = (__13.y-v__14.y);
        __11.x = exp2f(__12.x);
        __11.y = exp2f(__12.y);
        *(float2*)(acc_s + (i_38 * 2)) = __11;
      }
    }
  }
  int condval_69;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_69 = 1;
  } else {
    condval_69 = 0;
  }
  if (1 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_69)) {
    int condval_70;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_70 = 1;
    } else {
      condval_70 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_70) - 1)]) {
      #pragma unroll
      for (int i_39 = 0; i_39 < 2; ++i_39) {
        scores_sum[i_39] = 0.000000e+00f;
        #pragma unroll
        for (int rv_5 = 0; rv_5 < 8; ++rv_5) {
          scores_sum[i_39] = (scores_sum[i_39] + acc_s[((((rv_5 & 3) * 4) + (i_39 * 2)) + (rv_5 >> 2))]);
        }
        scores_sum[i_39] = tl::AllReduce<tl::SumOp, 4, 1>::run(scores_sum[i_39]);
      }
    }
  }
  int condval_71;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_71 = 1;
  } else {
    condval_71 = 0;
  }
  if (1 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_71)) {
    int condval_72;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_72 = 1;
    } else {
      condval_72 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_72) - 1)]) {
      #pragma unroll
      for (int i_40 = 0; i_40 < 2; ++i_40) {
        logsum[i_40] = ((logsum[i_40] * scores_scale[i_40]) + scores_sum[i_40]);
      }
    }
  }
  int condval_73;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_73 = 1;
  } else {
    condval_73 = 0;
  }
  if (1 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_73)) {
    int condval_74;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_74 = 1;
    } else {
      condval_74 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_74) - 1)]) {
      #pragma unroll
      for (int i_41 = 0; i_41 < 8; ++i_41) {
        uint1 __14;
        float2 v__15 = *(float2*)(acc_s + (i_41 * 2));
        ((half2*)(&(__14.x)))->x = (half_t)(v__15.x);
        ((half2*)(&(__14.x)))->y = (half_t)(v__15.y);
        *(uint1*)(acc_s_cast + (i_41 * 2)) = __14;
      }
    }
  }
  int condval_75;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_75 = 1;
  } else {
    condval_75 = 0;
  }
  if (1 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_75)) {
    int condval_76;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_76 = 1;
    } else {
      condval_76 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_76) - 1)]) {
      #pragma unroll
      for (int i_42 = 0; i_42 < 32; ++i_42) {
        float2 __15;
          float2 v__16 = *(float2*)(acc_o + (i_42 * 2));
          float2 v__17 = make_float2(scores_scale[(i_42 & 1)], scores_scale[(i_42 & 1)]);
          __15.x = (v__16.x*v__17.x);
          __15.y = (v__16.y*v__17.y);
        *(float2*)(acc_o + (i_42 * 2)) = __15;
      }
    }
  }
  int condval_77;
  if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
    condval_77 = 1;
  } else {
    condval_77 = 0;
  }
  if (1 <= (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_77)) {
    tl::cp_async_wait<0>();
    __syncthreads();
    int condval_78;
    if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
      condval_78 = 1;
    } else {
      condval_78 = 0;
    }
    if (0 <= block_indices[(((((min(((int)blockIdx.z), ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split))) + (((int)blockIdx.y) * max_selected_blocks)) + (((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) * ((int)blockIdx.z))) + ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1))) + condval_78) - 1)]) {
      int condval_79;
      if ((((int)blockIdx.z) < ((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks % num_split) : ((max_selected_blocks % num_split) + num_split)))) {
        condval_79 = 1;
      } else {
        condval_79 = 0;
      }
      tl::gemm_rs<64, 128, 32, 4, 1, 0, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[(((((((((0 <= num_split) && (0 <= (max_selected_blocks % num_split))) || ((num_split < 0) && ((max_selected_blocks % num_split) <= 0))) ? (max_selected_blocks / num_split) : ((max_selected_blocks / num_split) - 1)) + condval_79) + 1) & 1) * 4096) + 16384)])), (&(acc_o[0])));
    }
  }
  if ((bool)has_valid_block) {
    #pragma unroll
    for (int i_43 = 0; i_43 < 32; ++i_43) {
      float2 __16;
        float2 v__18 = *(float2*)(acc_o + (i_43 * 2));
        float2 v__19 = make_float2(logsum[(i_43 & 1)], logsum[(i_43 & 1)]);
        __16.x = (v__18.x/v__19.x);
        __16.y = (v__18.y/v__19.y);
      *(float2*)(acc_o + (i_43 * 2)) = __16;
    }
    #pragma unroll
    for (int i_44 = 0; i_44 < 2; ++i_44) {
      logsum[i_44] = (__log2f(logsum[i_44]) + (scores_max[i_44] * 1.275174e-01f));
    }
  }
  if (((((int)threadIdx.x) & 3) >> 1) == 0) {
    #pragma unroll
    for (int i_45 = 0; i_45 < 1; ++i_45) {
      if (((((((int)threadIdx.x) >> 5) * 16) + ((((int)threadIdx.x) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < 7) {
        if ((((((((int)threadIdx.x) >> 5) * 16) + ((((int)threadIdx.x) & 1) * 8)) + (((int)blockIdx.y) * 7)) + ((((int)threadIdx.x) & 31) >> 2)) < 28) {
          glse[(((((((((int)threadIdx.x) >> 5) * 16) + ((((int)threadIdx.x) & 1) * 8)) + (((int)blockIdx.y) * 7)) + ((((int)threadIdx.x) & 31) >> 2)) * num_split) + ((int)blockIdx.z))] = logsum[(((int)threadIdx.x) & 1)];
        }
      }
    }
  }
  #pragma unroll
  for (int i_46 = 0; i_46 < 32; ++i_46) {
    if (((((((int)threadIdx.x) >> 5) * 16) + ((i_46 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < 7) {
      if ((((((((int)threadIdx.x) >> 5) * 16) + ((i_46 & 1) * 8)) + (((int)blockIdx.y) * 7)) + ((((int)threadIdx.x) & 31) >> 2)) < 28) {
        *(float2*)(Output_partial + ((((((((((((int)threadIdx.x) >> 5) * 16) + ((i_46 & 1) * 8)) + (((int)blockIdx.y) * 7)) + ((((int)threadIdx.x) & 31) >> 2)) * num_split) * 128) + (((int)blockIdx.z) * 128)) + ((i_46 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_o + (i_46 * 2));
      }
    }
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
    if (result_main_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 49152, cudaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

// extern "C" int call(half_t* __restrict__ Q, half_t* __restrict__ K, half_t* __restrict__ V, int* __restrict__ block_indices, int* __restrict__ cache_seqlens, float* __restrict__ glse, float* __restrict__ Output_partial, half_t* __restrict__ Output, int max_cache_seqlen, int max_selected_blocks, int num_split, int batch, int heads, int heads_k, cudaStream_t stream=cudaStreamDefault) {
// 	cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
//   main_kernel<<<dim3(batch, heads_k, num_split), dim3(128, 1, 1), 49152, stream>>>(K, Output_partial, Q, V, block_indices, cache_seqlens, glse, max_cache_seqlen, max_selected_blocks, num_split);
// 	main_kernel_1<<<dim3(heads, 1, 1), dim3(128, 1, 1), 0, stream>>>(Output, Output_partial, glse, num_split);

//     return 0;
// }

extern "C" int call(void* Q, void*  K, void*  V, void*  block_indices, void*  cache_seqlens, void*  glse, void*  Output_partial, void* Output, int max_cache_seqlen, int max_selected_blocks, int num_split, int batch, int heads, int heads_k, cudaStream_t stream=cudaStreamDefault) {
	cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
  main_kernel<<<dim3(batch, heads_k, num_split), dim3(128, 1, 1), 49152, stream>>>(reinterpret_cast<half_t*>(K), reinterpret_cast<float*>(Output_partial), reinterpret_cast<half_t*>(Q), reinterpret_cast<half_t*>(V), 
  reinterpret_cast<int*>(block_indices), reinterpret_cast<int*>(cache_seqlens), reinterpret_cast<float*>(glse), max_cache_seqlen, max_selected_blocks, num_split);
	main_kernel_1<<<dim3(heads, 1, 1), dim3(128, 1, 1), 0, stream>>>(reinterpret_cast<half_t*>(Output), reinterpret_cast<float*>(Output_partial), reinterpret_cast<float*>(glse), num_split);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

    return 0;
}
