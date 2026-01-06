import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load_inline
from nanovllm.utils.context import get_context

# ==========================================
# 1. CUDA C++ Source Code (Fixed)
# ==========================================
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE_KV 16
#define WARP_SIZE 32

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void paged_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_Cache,
    const float* __restrict__ V_Cache,
    const int* __restrict__ Block_Tables,
    const int* __restrict__ Context_Lens,
    float* __restrict__ Out,
    const float scale,
    const int num_kv_heads,
    const int head_dim,
    const int max_num_blocks,
    const int q_stride_batch, const int q_stride_head,
    const int k_stride_block, const int k_stride_tok, const int k_stride_head,
    const int v_stride_block, const int v_stride_tok, const int v_stride_head,
    const int bt_stride_batch, const int bt_stride_block,
    const int out_stride_batch, const int out_stride_head
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    int num_q_heads = gridDim.y;
    int group_size = num_q_heads / num_kv_heads;
    int kv_head_idx = head_idx / group_size;

    extern __shared__ float s_Q[];
    
    long long q_offset = (long long)batch_idx * q_stride_batch + (long long)head_idx * q_stride_head + tid;
    float q_val = Q[q_offset];
    
    if (tid < head_dim) {
        s_Q[tid] = q_val; 
    }
    __syncthreads();

    float m_i = -1e20f;
    float l_i = 0.0f;
    float acc_val = 0.0f;

    int context_len = Context_Lens[batch_idx];
    int num_blocks = (context_len + BLOCK_SIZE_KV - 1) / BLOCK_SIZE_KV;

    for (int i = 0; i < num_blocks; ++i) {
        long long bt_offset = (long long)batch_idx * bt_stride_batch + (long long)i * bt_stride_block;
        int physical_block = Block_Tables[bt_offset];

        for (int j = 0; j < BLOCK_SIZE_KV; ++j) {
            int current_pos = i * BLOCK_SIZE_KV + j;
            if (current_pos >= context_len) break;

            long long k_offset = (long long)physical_block * k_stride_block + 
                                 (long long)j * k_stride_tok + 
                                 (long long)kv_head_idx * k_stride_head + tid;
            
            float k_val = (tid < head_dim) ? K_Cache[k_offset] : 0.0f;
            
            float dot = (tid < head_dim) ? (s_Q[tid] * k_val) : 0.0f;
            
            __shared__ float s_reduce[1024]; 
            s_reduce[tid] = dot;
            __syncthreads();
            
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    s_reduce[tid] += s_reduce[tid + s];
                }
                __syncthreads();
            }
            float score = s_reduce[0];
            
            score *= scale;

            // FIX: Use fmaxf instead of max to avoid ambiguity
            float m_prev = m_i;
            m_i = fmaxf(m_prev, score);
            float alpha = expf(m_prev - m_i);
            float p = expf(score - m_i);

            l_i = l_i * alpha + p;

            acc_val *= alpha;
            
            long long v_offset = (long long)physical_block * v_stride_block + 
                                 (long long)j * v_stride_tok + 
                                 (long long)kv_head_idx * v_stride_head + tid;
            float v_val = (tid < head_dim) ? V_Cache[v_offset] : 0.0f;

            acc_val += p * v_val;
        }
    }

    acc_val /= l_i;
    
    if (tid < head_dim) {
        long long out_offset = (long long)batch_idx * out_stride_batch + (long long)head_idx * out_stride_head + tid;
        Out[out_offset] = acc_val;
    }
}
"""

cpp_source = """
torch::Tensor paged_attention_cuda(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    double scale) {
    
    auto batch_size = q.size(0);
    auto num_heads = q.size(1);
    auto head_dim = q.size(2);
    
    auto num_kv_heads = k_cache.size(2);
    auto max_num_blocks = block_tables.size(1);

    auto options = torch::TensorOptions().dtype(q.dtype()).device(q.device());
    auto out = torch::empty({batch_size, num_heads, head_dim}, options);

    dim3 grid(batch_size, num_heads);
    dim3 block(head_dim);
    int shared_mem_size = head_dim * sizeof(float);

    paged_attention_kernel<<<grid, block, shared_mem_size>>>(
        q.data_ptr<float>(),
        k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(),
        block_tables.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        out.data_ptr<float>(),
        (float)scale,
        num_kv_heads,
        head_dim,
        max_num_blocks,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        out.stride(0), out.stride(1)
    );

    return out;
}
"""

# Compile with verbose=True to see errors if they happen
paged_attn_cuda = load_inline(
    name="paged_attention_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["paged_attention_cuda"],
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    verbose=True,
)


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if context.is_prefill:
            if q.dim() == 4:
                q = q.flatten(0, 1)
            if k.dim() == 4:
                k = k.flatten(0, 1)
            if v.dim() == 4:
                v = v.flatten(0, 1)

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            o = varlen_attention_prefill(
                q,
                k,
                v,
                context.cu_seqlens_q,
                context.cu_seqlens_k,
                context.max_seqlen_q,
                context.max_seqlen_k,
                self.scale,
            )
        else:
            q_input = q.squeeze(1) if q.dim() == 4 else q

            original_dtype = q_input.dtype
            if q_input.dtype != torch.float32:
                q_input = q_input.float()

            o = paged_attn_cuda.paged_attention_cuda(
                q_input,
                k_cache.float(),
                v_cache.float(),
                context.block_tables.int(),
                context.context_lens.int(),
                self.scale,
            )

            o = o.unsqueeze(1).to(original_dtype)

        return o


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    if key.dim() == 4:
        key, value = key.flatten(0, 1), value.flatten(0, 1)
    N, num_heads, head_dim = key.shape
    flat_k = k_cache.view(-1, num_heads, head_dim)
    flat_v = v_cache.view(-1, num_heads, head_dim)
    mask = slot_mapping != -1
    if mask.any():
        slots = slot_mapping[mask].long()
        flat_k[slots] = key[mask]
        flat_v[slots] = value[mask]


def varlen_attention_prefill(q, k, v, cu_q, cu_k, max_q, max_k, scale):
    out = torch.empty_like(q)
    for i in range(len(cu_q) - 1):
        q_i = q[cu_q[i] : cu_q[i + 1]].unsqueeze(0).transpose(1, 2)
        k_i = k[cu_k[i] : cu_k[i + 1]].unsqueeze(0).transpose(1, 2)
        v_i = v[cu_k[i] : cu_k[i + 1]].unsqueeze(0).transpose(1, 2)

        if q_i.shape[1] != k_i.shape[1]:
            rep = q_i.shape[1] // k_i.shape[1]
            k_i = k_i.repeat_interleave(rep, dim=1)
            v_i = v_i.repeat_interleave(rep, dim=1)

        L, S = q_i.size(-2), k_i.size(-2)
        attn_mask = torch.ones(L, S, dtype=torch.bool, device=q.device).tril(diagonal=0)
        o_i = F.scaled_dot_product_attention(
            q_i, k_i, v_i, attn_mask=attn_mask, scale=scale
        )
        out[cu_q[i] : cu_q[i + 1]] = o_i.transpose(1, 2).squeeze(0)
    return out
