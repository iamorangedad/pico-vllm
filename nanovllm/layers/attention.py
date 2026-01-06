import torch
from torch import nn
from torch.nn import functional as F
from nanovllm.utils.context import get_context


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    flat_k_cache = k_cache.view(-1, num_heads, head_dim)
    flat_v_cache = v_cache.view(-1, num_heads, head_dim)
    valid_mask = slot_mapping != -1
    if not valid_mask.any():
        return
    valid_slots = slot_mapping[valid_mask].long()
    flat_k_cache[valid_slots] = key[valid_mask]
    flat_v_cache[valid_slots] = value[valid_mask]


def varlen_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
):
    """
    Replacement for flash_attn_varlen_func using loop + standard SDPA.
    Since 'q' is flattened (packed), we iterate via cu_seqlens.
    """
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    output = torch.empty_like(q)

    # Iterate over each sample in the batch
    for i in range(len(cu_seqlens_q) - 1):
        start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i + 1]

        # Slice the packed tensors
        q_i = q[start_q:end_q].unsqueeze(0).transpose(1, 2)  # (1, Heads, Seq_Q, Dim)
        k_i = k[start_k:end_k].unsqueeze(0).transpose(1, 2)  # (1, KV_Heads, Seq_K, Dim)
        v_i = v[start_k:end_k].unsqueeze(0).transpose(1, 2)  # (1, KV_Heads, Seq_K, Dim)

        # Causal Masking (if needed, usually prefill is causal)
        L, S = q_i.size(-2), k_i.size(-2)
        attn_mask = torch.ones(L, S, dtype=torch.bool, device=q.device).tril(diagonal=0)

        # Standard Attention
        o_i = F.scaled_dot_product_attention(
            q_i, k_i, v_i, attn_mask=attn_mask, scale=scale
        )

        # Write back
        output[start_q:end_q] = o_i.transpose(1, 2).squeeze(0)

    return output


def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
):
    """
    Native PyTorch implementation of PagedAttention for decoding.
    """
    # q shape: (Batch, 1, Num_Heads, Head_Dim) or (Batch, Num_Heads, Head_Dim)
    if q.dim() == 3:
        q = q.unsqueeze(2)  # Ensure (Batch, Num_Heads, 1, Head_Dim) for SDPA

    q = q.transpose(1, 2)  # (Batch, Num_Heads, 1, Head_Dim)

    batch_size = block_tables.shape[0]
    block_size = k_cache.shape[1]

    # 1. READ BLOCKS: Use block_tables to gather the physical blocks
    # block_tables: (Batch, Max_Num_Blocks)
    # k_cache: (Total_Physical_Blocks, Block_Size, KV_Heads, Head_Dim)

    # Expand indices for gather: (Batch, Max_Num_Blocks, Block_Size, KV_Heads, Head_Dim)
    # This gathers all relevant blocks for each batch item into a contiguous temporary tensor
    valid_blocks = block_tables.long()

    # Gather K and V
    # Result: (Batch, Max_Num_Blocks, Block_Size, KV_Heads, Head_Dim)
    k_gathered = k_cache[valid_blocks]
    v_gathered = v_cache[valid_blocks]

    # 2. RESHAPE: Flatten the blocks into a sequence
    # Result: (Batch, Max_Logical_Seq_Len, KV_Heads, Head_Dim)
    # Note: Max_Logical_Seq_Len = Max_Num_Blocks * Block_Size
    k_seq = k_gathered.view(batch_size, -1, k_gathered.shape[-2], k_gathered.shape[-1])
    v_seq = v_gathered.view(batch_size, -1, v_gathered.shape[-2], v_gathered.shape[-1])

    # Transpose for SDPA: (Batch, KV_Heads, Seq_Len, Head_Dim)
    k_seq = k_seq.transpose(1, 2)
    v_seq = v_seq.transpose(1, 2)

    # 3. MASKING: The gathered blocks include padding (trash data) at the end.
    # We need to mask out positions beyond context_lens.
    max_seq_len = k_seq.shape[2]

    # Create mask: (Batch, 1, 1, Max_Seq_Len)
    mask = torch.arange(max_seq_len, device=q.device).expand(batch_size, max_seq_len)
    mask = mask < context_lens.unsqueeze(1)
    mask = mask[:, None, None, :]  # Expand for heads and query dim

    # 4. COMPUTE: Scaled Dot Product Attention
    # PyTorch 2.0+ scaled_dot_product_attention handles GQA broadcasting automatically
    # if q has M heads and k has N heads (where M is a multiple of N).
    o = F.scaled_dot_product_attention(q, k_seq, v_seq, attn_mask=mask, scale=scale)

    # Output: (Batch, Num_Heads, 1, Head_Dim) -> Squeeze to (Batch, Num_Heads, Head_Dim)
    return o.transpose(1, 2).squeeze(2)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache
                pass
            o = varlen_attention_prefill(
                q,
                k,
                v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                scale=self.scale,
            )
        else:  # decode
            o = paged_attention_decode(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                block_table=context.block_tables,
                context_lens=context.context_lens,
                scale=self.scale,
            )
        return o
