import torch
from torch import nn
from torch.nn import functional as F
from nanovllm.utils.context import get_context


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats the Key/Value heads to match the Query heads (for GQA).
    Input:  (Batch, Num_KV_Heads, SeqLen, Head_Dim)
    Output: (Batch, Num_Query_Heads, SeqLen, Head_Dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    # If key/value are 4D (Batch, Seq, Heads, Dim), flatten to (Total_Tokens, Heads, Dim)
    if key.dim() == 4:
        key = key.flatten(0, 1)
        value = value.flatten(0, 1)

    N, num_heads, head_dim = key.shape
    flat_k_cache = k_cache.view(-1, num_heads, head_dim)
    flat_v_cache = v_cache.view(-1, num_heads, head_dim)
    valid_mask = slot_mapping != -1
    if not valid_mask.any():
        return
    valid_slots = slot_mapping[valid_mask].long()
    flat_k_cache[valid_slots] = key[valid_mask]
    flat_v_cache[valid_slots] = value[valid_mask]


def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
):
    # q shape: (Batch, 1, Num_Heads, Head_Dim) or (Batch, Num_Heads, Head_Dim)
    if q.dim() == 3:
        q = q.unsqueeze(2)  # Ensure (Batch, Num_Heads, 1, Head_Dim)
    elif q.dim() == 4 and q.shape[1] != 1:
        # Case where q is (Batch, 1, Heads, Dim) coming in as (Batch, 1, Heads, Dim)
        # We need (Batch, Heads, 1, Dim) for SDPA
        q = q.transpose(1, 2)
    elif q.dim() == 4 and q.shape[2] == 1:
        # Case where q is (Batch, Heads, 1, Dim) - already correct for SDPA
        pass
    else:
        # Standardize to (Batch, Heads, 1, Dim)
        # Assuming input is (Batch, 1, Heads, Dim)
        q = q.transpose(1, 2)

    batch_size = block_tables.shape[0]

    # 1. GATHER BLOCKS
    valid_blocks = block_tables.long()
    k_gathered = k_cache[valid_blocks]
    v_gathered = v_cache[valid_blocks]

    # 2. RESHAPE TO SEQUENCE
    k_seq = k_gathered.view(batch_size, -1, k_gathered.shape[-2], k_gathered.shape[-1])
    v_seq = v_gathered.view(batch_size, -1, v_gathered.shape[-2], v_gathered.shape[-1])

    # Transpose for SDPA: (Batch, KV_Heads, Seq_Len, Head_Dim)
    k_seq = k_seq.transpose(1, 2)
    v_seq = v_seq.transpose(1, 2)

    # --- GQA Handle ---
    num_q_heads = q.shape[1]
    num_kv_heads = k_seq.shape[1]
    if num_q_heads != num_kv_heads:
        n_rep = num_q_heads // num_kv_heads
        k_seq = repeat_kv(k_seq, n_rep)
        v_seq = repeat_kv(v_seq, n_rep)

    # 3. MASKING
    max_seq_len = k_seq.shape[2]
    mask = torch.arange(max_seq_len, device=q.device).expand(batch_size, max_seq_len)
    mask = mask < context_lens.unsqueeze(1)
    mask = mask[:, None, None, :]

    # 4. COMPUTE
    # q: (Batch, Heads, 1, Dim)
    # k: (Batch, Heads, Seq, Dim)
    o = F.scaled_dot_product_attention(q, k_seq, v_seq, attn_mask=mask, scale=scale)

    # Output: (Batch, Heads, 1, Dim)
    # Target: (Batch, 1, Heads, Dim)
    o = o.transpose(1, 2)
    return o


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
    Inputs q, k, v are expected to be PACKED: (Total_Tokens, Heads, Dim).
    """
    output = torch.empty_like(q)

    for i in range(len(cu_seqlens_q) - 1):
        start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i + 1]

        # Extract sequence
        q_i = q[start_q:end_q].unsqueeze(0).transpose(1, 2)  # (1, Heads, Seq_Q, Dim)
        k_i = k[start_k:end_k].unsqueeze(0).transpose(1, 2)
        v_i = v[start_k:end_k].unsqueeze(0).transpose(1, 2)

        # --- GQA Handle ---
        num_q_heads = q_i.shape[1]
        num_kv_heads = k_i.shape[1]
        if num_q_heads != num_kv_heads:
            n_rep = num_q_heads // num_kv_heads
            k_i = repeat_kv(k_i, n_rep)
            v_i = repeat_kv(v_i, n_rep)

        # Causal Mask
        L, S = q_i.size(-2), k_i.size(-2)
        attn_mask = torch.ones(L, S, dtype=torch.bool, device=q.device).tril(diagonal=0)

        o_i = F.scaled_dot_product_attention(
            q_i, k_i, v_i, attn_mask=attn_mask, scale=scale
        )

        # Write back
        # o_i: (1, Heads, Seq, Dim) -> (1, Seq, Heads, Dim) -> (Seq, Heads, Dim)
        output[start_q:end_q] = o_i.transpose(1, 2).squeeze(0)

    return output


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

        # --- FIX: Ensure inputs are flattened for prefill ---
        # nanovllm passes (Batch, Seq, Heads, Dim), but we want (Total_Tokens, Heads, Dim)
        # to mimic flash_attn_varlen behavior and match the model's expected 2D output structure.
        if context.is_prefill:
            if q.dim() == 4:
                q = q.flatten(0, 1)
            if k.dim() == 4:
                k = k.flatten(0, 1)
            if v.dim() == 4:
                v = v.flatten(0, 1)

        # 1. Update Cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 2. Compute Attention
        if context.is_prefill:
            if context.block_tables is not None:
                # Prefix caching logic (simplified)
                # In a real implementation, you'd mix k/v from input with k_cache
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
        else:
            o = paged_attention_decode(
                q,
                k_cache,
                v_cache,
                block_tables=context.block_tables,
                context_lens=context.context_lens,
                scale=self.scale,
            )
        return o
