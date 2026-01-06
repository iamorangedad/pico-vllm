import torch
from torch import nn
from torch.nn import functional as F
from nanovllm.utils.context import get_context


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats the Key/Value heads to match the Query heads.
    Input:  (Batch, Num_KV_Heads, SeqLen, Head_Dim)
    Output: (Batch, Num_Query_Heads, SeqLen, Head_Dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    # Expand 8 heads to 8 groups of 'n_rep' copies
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    # Flatten the groups to get 16 heads
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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


def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
):
    # q shape: (Batch, Num_Heads, Head_Dim)
    if q.dim() == 3:
        q = q.unsqueeze(2)  # (Batch, Num_Heads, 1, Head_Dim)

    q = q.transpose(1, 2)  # (Batch, Num_Heads, 1, Head_Dim)

    batch_size = block_tables.shape[0]

    # 1. GATHER BLOCKS (Fetch data from scattered memory)
    valid_blocks = block_tables.long()
    k_gathered = k_cache[valid_blocks]
    v_gathered = v_cache[valid_blocks]

    # 2. RESHAPE TO SEQUENCE
    k_seq = k_gathered.view(batch_size, -1, k_gathered.shape[-2], k_gathered.shape[-1])
    v_seq = v_gathered.view(batch_size, -1, v_gathered.shape[-2], v_gathered.shape[-1])

    # Transpose: (Batch, KV_Heads, Seq_Len, Head_Dim)
    k_seq = k_seq.transpose(1, 2)
    v_seq = v_seq.transpose(1, 2)

    # --- FIX: HANDLE GQA (Grouped Query Attention) ---
    num_q_heads = q.shape[1]
    num_kv_heads = k_seq.shape[1]
    if num_q_heads != num_kv_heads:
        n_rep = num_q_heads // num_kv_heads
        k_seq = repeat_kv(k_seq, n_rep)
        v_seq = repeat_kv(v_seq, n_rep)
    # -------------------------------------------------

    # 3. MASKING (Remove padding from the gathered blocks)
    max_seq_len = k_seq.shape[2]
    mask = torch.arange(max_seq_len, device=q.device).expand(batch_size, max_seq_len)
    mask = mask < context_lens.unsqueeze(1)
    mask = mask[:, None, None, :]

    # 4. COMPUTE
    o = F.scaled_dot_product_attention(q, k_seq, v_seq, attn_mask=mask, scale=scale)

    return o.transpose(1, 2).squeeze(2)


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
    output = torch.empty_like(q)

    # Loop through each sequence in the batch manually
    for i in range(len(cu_seqlens_q) - 1):
        start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i + 1]

        # Extract the sequence for this batch item
        q_i = q[start_q:end_q].unsqueeze(0).transpose(1, 2)  # (1, Heads, Seq_Q, Dim)
        k_i = k[start_k:end_k].unsqueeze(0).transpose(1, 2)  # (1, KV_Heads, Seq_K, Dim)
        v_i = v[start_k:end_k].unsqueeze(0).transpose(1, 2)

        # --- FIX: HANDLE GQA IN PREFILL ---
        # This is where your error was occurring (16 vs 8)
        num_q_heads = q_i.shape[1]
        num_kv_heads = k_i.shape[1]
        if num_q_heads != num_kv_heads:
            n_rep = num_q_heads // num_kv_heads
            k_i = repeat_kv(k_i, n_rep)
            v_i = repeat_kv(v_i, n_rep)
        # ----------------------------------

        # Causal Mask (Lower Triangular)
        L, S = q_i.size(-2), k_i.size(-2)
        attn_mask = torch.ones(L, S, dtype=torch.bool, device=q.device).tril(diagonal=0)

        o_i = F.scaled_dot_product_attention(
            q_i, k_i, v_i, attn_mask=attn_mask, scale=scale
        )

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

        # 1. Store new tokens into KV Cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 2. Compute Attention
        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache

            # This triggers during warmup
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
            # This triggers during generation
            o = paged_attention_decode(
                q,
                k_cache,
                v_cache,
                block_tables=context.block_tables,
                context_lens=context.context_lens,
                scale=self.scale,
            )
        return o
