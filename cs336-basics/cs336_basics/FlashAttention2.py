import math

import torch

import triton
import triton.language as tl

from einops import einsum

def tiled_fa_torch(
    Q, 
    K,
    V,
):
    H, seq_len, d = Q.shape
    Q_TILE = 16
    KV_TILE = 16
    O = torch.empty_like(Q)
    L = torch.empty(H, seq_len)
    scale = 1 / math.sqrt(d)

    T_Q = seq_len // Q_TILE
    T_KV = K.shape[1] // KV_TILE

    for h in range(H):
        for i in range(T_Q):
            q_tile = Q[h, i * Q_TILE: (i + 1) * Q_TILE, :]
            o_tile = torch.zeros(Q_TILE, d)
            L_TILE = torch.zeros(Q_TILE, 1)
            l = torch.zeros(Q_TILE, 1)
            m = torch.full((Q_TILE, 1), float("-inf"))
            for j in range(T_KV):
                k_tile = K[h, j * KV_TILE: (j + 1) * KV_TILE, :]
                v_tile = V[h, j * KV_TILE: (j + 1) * KV_TILE, :]

                qk = einsum(q_tile, k_tile, "sq d, sk d -> sq sk")
                qk *= scale
                L_TILE += torch.sum(torch.exp(qk), dim=-1, keepdim=True)
                
                qk_row_max, _ = qk.max(dim=-1, keepdim=True)

                new_m = torch.max(m, qk_row_max)
                P = torch.exp(qk - new_m)
                
                l = torch.exp(m - new_m) * l + P.sum(dim=-1, keepdim=True)
                qkv = einsum(P, v_tile, "sq sk, sk d -> sq d")
                o_tile = torch.exp(m - new_m) * o_tile + qkv
                m = new_m
            o_tile = o_tile / l
            O[h, i * Q_TILE: (i + 1) * Q_TILE, :] = o_tile
            L[h, i * Q_TILE: (i + 1) * Q_TILE] = torch.log(L_TILE.squeeze(1))
    return O, L

class FlashAttention2Torch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        O, L = tiled_fa_torch(
            Q, 
            K, 
            V,
        )
        ctx.save_for_backward(Q, K, V, O, L)
        return O
    
    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, O, L = ctx.saved_tensors
        D = torch.sum(O * grad_out, dim=-1, keepdim=True)

        d = Q.shape[-1]
        scale = 1 / (d ** 0.5)

        qk = einsum(Q, K, "... seq_q d, ... seq_k d -> ... seq_q seq_k") * scale
        P = torch.exp(qk - L.unsqueeze(-1))
        dV = einsum(P, grad_out, "... seq_q seq_k, ... seq_q d -> ... seq_k d")
        dP = einsum(grad_out, V, "... seq_q d, ... seq_k d -> ... seq_q seq_k")
        dS = P * (dP - D)
        dQ = einsum(dS, K, "... seq_q seq_k, ... seq_k d -> ... seq_q d") * scale
        dK = einsum(dS, Q, "... seq_q seq_k, ... seq_q d -> ... seq_k d") * scale
        return dQ, dK, dV, None

@triton.jit
def fa2_fwd(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale, 
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    q_tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_id * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile_id * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    o_block_ptr = tl.make_block_ptr(
        O_ptr + batch_id * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_tile_id * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        K_ptr + batch_id * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    v_block_ptr = tl.make_block_ptr(
        V_ptr + batch_id * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    l_block_ptr = tl.make_block_ptr(
        L_ptr + batch_id * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(q_tile_id * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, )
    )

    o_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE, 1), float("-inf"), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)

    l_log_sum_exp = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    q_tile = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_tile = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_tile = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(q_tile, tl.trans(k_tile))
        qk *= scale
        if is_causal:
            col = tl.arange(0, K_TILE_SIZE)[None, :] + i * K_TILE_SIZE
            row = tl.arange(0, Q_TILE_SIZE)[:, None] + q_tile_id * Q_TILE_SIZE
            mask = row >= col            
            qk = tl.where(mask, qk, -1e6)

        l_log_sum_exp += tl.sum(tl.exp(qk), axis=1)

        qk_row_max = tl.max(qk, axis=1, keep_dims=True)
        
        new_m = tl.maximum(m, qk_row_max)
        
        P = tl.exp(qk - new_m)
        
        l = tl.exp(m - new_m) * l + tl.sum(P, axis=1, keep_dims=True)
        o_tile = tl.exp(m - new_m) * o_tile
        o_tile = tl.dot(P.to(v_tile.dtype), v_tile, acc=o_tile)
        m = new_m

        k_block_ptr = k_block_ptr.advance((K_TILE_SIZE, 0))
        v_block_ptr = v_block_ptr.advance((K_TILE_SIZE, 0))
    
    o_tile = o_tile / l
    l_log_sum_exp = tl.log(l_log_sum_exp)
    o_tile = o_tile.to(tl.float32)
    tl.store(o_block_ptr, o_tile, boundary_check=(0, 1))
    tl.store(l_block_ptr, l_log_sum_exp, boundary_check=(0,))


def tiled_fa_triton(
    Q,
    K, 
    V,
    is_causal,
):
    B, Q_len, d = Q.shape
    _, K_len, _ = K.shape

    O = torch.zeros_like(Q)
    L = torch.empty(B, Q_len, device="cuda")

    Q_TILE = 16
    K_TILE = 16

    B_Q = Q_len // 16

    grid = (B_Q, B)

    fa2_fwd[grid](
        Q, K, V,
        O, L,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        L.stride(0), L.stride(1),
        N_QUERIES=Q_len,
        N_KEYS=K_len,
        scale= 1 / (d ** 0.5),
        D=d,
        Q_TILE_SIZE=Q_TILE,
        K_TILE_SIZE=K_TILE,
        is_causal=is_causal,
    )
    return O, L



class FlashAttention2Triton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        O, L = tiled_fa_triton(
            Q, 
            K, 
            V,
            is_causal
        )
        ctx.save_for_backward(Q, K, V, O, L)
        return O
    
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError



