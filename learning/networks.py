import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import List, Sequence, Union, Tuple, Optional, Literal
from game.action_space import ALL_ACTION_FUNCTIONS

from game.improvements import Improvements
from game.natural_wonders import ALL_NATURAL_WONDERS
from game.religion import ReligiousTenets
from game.social_policies import SocialPolicies

from game.units import GameUnits
from game.buildings import GameBuildings
from game.resources import ALL_RESOURCES
from game.techs import Technologies, ALL_TECH_COST


ACTIVATIONS = {
    "relu": nn.relu,
    "silu": nn.silu,
    "identity": lambda x: x,
    "sigmoid": nn.sigmoid
}


def make_terra_nova_network(
        # Map/Other Embeddings, map->patches
        embedding_dim=64, 
        embedding_init="normal_0.02", 
        embedding_dtype=jnp.float32,
        patch_embedding_mode="conv",
        patch_dim=128,
        patch_size=(6, 6),
        patch_position_embeddings=False,
        patch_proj_dtype=jnp.float32,
        patch_dtype=jnp.float32,
        norm_scatters=True,
        
        # MapEmbedder hypers
        me_n_map_selfattn=2,
        me_norm_streams_pre=False,
        me_dtype=jnp.float32,
        me_attn_num_heads=4,
        me_rope_query="2d",
        me_rope_key="2d",
        me_rope_base=10_000.0,
        me_attn_dropout=0.0,
        me_mlp_dropout=0.0,
        me_mlp_ratio=4,
        me_use_bias_qkv=False,
        me_use_bias_out=False,
        me_n_map_crossattn=1,
        me_n_pma_seeds=4,
        me_pma_dropout=0.0,

        # GamestateEmbedder
        ge_num_seeds_tech=4,
        ge_num_heads_tech=4,
        ge_num_seeds_pols=4,
        ge_num_heads_pols=4,
        ge_num_heads_trade_offer=4,
        ge_mlp_ratio_trade_offer=4,
        ge_pma_num_heads_trade_offer=4,
        ge_pma_num_seeds_trade_offer=1,
        ge_num_heads_trade_ledger=4,
        ge_mlp_ratio_trade_ledger=4,
        ge_pma_num_heads_trade_ledger=4,
        ge_pma_num_seeds_trade_ledger=2,
        ge_num_heads_trade_length=4,
        ge_mlp_ratio_trade_length=4,
        ge_pma_num_heads_trade_length=4,
        ge_pma_num_seeds_trade_length=2,
        ge_num_heads_res_adj=4,
        ge_mlp_ratio_res_adj=4,
        ge_pma_num_heads_res_adj=4,
        ge_pma_num_seeds_res_adj=4,
        ge_num_heads_trade_summary=4,
        ge_pma_num_heads_trade_summary=4,
        ge_pma_num_seeds_trade_summary=4,
        ge_num_heads_tenets_inner=4,
        ge_mlp_ratio_tenets_inner=4,
        ge_pma_num_heads_tenets_inner=4,
        ge_pma_num_seeds_tenets_inner=2, 
        ge_num_heads_tenets_players=4,
        ge_mlp_ratio_tenets_players=4,
        ge_pma_num_heads_tenets_players=4,
        ge_pma_num_seeds_tenets_players=2,
        ge_num_heads_cs=4,
        ge_mlp_ratio_cs=4,
        ge_pma_num_heads_cs=4,
        ge_pma_num_seeds_cs=2,
        ge_num_heads_dels=4,
        ge_mlp_ratio_dels=4,
        ge_pma_num_heads_dels=4,
        ge_pma_num_seeds_dels=2,
        ge_num_heads_gws=4,
        ge_mlp_ratio_gws=4,
        ge_pma_num_heads_gws=4,
        ge_pma_num_seeds_gws=2,
        ge_num_heads_gpps=4,
        ge_mlp_ratio_gpps=4,
        ge_pma_num_heads_gpps=4,
        ge_pma_num_seeds_gpps=2,
        ge_num_heads_have_met=4,
        ge_mlp_ratio_have_met=4,
        ge_pma_num_heads_have_met=4,
        ge_pma_num_seeds_have_met=2,
        ge_num_heads_tourism_inner=4,
        ge_mlp_ratio_tourism_inner=4,
        ge_pma_num_heads_tourism_inner=4,
        ge_pma_num_seeds_tourism_inner=2,
        ge_num_heads_tourism_players=4,
        ge_mlp_ratio_tourism_players=4,
        ge_pma_num_heads_tourism_players=4,
        ge_pma_num_seeds_tourism_players=2,
        ge_n_general_selfattn=3,
        ge_general_attention_bias=None,
        ge_general_num_heads=4,
        ge_general_mlp_ratio=4,
        ge_param_dtype=jnp.float32,
        ge_n_aggregate_selfattn=3,
        ge_general_pma_num_heads=4,
        ge_general_pma_num_seeds=4,
        ge_aggregate_attention_bias=None,
        ge_aggregate_num_heads=4,
        ge_aggregate_mlp_ratio=4,
        ge_aggregate_pma_num_heads=4,
        ge_aggregate_pma_num_seeds=4,
        
        # UnitEncoder
        ue_units_inner_num_heads=4,
        ue_units_inner_mlp_ratio=4,
        ue_units_inner_pma_num_heads=4,
        ue_units_inner_pma_num_seeds=4,
        ue_units_players_num_heads=4,
        ue_units_players_mlp_ratio=4,
        ue_units_players_pma_num_heads=4,
        ue_units_players_pma_num_seeds=4,
        ue_trade_yield_num_heads=4,
        ue_trade_yield_mlp_ratio=4,
        ue_trade_yield_pma_num_heads=4,
        ue_trade_yield_pma_num_seeds=2,
        ue_trade_units_num_heads=4,
        ue_trade_units_mlp_ratio=4,
        ue_trade_units_pma_num_heads=4,
        ue_trade_units_pma_num_seeds=2,
        ue_units_summary_num_heads=4,
        ue_units_summary_mlp_ratio=4,
        ue_units_summary_pma_num_heads=4,
        ue_units_summary_pma_num_seeds=4,
        ue_n_units_summary_selfattn=3,

        # CityStateEncoder
        cse_rel_num_heads=4,
        cse_rel_mlp_ratio=4,
        cse_rel_pma_num_heads=4,
        cse_rel_pma_num_seeds=4,
        cse_cs_num_heads=4,
        cse_cs_mlp_ratio=4,
        cse_cs_pma_num_heads=4,
        cse_cs_pma_num_seeds=4,
        cse_global_num_heads=4,
        cse_global_mlp_ratio=4,
        cse_global_pma_num_heads=4,
        cse_global_pma_num_seeds=4,
        cse_fuse_num_heads=4,
        cse_fuse_mlp_ratio=4,
        cse_fuse_pma_num_heads=4,
        cse_fuse_pma_num_seeds=4,
        cse_n_fuse_selfattn=3,

        # CityEncoder
        ce_pooling_num_heads=4,
        ce_pooling_mlp_ratio=4,
        ce_pooling_pma_num_seeds=1,
        ce_fuse_num_heads=4,
        ce_fuse_mlp_ratio=4,
        ce_fuse_pma_num_heads=4,
        ce_fuse_pma_num_seeds=3,
        ce_relation_num_heads=4,
        ce_relation_mlp_ratio=4,
        ce_relation_pma_num_heads=4,
        ce_relation_pma_num_seeds=1,
        ce_n_fuse_selfattn=3,
        ce_fuse_pma_num_seeds_final=4,

        # Value function
        V_use_stream_gates=True,
        V_n_selfattn=3,
        V_num_heads=4,
        V_mlp_ratio=4,
        V_param_dtype=jnp.float32,
        V_activation="gelu",
        V_use_bias_qkv=False,
        V_use_bias_out=False,
        V_attention_bias=None,
        V_pma_num_heads=4,
        V_pma_num_seeds=2,
        V_head_hidden_mult=128,
        
        # Action Head Trade Deals
        Atd_use_stream_gates=True,
        Atd_n_selfattn=3,
        Atd_num_heads=4,
        Atd_mlp_ratio=4,
        Atd_param_dtype=jnp.float32,
        Atd_activation="gelu",
        Atd_use_bias_qkv=False,
        Atd_use_bias_out=False,
        Atd_attention_bias=None,
        Atd_pma_num_heads=4,
        Atd_pma_num_seeds=10,  # global info bottleneck
        Atd_num_heads_trade_offer=4,
        Atd_mlp_ratio_trade_offer=4,
        Atd_num_heads_trade_ledger=4,
        Atd_mlp_ratio_trade_ledger=4,
        Atd_pma_num_heads_trade_ledger=4,
        Atd_pma_num_seeds_trade_ledger=6,
        Atd_num_heads_res_adj=4,
        Atd_mlp_ratio_res_adj=4,
        Atd_pma_num_heads_res_adj=4,
        Atd_pma_num_seeds_res_adj=6,
        Atd_num_heads_trade_summary=4,
        Atd_mlp_ratio_trade_summary=4,
        Atd_pma_num_heads_trade_summary=4,
        Atd_pma_num_seeds_trade_summary=6,
        
        # Action Head Social Policies
        Asp_use_stream_gates=True,
        Asp_n_selfattn=3,
        Asp_num_heads=4,
        Asp_mlp_ratio=4,
        Asp_pma_num_heads=4,
        Asp_pma_num_seeds=10,  # global info bottleneck
        Asp_param_dtype=jnp.float32,
        Asp_activation="gelu",
        Asp_use_bias_qkv=False,
        Asp_use_bias_out=False,
        Asp_attention_bias=None,
        Asp_num_heads_pols=4,
        Asp_n_general_selfattn=3,
        Asp_general_num_heads=4,
        Asp_general_mlp_ratio=4,
        
        # Action Head Religion
        Ar_use_stream_gates=True,
        Ar_n_selfattn=3,
        Ar_num_heads=4,
        Ar_mlp_ratio=4,
        Ar_param_dtype=jnp.float32,
        Ar_activation="gelu",
        Ar_use_bias_qkv=False,
        Ar_use_bias_out=False,
        Ar_attention_bias=None,
        Ar_pma_num_heads=4,
        Ar_pma_num_seeds=10,  # global info bottleneck
        Ar_pma_num_heads_tenets_inner=4,
        Ar_pma_num_seeds_tenets_inner=4,
        Ar_num_heads_tenets_inner=4,
        Ar_mlp_ratio_tenets_inner=4,
        Ar_general_num_heads=4,
        Ar_general_mlp_ratio=4,
        Ar_n_general_selfattn=3,
        
        # Action Head Technology
        At_use_stream_gates=True,
        At_n_selfattn=3,
        At_num_heads=4,
        At_mlp_ratio=4,
        At_param_dtype=jnp.float32,
        At_activation="gelu",
        At_use_bias_qkv=False,
        At_use_bias_out=False,
        At_attention_bias=None,
        At_pma_num_heads=4,
        At_pma_num_seeds=10,  # global info bottleneck
        At_num_heads_tech=4,
        At_general_num_heads=4,
        At_general_mlp_ratio=4,
        At_n_general_selfattn=3,
        
        # Action Head Units
        Au_use_stream_gates=True,
        Au_n_selfattn=3,
        Au_num_heads=4,
        Au_mlp_ratio=4,
        Au_param_dtype=jnp.float32,
        Au_activation="gelu",
        Au_use_bias_qkv=False,
        Au_use_bias_out=False,
        Au_attention_bias=None,
        Au_pma_num_heads=4,
        Au_pma_num_seeds=10,  # global info bottleneck
        Au_units_inner_num_heads=4,
        Au_units_inner_mlp_ratio=4,
        Au_units_players_num_heads=4,
        Au_units_players_mlp_ratio=4,
        Au_trade_yields_num_heads=4,
        Au_trade_yields_mlp_ratio=4,
        Au_trade_units_num_heads=4,
        Au_trade_units_mlp_ratio=4,
        Au_my_units_num_heads=4,
        Au_my_units_mlp_ratio=4,
        Au_n_my_units_selfattn=3,
        Au_trade_yield_pma_num_heads=4,
        Au_trade_yield_pma_num_seeds=3,
        
        # Action Head Cities
        Ac_use_stream_gates=True,
        Ac_n_selfattn=3,
        Ac_num_heads=4,
        Ac_mlp_ratio=4,
        Ac_param_dtype=jnp.float32,
        Ac_activation="gelu",
        Ac_use_bias_qkv=False,
        Ac_use_bias_out=False,
        Ac_attention_bias=None,
        Ac_pma_num_heads=4,
        Ac_pma_num_seeds=10,  # global info bottleneck
        Ac_pooling_num_heads=4,
        Ac_pooling_pma_seeds=4,
        Ac_pooling_mlp_ratio=4,
        Ac_fuse_num_heads=4,
        Ac_fuse_mlp_ratio=4,
        Ac_fuse_pma_num_heads=4,
        Ac_fuse_pma_num_seeds=2
    ):
    """
    AlphaStar unplugged (https://arxiv.org/pdf/2308.03526) does a "scatter" op, where they insert
    the learned unit-type embedding into the map
    """
    def fuse_dir6_into_channels(x: jnp.ndarray, *, expect_size: int = 6) -> jnp.ndarray:
        # expects (..., H, W, A, C) with the extra axis A at -2
        if x.ndim < 5:
            return x  # already (..., H, W, C) or smaller; no-op
        if expect_size is not None:
            assert x.shape[-2] == expect_size, f"expected extra axis {expect_size}, got {x.shape[-2]}"
        return x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
    
    def gate_cont(x, known_mask, scale, feature_dim, name):
        x_norm = (x / scale).astype(jnp.float32)
        x_norm = x_norm * known_mask.astype(jnp.float32)
        feat = nn.Dense(feature_dim, name=f"{name}_proj", use_bias=False)(x_norm[..., None])  # (B,6,Cmax,E)
        # learned bias for “known/unknown”
        k = nn.Embed(2, feature_dim, name=f"{name}_known_emb")(known_mask.astype(jnp.int32))  # 0/1→E
        return feat + k

    def scatter_cities(mask, tok, normalize=True):
        """
        mask: (B, 6, 10, H, W)  nonnegative weights (e.g., 0/1)
        tok : (B, 6, 10, D)
        returns: (B, H, W, D)
        """
        w = mask.astype(jnp.float32)
        agg = jnp.einsum('bpchw,bpcd->bhwd', w, tok)

        if not normalize:
            return agg

        den = w.sum(axis=(1, 2))  # (B,H,W)

        # add 1 where den==0 to avoid divide-by-zero
        # multiply output by (den>0) to zero empty tiles
        den_safe = den + (den == 0).astype(den.dtype)         # (B,H,W)
        out = agg * (1.0 / den_safe[..., None])               # (B,H,W,D)
        out = out * (den > 0)[..., None]                      # zero tiles with no contributors
        return out

    def scatter_units(
        unit_tokens,
        unit_rowcol,
        unit_exists_mask,
        map_hw,
        mean_normalize=True,
    ):
        """
        unit_tokens:  # (B, 6, U, z_dim)
        unit_rowcol:  # (B, 6, U, 2) int32
        unit_exists_mask:  # (B, 6, U) bool (e.g., type != -1 and pos known)
        map_hw:  # (H, W)

        Here, unit_exists_mask should also 0 out when our own unit does exist, as we cannot
        embed a non-existant unit into the map. So should be like unit_type > 0

        """
        B = unit_tokens.shape[0]
        H, W = map_hw

        # masks & indices
        valid = unit_exists_mask
        row = jnp.clip(unit_rowcol[..., 0], 0, H - 1)
        col = jnp.clip(unit_rowcol[..., 1], 0, W - 1)

        # flatten everything except batch for a single scatter per batch
        P, U = unit_tokens.shape[1], unit_tokens.shape[2]
        z_dim = unit_tokens.shape[-1]

        # build per-unit batch indices for .at[...] scatter
        batch_idx = jnp.arange(B, dtype=jnp.int32)[:, None, None]
        batch_idx = jnp.broadcast_to(batch_idx, (B, P, U))

        # filter out invalid units by zeroing their contribution
        values = unit_tokens * valid[..., None].astype(unit_tokens.dtype)  # (B,6,U,z_dim)

        # allocate outputs
        tiles = jnp.zeros((B, H, W, z_dim), dtype=unit_tokens.dtype)

        # scatter-add values and counts
        tiles = tiles.at[batch_idx, row, col].add(values)

        if not mean_normalize:
            return tiles

        # arithmetic-safe mean normalization per tile
        counts = jnp.zeros((B, H, W), dtype=jnp.int32)
        counts = counts.at[batch_idx, row, col].add(valid.astype(jnp.int32))
        counts_f = counts.astype(tiles.dtype)
        counts_safe = counts_f + (counts_f == 0).astype(tiles.dtype)
        tiles = tiles * (1.0 / counts_safe[..., None])
        tiles = tiles * (counts_f > 0)[..., None]
        return tiles


    def encode_single_value(x, *, denom=None, clip_hi=4.0, dim=64, name="scalar"):
        # x: (B,) or (B,1). No missingness.
        x = x.astype(jnp.float32)
        if denom is None:
            # preserve sign instead of zero-clipping negatives
            r = jnp.sign(x) * jnp.log1p(jnp.abs(x))
        else:
            r = jnp.clip(x / jnp.maximum(denom, 1e-6), -clip_hi, clip_hi)

        # tiny basis (poly + magnitude) → Dense
        feats = jnp.stack([r, r**2, jnp.log1p(jnp.abs(x))], axis=-1)
        return nn.Dense(dim, name=f"{name}_proj", use_bias=False)(feats)


    def encode_vector_values(x, *, denom=None, clip_hi=4.0, dim=64, name="vector"):
        # x: (..., K)
        x = x.astype(jnp.float32)

        if denom is None:
            r = jnp.sign(x) * jnp.log1p(jnp.abs(x))
        else:
            denom_b = jnp.broadcast_to(jnp.maximum(denom, 1e-6), x.shape)
            r = jnp.clip(x / denom_b, -clip_hi, clip_hi)

        feats = jnp.stack([r, r**2, jnp.log1p(jnp.abs(x))], axis=-1)  # (..., K, 3)
        return nn.Dense(dim, name=f"{name}_proj", use_bias=False)(feats)

    INITIALIZER_LOOKUP = {
        "normal_0.02": nn.initializers.normal(0.02),
        "xavier_uniform": nn.initializers.xavier_uniform(),
        "xavier_normal": nn.initializers.xavier_normal(),
        "kaiming_uniform": nn.initializers.variance_scaling(1.0, "fan_in", "uniform"),
        "kaiming_normal": nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"),
    }

    class PatchEmbed(nn.Module):
        mode: Literal["conv", "dense", "none"] = "conv"
        dim: Optional[int] = 128  # ignored in mode="none" if add_pos=False
        patch: Tuple[int, int] = (6, 6)
        add_pos: bool = False
        param_dtype: any = jnp.bfloat16
        out_dtype: any = jnp.float32

        @nn.compact
        def __call__(self, x: jnp.ndarray):
            # x: (..., H, W, C)
            ph, pw = self.patch
            *lead, H, W, C = x.shape
            assert H % ph == 0 and W % pw == 0, "patch must divide H and W"
            nh, nw = H // ph, W // pw

            if self.mode == "conv":
                # collapse leading dims to one big batch for a single conv
                x_b = x.reshape((-1, H, W, C))
                y = nn.Conv(
                    features=int(self.dim),
                    kernel_size=(ph, pw),
                    strides=(ph, pw),
                    padding="VALID",
                    use_bias=True,
                    dtype=self.out_dtype,
                    param_dtype=self.param_dtype,
                    name="patch_proj",
                )(x_b)
                y = y.reshape((*lead, nh * nw, int(self.dim)))
                tokens = y

            elif self.mode == "dense":
                # (..., nh, ph, nw, pw, C) -> (..., nh, nw, ph, pw, C) -> (..., N, ph*pw*C)
                y = x.reshape(*lead, nh, ph, nw, pw, C)
                perm = (*range(len(lead)), len(lead), len(lead) + 2, len(lead) + 1, len(lead) + 3, len(lead) + 4)
                y = y.transpose(perm).reshape(*lead, nh * nw, ph * pw * C)
                tokens = nn.Dense(
                    int(self.dim),
                    dtype=self.out_dtype,
                    param_dtype=self.param_dtype,
                    name="patch_proj",
                )(y)

            elif self.mode == "none":
                y = x.reshape(*lead, nh, ph, nw, pw, C)
                perm = (*range(len(lead)), len(lead), len(lead) + 2, len(lead) + 1, len(lead) + 3, len(lead) + 4)
                tokens = y.transpose(perm).reshape(*lead, nh * nw, ph * pw * C)
                # dim is ph*pw*C here; no learned projection
                if self.add_pos:
                    pos = self.param(
                        "pos_embed",
                        nn.initializers.normal(0.02),
                        (1,) * (tokens.ndim - 2) + (tokens.shape[-2], tokens.shape[-1]),
                    )
                    tokens = tokens + pos
                return tokens

            else:
                raise ValueError(f"unknown mode: {self.mode}")

            if self.add_pos:
                pos = self.param(
                    "pos_embed",
                    nn.initializers.normal(0.02),
                    (1,) * (tokens.ndim - 2) + (tokens.shape[-2], tokens.shape[-1]),
                )
                tokens = tokens + pos
            return tokens

    def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
        left, right = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-right, left], axis=-1)

    def rope_factors_1d(seq_len: int, head_dim: int, base: float = 10000.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        half = head_dim // 2
        inv_freq = 1.0 / (base ** (jnp.arange(0, half, dtype=jnp.float32) / half))
        positions = jnp.arange(seq_len, dtype=jnp.float32)
        angles = positions[:, None] * inv_freq[None, :]                    # (seq_len, half)
        angles = jnp.concatenate([angles, angles], axis=-1)                # (seq_len, head_dim)
        cos = jnp.cos(angles)[None, None, :, :]                            # (1,1,seq_len,head_dim)
        sin = jnp.sin(angles)[None, None, :, :]
        return cos, sin

    def rope_factors_2d(rows: int, cols: int, head_dim: int, base: float = 10000.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        assert head_dim % 2 == 0, "head_dim must be even for 2D RoPE"
        dim_row = head_dim // 2
        dim_col = head_dim - dim_row
        cos_r, sin_r = rope_factors_1d(rows, dim_row, base)                # (1,1,rows,dim_row)
        cos_c, sin_c = rope_factors_1d(cols, dim_col, base)                # (1,1,cols,dim_col)
        cos_r = jnp.repeat(cos_r, cols, axis=2).reshape(1, 1, rows * cols, dim_row)
        sin_r = jnp.repeat(sin_r, cols, axis=2).reshape(1, 1, rows * cols, dim_row)
        cos_c = jnp.tile(cos_c, (1, 1, rows, 1)).reshape(1, 1, rows * cols, dim_col)
        sin_c = jnp.tile(sin_c, (1, 1, rows, 1)).reshape(1, 1, rows * cols, dim_col)
        cos = jnp.concatenate([cos_r, cos_c], axis=-1)                     # (1,1,N,head_dim)
        sin = jnp.concatenate([sin_r, sin_c], axis=-1)
        return cos, sin

    def apply_rope_to_qk(q_heads: jnp.ndarray, k_heads: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q_rot = q_heads * cos + rotate_half(q_heads) * sin
        k_rot = k_heads * cos + rotate_half(k_heads) * sin
        return q_rot, k_rot

    def film_modulate(
        array,
        modulate_variable,
        *,
        name: str = "film",
        param_dtype=jnp.bfloat16,
        out_dtype=jnp.float32,
    ):
        """
        Feature-wise linear modulation (FiLM).

        Args:
            array: Tensor to modulate, shape (..., features).
            modulate_variable: Conditioning tensor with same leading dims as `array`,
                arbitrary last-dim; will be projected to 2*features.
            name: Scope name prefix for parameters.
            param_dtype: Parameter dtype (e.g., jnp.bfloat16).
            out_dtype: Compute/output dtype (e.g., jnp.float32).

        Returns:
            Modulated tensor with same shape as `array`.
        """
        feature_dim = array.shape[-1]

        proj = nn.Dense(
            features=2 * feature_dim,
            use_bias=True,
            dtype=out_dtype,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.zeros,   # start as identity: gamma=1, beta=0
            bias_init=nn.initializers.zeros,
            name=f"{name}_proj",
        )(modulate_variable)

        gamma_delta, beta = jnp.split(proj, 2, axis=-1)
        gamma = gamma_delta + 1.0

        return gamma * array + beta


    # =========================
    # General Multi-Head Attention (self or cross)
    # =========================

    class MultiHeadAttention(nn.Module):
        # Use RoPE for tokens with real geometry or order (map=2d, time=1d).
        # Leave RoPE off for sets so SAB/PMA remain permutation-equivariant/invariant.
        # In cross-attn, it's fine for queries to have RoPE and keys to have none.

        hidden_dim: int
        num_heads: int
        rope_query: Literal["none", "1d", "2d"] = "none"
        rope_key: Literal["none", "1d", "2d"] = "none"
        rope_base: float = 10000.0
        query_grid_hw: Optional[Tuple[int, int]] = None
        key_grid_hw: Optional[Tuple[int, int]] = None
        dropout_rate: float = 0.0
        param_dtype: any = jnp.bfloat16
        out_dtype: any = jnp.float32
        use_bias_qkv: bool = False
        use_bias_out: bool = False

        @nn.compact
        def __call__(
            self,
            query_tokens: jnp.ndarray,  # (batch, nq, hidden_dim)
            key_value_tokens: jnp.ndarray,  # (batch, nk, hidden_dim)
            *,
            attention_bias: Optional[jnp.ndarray] = None,  # broadcastable to (batch, heads, nq, nk)
            deterministic: bool = True,
            return_attention: bool = False,
        ):
            batch_size, num_query, model_dim_q = query_tokens.shape
            _, num_key, model_dim_k = key_value_tokens.shape
            assert model_dim_q == model_dim_k == self.hidden_dim, "hidden_dim mismatch"
            assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
            head_dim = self.hidden_dim // self.num_heads

            q_proj = nn.Dense(self.hidden_dim, use_bias=self.use_bias_qkv, dtype=self.out_dtype, param_dtype=self.param_dtype, name="q_proj")
            k_proj = nn.Dense(self.hidden_dim, use_bias=self.use_bias_qkv, dtype=self.out_dtype, param_dtype=self.param_dtype, name="k_proj")
            v_proj = nn.Dense(self.hidden_dim, use_bias=self.use_bias_qkv, dtype=self.out_dtype, param_dtype=self.param_dtype, name="v_proj")

            query = q_proj(query_tokens)
            key = k_proj(key_value_tokens)
            value = v_proj(key_value_tokens)

            def split_heads(x: jnp.ndarray, seq_len: int) -> jnp.ndarray:
                return x.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)

            query_heads = split_heads(query, num_query)  # (B,H,Nq,Dh)
            key_heads = split_heads(key, num_key)  # (B,H,Nk,Dh)
            value_heads = split_heads(value, num_key)  # (B,H,Nk,Dh)

            # Optional RoPE per side
            if self.rope_query != "none":
                if self.rope_query == "2d":
                    assert self.query_grid_hw is not None and self.query_grid_hw[0] * self.query_grid_hw[1] == num_query
                    cos_q, sin_q = rope_factors_2d(self.query_grid_hw[0], self.query_grid_hw[1], head_dim, self.rope_base)
                else:
                    cos_q, sin_q = rope_factors_1d(num_query, head_dim, self.rope_base)
                query_heads, _ = apply_rope_to_qk(query_heads, query_heads, cos_q, sin_q)

            if self.rope_key != "none":
                if self.rope_key == "2d":
                    assert self.key_grid_hw is not None and self.key_grid_hw[0] * self.key_grid_hw[1] == num_key
                    cos_k, sin_k = rope_factors_2d(self.key_grid_hw[0], self.key_grid_hw[1], head_dim, self.rope_base)
                else:
                    cos_k, sin_k = rope_factors_1d(num_key, head_dim, self.rope_base)
                key_heads, _ = apply_rope_to_qk(key_heads, key_heads, cos_k, sin_k)

            scale = 1.0 / jnp.sqrt(head_dim)
            attention_logits = jnp.einsum("bhqd,bhkd->bhqk", query_heads, key_heads) * scale
            if attention_bias is not None:
                attention_logits = attention_logits + attention_bias

            attention_weights = jax.nn.softmax(attention_logits, axis=-1)
            attention_weights = nn.Dropout(rate=self.dropout_rate)(attention_weights, deterministic=deterministic)

            attended_heads = jnp.einsum("bhqk,bhkd->bhqd", attention_weights, value_heads)   # (B,H,Nq,Dh)
            attended = attended_heads.transpose(0, 2, 1, 3).reshape(batch_size, num_query, self.hidden_dim)

            output = nn.Dense(self.hidden_dim, use_bias=self.use_bias_out, dtype=self.out_dtype, param_dtype=self.param_dtype, name="o_proj")(attended)

            if return_attention:
                return output, attention_weights
            return output


    # =========================
    # Transformer "Cross" Layer (works for self-attn or cross-attn)
    # =========================
    class TransformerCrossLayer(nn.Module):
        hidden_dim: int
        num_heads: int
        rope_query: Literal["none", "1d", "2d"] = "none"
        rope_key: Literal["none", "1d", "2d"] = "none"
        rope_base: float = 10000.0
        query_grid_hw: Optional[Tuple[int, int]] = None
        key_grid_hw: Optional[Tuple[int, int]] = None
        attn_dropout: float = 0.0
        mlp_dropout: float = 0.0
        mlp_ratio: float = 4.0
        param_dtype: any = jnp.bfloat16
        out_dtype: any = jnp.float32
        activation: Literal["gelu", "silu"] = "gelu"
        use_bias_qkv: bool = False
        use_bias_out: bool = False

        @nn.compact
        def __call__(
            self,
            query_tokens: jnp.ndarray,           # (batch, nq, hidden_dim)
            key_value_tokens: jnp.ndarray,       # (batch, nk, hidden_dim)
            *,
            attention_bias: Optional[jnp.ndarray] = None,
            deterministic: bool = True,
        ) -> jnp.ndarray:
            # pre-norm on each side
            query_norm = nn.LayerNorm(epsilon=1e-6, name="ln_query")(query_tokens)
            key_value_norm = nn.LayerNorm(epsilon=1e-6, name="ln_keyvalue")(key_value_tokens)

            attention_out = MultiHeadAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                rope_query=self.rope_query,
                rope_key=self.rope_key,
                rope_base=self.rope_base,
                query_grid_hw=self.query_grid_hw,
                key_grid_hw=self.key_grid_hw,
                dropout_rate=self.attn_dropout,
                param_dtype=self.param_dtype,
                out_dtype=self.out_dtype,
                use_bias_qkv=self.use_bias_qkv,
                use_bias_out=self.use_bias_out,
                name="mha",
            )(query_norm, key_value_norm, attention_bias=attention_bias, deterministic=deterministic)
            tokens_after_attn = query_tokens + attention_out

            # feed-forward block
            inner_dim = int(self.hidden_dim * self.mlp_ratio)
            act_fn = nn.gelu if self.activation == "gelu" else nn.silu

            ff_norm = nn.LayerNorm(epsilon=1e-6, name="ln_mlp")(tokens_after_attn)
            ff_out = nn.Dense(inner_dim, use_bias=True, dtype=self.out_dtype, param_dtype=self.param_dtype, name="ffn_fc1")(ff_norm)
            ff_out = act_fn(ff_out)
            ff_out = nn.Dropout(rate=self.mlp_dropout)(ff_out, deterministic=deterministic)
            ff_out = nn.Dense(self.hidden_dim, use_bias=True, dtype=self.out_dtype, param_dtype=self.param_dtype, name="ffn_fc2")(ff_out)
            tokens_after_ffn = tokens_after_attn + ff_out

            return tokens_after_ffn


    # =========================
    # PMA: Pooling by Multi-head Attention (Set Transformer style)
    # =========================
    class PMA(nn.Module):
        hidden_dim: int
        num_heads: int
        num_seeds: int = 4
        dropout_rate: float = 0.0
        param_dtype: any = jnp.bfloat16
        out_dtype: any = jnp.float32

        @nn.compact
        def __call__(self, input_tokens: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
            batch_size = input_tokens.shape[0]
            seeds = self.param("seeds", nn.initializers.normal(0.02), (self.num_seeds, self.hidden_dim))
            seed_tokens = jnp.broadcast_to(seeds[None, :, :], (batch_size, self.num_seeds, self.hidden_dim))

            seed_norm = nn.LayerNorm(epsilon=1e-6, name="ln_seeds")(seed_tokens)
            input_norm = nn.LayerNorm(epsilon=1e-6, name="ln_inputs")(input_tokens)

            pooled = MultiHeadAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                rope_query="none",
                rope_key="none",
                dropout_rate=self.dropout_rate,
                param_dtype=self.param_dtype,
                out_dtype=self.out_dtype,
                name="pma_mha",
            )(seed_norm, input_norm, deterministic=deterministic)
            return pooled



    class MapEmbedder(nn.Module):
        n_map_selfattn: int = 2
        attn_num_heads: int = 4
        rope_query: Literal["none", "1d", "2d"] = "none"
        rope_key: Literal["none", "1d", "2d"] = "none"
        rope_base: float = 10000.0
        attn_dropout: float = 0.0
        mlp_dropout: float = 0.0
        mlp_ratio: float = 4.0
        param_dtype: any = jnp.bfloat16
        activation: Literal["gelu", "silu"] = "gelu"
        use_bias_qkv: bool = False
        use_bias_out: bool = False
        norm_streams_pre: bool = False
        n_map_crossattn: int = 1
        n_pma_seeds: int = 4
        pma_dropout: float = 0.0
        out_dim: int = 64

        @nn.compact
        def __call__(
            self,
            # Player-agnostic, map-only information
            z_elevation,
            z_terrain,
            z_features,
            z_nw,
            z_resource,
            z_improvement,
            z_road,
            z_edge_river,
            z_cs_ownership,
            z_cs_centers_map,
            z_yield_map,
            z_improvement_yield_map,
            z_culture_yield_map,
            z_city_yield_map,
            z_religion_yield_map,

            # Visibility map
            z_visibility_map,

            # Player-specific information
            z_cities,
            z_units,
            
            # General hypers
            twod_rope_nhnw,
            training,
            attention_bias=None,
        ):
            z_dim = z_elevation.shape[-1]

            # First fusing the map-related information into one sequence embedding
            # This is one learned embedding per map-variable type: (1, 14, z_dim)
            stream_embedding = nn.Embed(num_embeddings=14, features=z_dim, 
                                        name="stream_embedding", dtype=z_elevation.dtype, 
                                        param_dtype=self.param_dtype)(jnp.arange(14, dtype=jnp.int32))

            # Mainly because there is more than one way to reshape an array, we explcitly 
            # reshape with an iter here... This may be slightly slower than a 
            # .reshape(1, 1, -1), but I trust it more
            # (15 * z_dim)
            stream_embedding = jnp.concatenate([stream_embedding[i] for i in range(15)], axis=-1)
            
            if self.norm_streams_pre:
                z_elevation = nn.LayerNorm(epsilon=1e-6, name="elevation_ln")(z_elevation)
                z_terrain = nn.LayerNorm(epsilon=1e-6, name="terrain_ln")(z_terrain)
                z_features = nn.LayerNorm(epsilon=1e-6, name="feature_ln")(z_features)
                z_nw = nn.LayerNorm(epsilon=1e-6, name="nw_ln")(z_nw)
                z_resource = nn.LayerNorm(epsilon=1e-6, name="resource_ln")(z_resource)
                z_improvement = nn.LayerNorm(epsilon=1e-6, name="improvement_ln")(z_improvement)
                z_road = nn.LayerNorm(epsilon=1e-6, name="road_ln")(z_road)
                z_edge_river = nn.LayerNorm(epsilon=1e-6, name="edgeriver_ln")(z_edge_river)
                z_cs_ownership = nn.LayerNorm(epsilon=1e-6, name="csownership_ln")(z_cs_ownership)
                z_cs_centers_map = nn.LayerNorm(epsilon=1e-6, name="csownership_ln")(z_cs_centers_map)
                z_yield_map = nn.LayerNorm(epsilon=1e-6, name="yield_ln")(z_yield_map)
                z_improvement_yield_map = nn.LayerNorm(epsilon=1e-6, name="improvementyield_ln")(z_improvement_yield_map)
                z_culture_yield_map = nn.LayerNorm(epsilon=1e-6, name="cultureyield_ln")(z_culture_yield_map)
                z_city_yield_map = nn.LayerNorm(epsilon=1e-6, name="cityyield_ln")(z_city_yield_map)
                z_religion_yield_map = nn.LayerNorm(epsilon=1e-6, name="religionyield_ln")(z_religion_yield_map)
            
            # (B, 77, 14 * z_dim) + (1, 1, 14 * z_dim)
            z_fused = jnp.concatenate([
                z_elevation,
                z_terrain,
                z_features,
                z_nw,
                z_resource,
                z_improvement,
                z_road,
                z_edge_river,
                z_cs_ownership,
                z_cs_centers_map,
                z_yield_map,
                z_improvement_yield_map,
                z_culture_yield_map,
                z_city_yield_map,
                z_religion_yield_map,
            ], axis=-1)
            z_fused = z_fused + stream_embedding[None, None]

            # Small linear mixer + FiLM style mod using the visibility map 
            z_fused = nn.Dense(z_dim, dtype=z_fused.dtype, param_dtype=self.param_dtype)(z_fused)
            z_fused = film_modulate(
                z_fused, z_visibility_map, name="map_fused_film", 
                param_dtype=self.param_dtype, out_dtype=z_fused.dtype
            )

            # Now the N loops of self-attention over the game map
            for _ in range(self.n_map_selfattn):
                z_fused = TransformerCrossLayer(
                    hidden_dim=z_dim,
                    num_heads=self.attn_num_heads,
                    rope_query=self.rope_query,
                    rope_key=self.rope_key,
                    rope_base=self.rope_base,
                    query_grid_hw=twod_rope_nhnw,
                    key_grid_hw=twod_rope_nhnw,
                    attn_dropout=self.attn_dropout,
                    mlp_dropout=self.mlp_dropout,
                    mlp_ratio=self.mlp_ratio,
                    param_dtype=self.param_dtype,
                    out_dtype=z_fused.dtype,
                    activation=self.activation,
                    use_bias_qkv=self.use_bias_qkv,
                    use_bias_out=self.use_bias_out,
                )(
                    query_tokens=z_fused, key_value_tokens=z_fused,
                    attention_bias=attention_bias,
                    deterministic=not training,
                )

            # Cross attention time.
            # map <-> cities
            # map <-> units
            z_cities = film_modulate(
                z_cities, z_visibility_map, name="map_cities_film",
                param_dtype=self.param_dtype, out_dtype=z_cities.dtype
            )
            z_units = film_modulate(
                z_units, z_visibility_map, name="map_units_film",
                param_dtype=self.param_dtype, out_dtype=z_units.dtype
            )

            for _ in range(self.n_map_crossattn):
                z_fused = TransformerCrossLayer(
                    hidden_dim=z_dim,
                    num_heads=self.attn_num_heads,
                    rope_query=self.rope_query,
                    rope_key=self.rope_key,
                    rope_base=self.rope_base,
                    query_grid_hw=twod_rope_nhnw,
                    key_grid_hw=twod_rope_nhnw,
                    attn_dropout=self.attn_dropout,
                    mlp_dropout=self.mlp_dropout,
                    mlp_ratio=self.mlp_ratio,
                    param_dtype=self.param_dtype,
                    out_dtype=z_fused.dtype,
                    activation=self.activation,
                    use_bias_qkv=self.use_bias_qkv,
                    use_bias_out=self.use_bias_out,
                )(
                    query_tokens=z_fused, key_value_tokens=z_cities,
                    attention_bias=attention_bias,
                    deterministic=not training,
                )
            
            for _ in range(self.n_map_crossattn):
                z_fused = TransformerCrossLayer(
                    hidden_dim=z_dim,
                    num_heads=self.attn_num_heads,
                    rope_query=self.rope_query,
                    rope_key=self.rope_key,
                    rope_base=self.rope_base,
                    query_grid_hw=twod_rope_nhnw,
                    key_grid_hw=twod_rope_nhnw,
                    attn_dropout=self.attn_dropout,
                    mlp_dropout=self.mlp_dropout,
                    mlp_ratio=self.mlp_ratio,
                    param_dtype=self.param_dtype,
                    out_dtype=z_fused.dtype,
                    activation=self.activation,
                    use_bias_qkv=self.use_bias_qkv,
                    use_bias_out=self.use_bias_out,
                )(
                    query_tokens=z_fused, key_value_tokens=z_units,
                    attention_bias=attention_bias,
                    deterministic=not training,
                )

            # Finally time to pool all of this into a small representation space 
            z_map = PMA(
                hidden_dim=z_dim,
                num_heads=self.attn_num_heads,
                num_seeds=self.n_pma_seeds,
                dropout_rate=self.pma_dropout,
                param_dtype=self.param_dtype,
                out_dtype=z_fused.dtype
            )(z_fused, deterministic=not training)

            z_map = nn.Dense(self.out_dim)(z_map)
            
            return z_map

    class GamestateEmbedder(nn.Module):
        num_seeds_tech: int
        num_heads_tech: int
        num_seeds_pols: int
        num_heads_pols: int
        num_heads_trade_offer: int
        mlp_ratio_trade_offer: int
        pma_num_heads_trade_offer: int
        pma_num_seeds_trade_offer: int
        num_heads_trade_ledger: int
        mlp_ratio_trade_ledger: int
        pma_num_heads_trade_ledger: int
        pma_num_seeds_trade_ledger: int
        num_heads_trade_length: int
        mlp_ratio_trade_length: int
        pma_num_heads_trade_length: int
        pma_num_seeds_trade_length: int
        num_heads_res_adj: int
        mlp_ratio_res_adj: int
        pma_num_heads_res_adj: int
        pma_num_seeds_res_adj: int
        num_heads_trade_summary: int
        pma_num_heads_trade_summary: int
        pma_num_seeds_trade_summary: int
        num_heads_tenets_inner: int
        mlp_ratio_tenets_inner: int
        pma_num_heads_tenets_inner: int
        pma_num_seeds_tenets_inner: int 
        num_heads_tenets_players: int
        mlp_ratio_tenets_players: int
        pma_num_heads_tenets_players: int
        pma_num_seeds_tenets_players: int
        # City-states
        num_heads_cs: int
        mlp_ratio_cs: int
        pma_num_heads_cs: int
        pma_num_seeds_cs: int
        # Delegates (players)
        num_heads_delegates: int
        mlp_ratio_delegates: int
        pma_num_heads_delegates: int
        pma_num_seeds_delegates: int
        # Great Works (4)
        num_heads_gws: int 
        mlp_ratio_gws: int 
        pma_num_heads_gws: int 
        pma_num_seeds_gws: int

        ## GPPs (6)
        num_heads_gpps: int
        mlp_ratio_gpps: int
        pma_num_heads_gpps: int
        pma_num_seeds_gpps: int
        # Have met (18)
        num_heads_have_met: int
        mlp_ratio_have_met: int
        pma_num_heads_have_met: int
        pma_num_seeds_have_met: int
        ## Tourism (6x6) – inner row/col pooling then across players
        num_heads_tourism_inner: int
        mlp_ratio_tourism_inner: int
        pma_num_heads_tourism_inner: int
        pma_num_seeds_tourism_inner: int
        num_heads_tourism_players: int
        mlp_ratio_tourism_players: int
        pma_num_heads_tourism_players: int
        pma_num_seeds_tourism_players: int

        n_general_selfattn: int
        general_num_heads: int
        general_mlp_ratio: int
        general_pma_num_heads: int
        general_pma_num_seeds: int
        
        n_aggregate_selfattn: int
        aggregate_num_heads: int
        aggregate_mlp_ratio: int
        aggregate_pma_num_heads: int
        aggregate_pma_num_seeds: int

        param_dtype: any = jnp.float32
        activation: Literal["gelu", "silu"] = "gelu"
        use_bias_qkv: bool = False
        use_bias_out: bool = False
        general_attention_bias: any = None
        aggregate_attention_bias: any = None

        @nn.compact
        def __call__(
            self,
            z_tech,
            z_tech_mask,
            z_pols,
            z_pols_mask,
            z_is_researching,
            z_sci_reserves,
            z_culture_reserves,
            z_faith_reserves,
            z_num_trade_routes,
            z_cs_perturn_influence,
            z_num_delegates,
            z_rel_tenets,
            z_free_techs,
            z_free_policies,
            z_gws,
            z_gpps,
            z_golden_age_turns,
            z_trade_offers,
            z_trade_ledger,
            z_trade_length,
            z_trade_gpt,
            z_res_adj,
            z_have_met,
            z_treasury,
            z_tourism_total,
            z_culture_total,
            z_happiness,
            z_most_pop,
            z_least_pop,
            z_most_crop,
            z_least_crop,
            z_most_prod,
            z_least_prod,
            z_most_gnp,
            z_least_gnp,
            z_most_land,
            z_least_land,
            z_most_army,
            z_least_army,
            z_most_approval,
            z_least_approval,
            z_most_literacy,
            z_least_literacy,
            z_culture_cs_resting_influence,
            z_at_war,
            z_has_sacked,
            me_token,
            z_current_turn,
            training
        ):

            def _pool_membership_tokens(tokens, present_mask, *, num_seeds=2, heads=4, name="set", training=False):
                """
                tokens: (B, N, E)  e.g., z_tech or z_pols (absent has its own embedding)
                present_mask: (B, N) 1=present, 0=absent (binary)
                returns: (B, num_seeds, E)
                """
                E = tokens.shape[-1]
                pm = present_mask.astype(jnp.int32)

                # Learned scalar gate per state; init present≈0.99, absent≈0.5 (soft down-weight, not zero)
                def _gate_init(key, shape, dtype):
                    # rows: [absent, present]
                    p_absent  = 0.5
                    p_present = 0.99
                    eps = 1e-6  # avoid inf at exactly 0 or 1
                    def logit(p):
                        p = jnp.clip(p, eps, 1.0 - eps)
                        return jnp.log(p / (1.0 - p))
                    return jnp.array([[logit(p_absent)], [logit(p_present)]], dtype=dtype)

                raw_gate = nn.Embed(2, 1, name=f"{name}_gate", embedding_init=_gate_init)(pm)  # (B,N,1)
                gate = jax.nn.sigmoid(raw_gate)  # (B,N,1)

                z = tokens * gate  # scale, don't erase semantics

                # Optional light SAB so present items can interact once
                z = TransformerCrossLayer(
                    hidden_dim=E, num_heads=heads, rope_query="none", rope_key="none",
                    attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=4.0,
                    use_bias_qkv=False, use_bias_out=False, name=f"{name}_sab1",
                )(z, z, deterministic=not training)

                z_pooled = PMA(
                    hidden_dim=E, num_heads=heads, num_seeds=num_seeds,
                    dropout_rate=0.0, name=f"{name}_pma",
                )(z, deterministic=not training)

                return z_pooled
            
            z_tech = _pool_membership_tokens(
                z_tech, z_tech_mask,
                num_seeds=self.num_seeds_tech, heads=self.num_heads_tech,
                name="tech_set", training=training
            )
            
            z_pols = _pool_membership_tokens(
                z_pols, z_pols_mask,
                num_seeds=self.num_seeds_pols, heads=self.num_heads_pols,
                name="pols_set", training=training
            )

            # ---------- Trade: offers per opponent (B,6,2,E) → (B,6,E) ----------
            B, P, K, E = z_trade_offers.shape  # K=2 ask/offer
            L = P * K

            offers = z_trade_offers.reshape(B, L, E)   # canonical order: [p0.k0, p0.k1, p1.k0, p1.k1, ...]

            # Minimal identity: slot embedding so the model knows direction (ask vs offer)
            slot_ids = jnp.tile(jnp.arange(K, dtype=jnp.int32), reps=(P,))[None, :]  # (1, L)
            slot_ids = jnp.broadcast_to(slot_ids, (B, L))
            offers = offers + nn.Embed(K, E, name="offer_slot_id")(slot_ids)

            # Let offers interact globally with 1D RoPE (order is meaningful and stable)
            offers = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.num_heads_trade_offer,
                rope_query="1d",  # <-- use 1D RoPE here
                rope_key="1d",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.mlp_ratio_trade_offer,
                use_bias_qkv=False,
                use_bias_out=False,
                name="trade_offers_sab1",
            )(offers, offers, deterministic=not training)   # (B, L, E)

            z_trade_offers_tokens = offers  # keep as (B, P*K, E) so the head can pick among competing offers


            # ---------- Trade: active ledger per opponent (B,6,10,2,E) → (B,6,2,E) ----------
            # z_trade_ledger: (B, P=6, D=10, S=2, E)
            # z_trade_length: (B, D=10, E)  -- aligned by deal index d
            B, P, D, S, E = z_trade_ledger.shape

            # (0) Add explicit slot-id (direction) so tokens know offer vs ask
            slot_ids = jnp.arange(S, dtype=jnp.int32)[None, None, None, :, None]  # (1,1,1,S,1)
            slot_ids = jnp.broadcast_to(slot_ids, (B, P, D, S, 1)).squeeze(-1)   # (B,P,D,S)
            z_trade_ledger = z_trade_ledger + nn.Embed(S, E, name="trade_ledger_slot_id")(slot_ids)

            # (1) FiLM-modulate each (p, d, s) with its time-remaining token at index d
            # Broadcast lengths over player & slot: (B,1,D,1,E) -> (B,P,D,S,E)
            len_tok = jnp.broadcast_to(z_trade_length[:, None, :, None, :], (B, P, D, S, E))
            ledger_time_aware = film_modulate(
                z_trade_ledger, len_tok, name="trade_ledger_time_film",
                param_dtype=z_trade_ledger.dtype, out_dtype=z_trade_ledger.dtype
            )  # (B,P,D,S,E)

            # (2) Treat D as an unordered set per (player, slot) → SAB("none") + PMA over D
            x = ledger_time_aware.reshape(B * P * S, D, E)  # collapse (p,slot), keep the D-set

            x = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_trade_ledger,
                rope_query="none", rope_key="none",                 # D has no meaningful order; time info is in features
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_trade_ledger,
                use_bias_qkv=False, use_bias_out=False, name="trade_ledger_items_sab1",
            )(x, x, deterministic=not training)

            x = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_trade_ledger,
                num_seeds=self.pma_num_seeds_trade_ledger, dropout_rate=0.0,
                name="trade_ledger_items_pma",
            )(x, deterministic=not training)                               # (B*P*S, S_ledger, E)

            # (3) Sequence across P×S (canonical order) with 1D RoPE so items can interact globally
            S_ledger = self.pma_num_seeds_trade_ledger
            seq = x.reshape(B, P, S, S_ledger, E).reshape(B, P * S * S_ledger, E)   # (B, P*S*S_ledger, E)

            z_trade_ledger_tokens = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_trade_ledger,
                rope_query="1d", rope_key="1d",                # P and S are fixed-order axes
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_trade_ledger,
                use_bias_qkv=False, use_bias_out=False, name="trade_ledger_seq_sab1",
            )(seq, seq, deterministic=not training)                                  # (B, P*S*S_ledger, E)

            # ---------- Trade: resource adjustments (B,52,E) → (B,4,E) ----------
            B, R, E = z_res_adj.shape
            x = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_res_adj, rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_res_adj,
                use_bias_qkv=False, use_bias_out=False, name="res_adj_sab1",
            )(z_res_adj, z_res_adj, deterministic=not training)            # (B,52,E)

            z_res_adj_pooled = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_res_adj, 
                num_seeds=self.pma_num_seeds_res_adj, dropout_rate=0.0, name="res_adj_pma",
            )(x, deterministic=not training)                                # (B,4,E)

            # ---------- Trade: GPT adjustment stays scalar token (B,E) ----------
            z_trade_gpt_token = z_trade_gpt  # (B,E)

            # ---- Trade fuse: concat -> global SAB (no RoPE) -> PMA summary ----
            E = z_trade_offers_tokens.shape[-1]

            # 1) Concatenate along sequence axis
            trade_seq = jnp.concatenate([
                z_trade_offers_tokens,  # (B, L_offers, E)
                z_trade_ledger_tokens,  # (B, L_ledger, E)
                z_res_adj_pooled,  # (B, S_res, E)
                z_trade_gpt_token[:, None, :],  # (B, 1, E)
            ], axis=1)  # -> (B, L_total, E)

            # 2) Vectorized type IDs (offers=0, ledger=1, res=2, gpt=3)
            B, L_total, E = trade_seq.shape
            L_offers = z_trade_offers_tokens.shape[1]
            L_ledger = z_trade_ledger_tokens.shape[1]
            L_res = z_res_adj_pooled.shape[1]

            type_ids_1d = jnp.concatenate(
                [
                    jnp.full((L_offers,), 0, dtype=jnp.int32),  # offers
                    jnp.full((L_ledger,), 1, dtype=jnp.int32),  # ledger
                    jnp.full((L_res,),    2, dtype=jnp.int32),  # res
                    jnp.full((1,),        3, dtype=jnp.int32),  # gpt
                ],
                axis=0,
            )  # (L_total,)
            #sizes = jnp.array([
            #    z_trade_offers_tokens.shape[1],
            #    z_trade_ledger_tokens.shape[1],
            #    z_res_adj_pooled.shape[1],
            #    1,
            #], dtype=jnp.int32)  # (4,)

            #type_ids_1d = jnp.repeat(jnp.arange(4, dtype=jnp.int32), sizes)   # (L_total,)
            type_ids = jnp.broadcast_to(type_ids_1d[None, :], (B, L_total))   # (B, L_total)

            trade_seq = trade_seq + nn.Embed(4, E, name="trade_fuse_type_id")(type_ids)

            # 3) One global self-attn (set-style: no RoPE)
            trade_seq = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_trade_summary,  # add this field
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=4.0,
                use_bias_qkv=False, use_bias_out=False,
                name="trade_fuse_sab",
            )(trade_seq, trade_seq, deterministic=not training)  # (B, L_total, E)

            # 4) PMA final trade summary token set
            z_trade_summary = PMA(
                hidden_dim=E,
                num_heads=self.pma_num_heads_trade_summary,  # add this field
                num_seeds=self.pma_num_seeds_trade_summary,  # add this field
                dropout_rate=0.0,
                name="trade_fuse_pma",
            )(trade_seq, deterministic=not training)  # (B, S_trade, E)

            # ---------- Religious tenets: per-player set -> player sequence -> summary ----------
            B, P, T, E = z_rel_tenets.shape  # P=6 players, T=91 tenets

            # (1) Per-player set pooling (tenets are unordered)  -- RoPE: none
            x = z_rel_tenets.reshape(B * P, T, E)  # collapse player axis
            x = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.num_heads_tenets_inner,
                rope_query="none",
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.mlp_ratio_tenets_inner,
                use_bias_qkv=False,
                use_bias_out=False,
                name="tenets_inner_sab",
            )(x, x, deterministic=not training)

            x = PMA(
                hidden_dim=E,
                num_heads=self.pma_num_heads_tenets_inner,
                num_seeds=self.pma_num_seeds_tenets_inner,  # S_in (e.g., 1)
                dropout_rate=0.0,
                name="tenets_inner_pma",
            )(x, deterministic=not training)  # (B*P, S_in, E)

            S_in = self.pma_num_seeds_tenets_inner
            # Bring players back (canonical order), flatten seeds across players
            x = x.reshape(B, P, S_in, E).reshape(B, P * S_in, E)  # (B, 6*S_in, E)

            # (2) Across players (fixed order)  -- RoPE: 1d
            x = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.num_heads_tenets_players,
                rope_query="1d",
                rope_key="1d",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.mlp_ratio_tenets_players,
                use_bias_qkv=False,
                use_bias_out=False,
                name="tenets_players_sab",
            )(x, x, deterministic=not training)  # (B, 6*S_in, E)

            # (3) Final summary seeds
            z_rel_tenets_summary = PMA(
                hidden_dim=E,
                num_heads=self.pma_num_heads_tenets_players,
                num_seeds=self.pma_num_seeds_tenets_players,  # S_out (e.g., 2)
                dropout_rate=0.0,
                name="tenets_players_pma",
            )(x, deterministic=not training)  # (B, S_out, E)

            # =========================
            # City-states: z_cs_perturn_influence (B, 12, E)  -- set → SAB(no RoPE) → PMA
            # =========================
            B, C, E = z_cs_perturn_influence.shape

            z_cs_perturn_influence = z_cs_perturn_influence + nn.Embed(C, E, name="cs_id")(
                jnp.arange(C, dtype=jnp.int32)[None]
            )

            x = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_cs,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_cs,
                use_bias_qkv=False, use_bias_out=False, name="cs_influence_sab",
            )(z_cs_perturn_influence, z_cs_perturn_influence, deterministic=not training)

            z_cs_influence_summary = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_cs,
                num_seeds=self.pma_num_seeds_cs, dropout_rate=0.0, name="cs_influence_pma",
            )(x, deterministic=not training)  # (B, S_cs, E)

            # =========================
            # Delegates: z_num_delegates (B, 6, E)  -- fixed player order → SAB(none) → PMA
            # =========================
            B, P, E = z_num_delegates.shape

            # Add identity: player-id embedding (0..P-1). Broadcast batch.
            player_ids = jnp.arange(P, dtype=jnp.int32)[None, :]  # (1, P)
            z_num_delegates = z_num_delegates + nn.Embed(P, E, name="delegates_player_id")(player_ids)  # (B, P, E)
            # Here we need some notation as to "player X is me"
            # but we do not want to FiLM modulate, as this effects the entire sequence.
            anchor = nn.Dense(E, name="delegates_me_anchor")(me_token)[:, None]  # (B,1,E)
            tokens = jnp.concatenate([anchor, z_num_delegates], axis=1)  # (B,1+P,E)

            tokens = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_delegates,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_delegates,
                use_bias_qkv=False, use_bias_out=False, name="delegates_sab_me_anchor",
            )(tokens, tokens, deterministic=not training)

            # Pool to a compact summary
            z_delegates_summary = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_delegates,
                num_seeds=self.pma_num_seeds_delegates, dropout_rate=0.0, name="delegates_pma",
            )(tokens, deterministic=not training)  # (B, S_delegates, E)

            # =========================
            # Great Works: z_gws (B, 4, E)  -- canonical short set → SAB(none) → PMA
            # =========================
            B, G, E = z_gws.shape
            z_gws = z_gws + nn.Embed(G, E, name="gw_id")(
                jnp.arange(G, dtype=jnp.int32)[None]
            )
            x = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_gws,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_gws,
                use_bias_qkv=False, use_bias_out=False, name="gws_sab",
            )(z_gws, z_gws, deterministic=not training)

            z_gws_summary = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_gws,
                num_seeds=self.pma_num_seeds_gws, dropout_rate=0.0, name="gws_pma",
            )(x, deterministic=not training)  # (B, S_gws, E)

            # =========================
            # GPPs: z_gpps (B, 6, E)  -- fixed GP-type order → SAB(none) → PMA
            # =========================
            B, Gp, E = z_gpps.shape
            z_gpps = z_gpps + nn.Embed(Gp, E, name="gpps_id")(
                jnp.arange(Gp, dtype=jnp.int32)[None]
            )
            x = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_gpps,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_gpps,
                use_bias_qkv=False, use_bias_out=False, name="gpps_sab",
            )(z_gpps, z_gpps, deterministic=not training)

            z_gpps_summary = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_gpps,
                num_seeds=self.pma_num_seeds_gpps, dropout_rate=0.0, name="gpps_pma",
            )(x, deterministic=not training)  # (B, S_gpps, E)


            # =========================
            # Have Met: z_have_met (B, 18, E)  -- conceptual set → SAB(no RoPE) → PMA
            # =========================
            B, M, E = z_have_met.shape
            z_have_met = z_have_met + nn.Embed(M, E, name="have_met_id")(
                jnp.arange(M, dtype=jnp.int32)[None]
            )
            x = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_have_met,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_have_met,
                use_bias_qkv=False, use_bias_out=False, name="have_met_sab",
            )(z_have_met, z_have_met, deterministic=not training)

            z_have_met_summary = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_have_met,
                num_seeds=self.pma_num_seeds_have_met, dropout_rate=0.0, name="have_met_pma",
            )(x, deterministic=not training)  # (B, S_met, E)


            # =========================
            # Tourism total: z_tourism_total (B, 6, 6, E)
            #   Step 1: per-source row pooling (set over targets, no RoPE) → (B, 6, S_in, E)
            #   Step 2: per-target column pooling (set over sources, no RoPE) → (B, 6, S_in, E)
            #   Step 3: concat 12 tokens → SAB(1d over players) → PMA
            # =========================
            B, P, Q, E = z_tourism_total.shape  # P=source, Q=target (=6)
            S_in = self.pma_num_seeds_tourism_inner

            # Row pooling (source → targets)
            row = z_tourism_total.reshape(B * P, Q, E)
            row = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_tourism_inner,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_tourism_inner,
                use_bias_qkv=False, use_bias_out=False, name="tourism_row_sab",
            )(row, row, deterministic=not training)
            row = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_tourism_inner,
                num_seeds=S_in, dropout_rate=0.0, name="tourism_row_pma",
            )(row, deterministic=not training)  # (B*P, S_in, E)
            row = row.reshape(B, P * S_in, E)  # (B, 6*S_in, E)

            # Column pooling (target ← sources)
            col = jnp.transpose(z_tourism_total, (0, 2, 1, 3))  # (B, Q=6, P=6, E)
            col = col.reshape(B * Q, P, E)
            col = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_tourism_inner,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_tourism_inner,
                use_bias_qkv=False, use_bias_out=False, name="tourism_col_sab",
            )(col, col, deterministic=not training)
            col = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_tourism_inner,
                num_seeds=S_in, dropout_rate=0.0, name="tourism_col_pma",
            )(col, deterministic=not training)  # (B*Q, S_in, E)
            col = col.reshape(B, Q * S_in, E)  # (B, 6*S_in, E)

            # Combine outbound+inbound by player and mix across the fixed 1-D player order
            tourism_players = jnp.concatenate([row, col], axis=1)  # (B, 12*S_in, E)
            tourism_players = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_tourism_players,
                rope_query="1d", rope_key="1d",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_tourism_players,
                use_bias_qkv=False, use_bias_out=False, name="tourism_players_sab",
            )(tourism_players, tourism_players, deterministic=not training)

            z_tourism_summary = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_tourism_players,
                num_seeds=self.pma_num_seeds_tourism_players, dropout_rate=0.0, name="tourism_players_pma",
            )(tourism_players, deterministic=not training)  # (B, S_tourism, E)

            # =========================
            # At war: z_at_war (B, 6, 6, E)
            #   Step 1: per-source row pooling (set over targets, no RoPE) → (B, 6, S_in, E)
            #   Step 2: per-target column pooling (set over sources, no RoPE) → (B, 6, S_in, E)
            #   Step 3: concat 12 tokens → SAB(1d over players) → PMA
            # =========================
            B, P, Q, E = z_at_war.shape  # P=source, Q=target (=6)
            S_in = self.pma_num_seeds_tourism_inner

            # Row pooling (source → targets)
            row = z_at_war.reshape(B * P, Q, E)
            row = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_tourism_inner,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_tourism_inner,
                use_bias_qkv=False, use_bias_out=False, name="at_war_row_sab",
            )(row, row, deterministic=not training)
            row = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_tourism_inner,
                num_seeds=S_in, dropout_rate=0.0, name="at_war_row_pma",
            )(row, deterministic=not training)  # (B*P, S_in, E)
            row = row.reshape(B, P * S_in, E)  # (B, 6*S_in, E)

            # Column pooling (target ← sources)
            col = jnp.transpose(z_at_war, (0, 2, 1, 3))  # (B, Q=6, P=6, E)
            col = col.reshape(B * Q, P, E)
            col = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_tourism_inner,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_tourism_inner,
                use_bias_qkv=False, use_bias_out=False, name="at_war_col_sab",
            )(col, col, deterministic=not training)
            col = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_tourism_inner,
                num_seeds=S_in, dropout_rate=0.0, name="at_war_col_pma",
            )(col, deterministic=not training)  # (B*Q, S_in, E)
            col = col.reshape(B, Q * S_in, E)  # (B, 6*S_in, E)

            # Combine outbound+inbound by player and mix across the fixed 1-D player order
            at_war = jnp.concatenate([row, col], axis=1)  # (B, 12*S_in, E)
            at_war = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_tourism_players,
                rope_query="1d", rope_key="1d",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_tourism_players,
                use_bias_qkv=False, use_bias_out=False, name="at_war_players_sab",
            )(at_war, at_war, deterministic=not training)

            z_at_war_summary = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_tourism_players,
                num_seeds=self.pma_num_seeds_tourism_players, dropout_rate=0.0, name="at_war_players_pma",
            )(at_war, deterministic=not training)  # (B, S_tourism, E)

            # =========================
            # has sacked: z_has_sacked (B, 6, 6, E)
            #   Step 1: per-source row pooling (set over targets, no RoPE) → (B, 6, S_in, E)
            #   Step 2: per-target column pooling (set over sources, no RoPE) → (B, 6, S_in, E)
            #   Step 3: concat 12 tokens → SAB(1d over players) → PMA
            # =========================
            B, P, Q, E = z_has_sacked.shape  # P=source, Q=target (=6)
            S_in = self.pma_num_seeds_tourism_inner

            # Row pooling (source → targets)
            row = z_has_sacked.reshape(B * P, Q, E)
            row = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_tourism_inner,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_tourism_inner,
                use_bias_qkv=False, use_bias_out=False, name="has_sacked_row_sab",
            )(row, row, deterministic=not training)
            row = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_tourism_inner,
                num_seeds=S_in, dropout_rate=0.0, name="has_sacked_row_pma",
            )(row, deterministic=not training)  # (B*P, S_in, E)
            row = row.reshape(B, P * S_in, E)  # (B, 6*S_in, E)

            # Column pooling (target ← sources)
            col = jnp.transpose(z_has_sacked, (0, 2, 1, 3))  # (B, Q=6, P=6, E)
            col = col.reshape(B * Q, P, E)
            col = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_tourism_inner,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_tourism_inner,
                use_bias_qkv=False, use_bias_out=False, name="has_sacked_col_sab",
            )(col, col, deterministic=not training)
            col = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_tourism_inner,
                num_seeds=S_in, dropout_rate=0.0, name="has_sacked_col_pma",
            )(col, deterministic=not training)  # (B*Q, S_in, E)
            col = col.reshape(B, Q * S_in, E)  # (B, 6*S_in, E)

            # Combine outbound+inbound by player and mix across the fixed 1-D player order
            has_sacked = jnp.concatenate([row, col], axis=1)  # (B, 12*S_in, E)
            has_sacked = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_tourism_players,
                rope_query="1d", rope_key="1d",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_tourism_players,
                use_bias_qkv=False, use_bias_out=False, name="has_sacked_players_sab",
            )(has_sacked, has_sacked, deterministic=not training)

            z_has_sacked_summary = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_tourism_players,
                num_seeds=self.pma_num_seeds_tourism_players, dropout_rate=0.0, name="has_sacked_pma",
            )(has_sacked, deterministic=not training)  # (B, S_tourism, E)



            # Pack scalar-ish tokens into a single sequence (B, N, E)
            tokens = jnp.concatenate([
                z_is_researching[:, None, :],
                z_sci_reserves[:, None, :],
                z_culture_reserves[:, None, :],
                z_culture_total,
                z_faith_reserves[:, None, :],
                z_num_trade_routes[:, None, :],
                z_treasury[:, None, :],
                z_free_techs[:, None, :],
                z_free_policies[:, None, :],
                z_golden_age_turns[:, None, :],
                z_happiness[:, None, :],
                z_most_pop[:, None, :],
                z_least_pop[:, None, :],
                z_most_crop[:, None, :],
                z_least_crop[:, None, :],
                z_most_prod[:, None, :],
                z_least_prod[:, None, :],
                z_most_gnp[:, None, :],
                z_least_gnp[:, None, :],
                z_most_land[:, None, :],
                z_least_land[:, None, :],
                z_most_army[:, None, :],
                z_least_army[:, None, :],
                z_most_approval[:, None, :],
                z_least_approval[:, None, :],
                z_most_literacy[:, None, :],
                z_culture_cs_resting_influence[:, None, :],
                z_least_literacy[:, None, :],
                me_token[:, None, :],
                z_current_turn,  # already (B, 1, E)
            ], axis=1)  # -> (B, N, E)

            E = tokens.shape[-1]

            tokens += nn.Embed(tokens.shape[1], E, name="scalar_token_id")(
                jnp.arange(tokens.shape[1], dtype=jnp.int32)[None, :]
            )

            for _ in range(self.n_general_selfattn):
                tokens = TransformerCrossLayer(
                    hidden_dim=E,
                    num_heads=self.general_num_heads,
                    rope_query="none",
                    rope_key="none",
                    attn_dropout=0.0,
                    mlp_dropout=0.0,
                    mlp_ratio=self.general_mlp_ratio,
                    param_dtype=self.param_dtype,
                    out_dtype=tokens.dtype,
                    activation=self.activation,
                    use_bias_qkv=self.use_bias_qkv,
                    use_bias_out=self.use_bias_out,
                )(
                    query_tokens=tokens, key_value_tokens=tokens,
                    attention_bias=self.general_attention_bias,
                    deterministic=not training,
                )
            
            tokens = PMA(
                hidden_dim=E, num_heads=self.general_pma_num_heads,
                num_seeds=self.general_pma_num_seeds, dropout_rate=0.0, name="general_pma",
            )(tokens, deterministic=not training)  # (B, S_tourism, E)
            
            aggregate_tokens = jnp.concatenate([
                tokens,
                z_tech,
                z_pols,
                z_trade_summary,
                z_rel_tenets_summary,
                z_cs_influence_summary,
                z_delegates_summary,
                z_gws_summary,
                z_gpps_summary,
                z_have_met_summary,
                z_tourism_summary,
                z_at_war_summary,
                z_has_sacked_summary,
            ], axis=1)

            for _ in range(self.n_aggregate_selfattn):
                aggregate_tokens = TransformerCrossLayer(
                    hidden_dim=E,
                    num_heads=self.aggregate_num_heads,
                    rope_query="none",
                    rope_key="none",
                    attn_dropout=0.0,
                    mlp_dropout=0.0,
                    mlp_ratio=self.aggregate_mlp_ratio,
                    param_dtype=self.param_dtype,
                    out_dtype=aggregate_tokens.dtype,
                    activation=self.activation,
                    use_bias_qkv=self.use_bias_qkv,
                    use_bias_out=self.use_bias_out,
                )(
                    query_tokens=aggregate_tokens, key_value_tokens=aggregate_tokens,
                    attention_bias=self.aggregate_attention_bias,
                    deterministic=not training,
                )
            
            aggregate_tokens = PMA(
                hidden_dim=E, num_heads=self.aggregate_pma_num_heads,
                num_seeds=self.aggregate_pma_num_seeds, dropout_rate=0.0, name="aggregate_pma",
            )(aggregate_tokens, deterministic=not training)  # (B, S_tourism, E)

            return aggregate_tokens 
    
    class UnitEncoder(nn.Module):
        units_inner_num_heads: int
        units_inner_mlp_ratio: int
        units_inner_pma_num_heads: int
        units_inner_pma_num_seeds: int
        units_players_num_heads: int
        units_players_mlp_ratio: int
        units_players_pma_num_heads: int
        units_players_pma_num_seeds: int
        trade_yield_num_heads: int
        trade_yield_mlp_ratio: int
        trade_yield_pma_num_heads: int
        trade_yield_pma_num_seeds: int
        trade_units_num_heads: int
        trade_units_mlp_ratio: int
        trade_units_pma_num_heads: int
        trade_units_pma_num_seeds: int
        units_summary_num_heads: int
        units_summary_mlp_ratio: int
        units_summary_pma_num_heads: int
        units_summary_pma_num_seeds: int
        n_units_summary_selfattn: int

        @nn.compact
        def __call__(
            self,
            z_units_nonscatter,
            cb_tok,
            hp_tok,
            ap_tok,
            pos_tok_units,
            z_engaged_n_turns,
            z_action_cat,
            z_trade_to_player_int,
            z_trade_to_city_int,
            z_trade_from_city_int,
            z_trade_yields,
            z_culture_ypk,
            z_culture_hf_ypk,
            z_culture_cs_trade_yields,
            trade_to_player_idx,
            trade_from_city_idx,
            me_token,
            player_embedding_table,
            trade_yields_mask,
            is_caravan_mask,
            training,
        ):
            # First step: mixing the units. These variables are over all units 
            # and have already been  modulated for "known" vs "unknown"
            unit_mix = (
                z_units_nonscatter +
                cb_tok +
                hp_tok +
                ap_tok + 
                pos_tok_units
            )
            unit_mix = nn.LayerNorm(epsilon=1e-6, name="unit_mix_ln")(unit_mix)

            # ---- Step 2: within-player pooling (set over 30 slots) ----
            B, P, S, E = unit_mix.shape  # (B,6,30,E)

            x = unit_mix.reshape(B * P, S, E)

            x = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.units_inner_num_heads,
                rope_query="none",
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.units_inner_mlp_ratio,
                use_bias_qkv=False,
                use_bias_out=False,
                name="units_inner_sab",
            )(x, x, deterministic=not training)  # (B*P, S, E)

            per_player = PMA(
                hidden_dim=E,
                num_heads=self.units_inner_pma_num_heads,
                num_seeds=self.units_inner_pma_num_seeds,   # S_in (e.g., 2–4)
                dropout_rate=0.0,
                name="units_inner_pma",
            )(x, deterministic=not training)             # (B*P, S_in, E)

            per_player = per_player.reshape(B, P, self.units_inner_pma_num_seeds, E)  # (B,6,S_in,E)

            # ---- Step 3: across players (fixed order) + me-aware anchor ----
            B, P, S_in, E = per_player.shape

            # Add player identity to each per-player seed using the passed-in table
            player_ids = jnp.arange(P, dtype=jnp.int32)[None, :, None]  # (1, P, 1)
            player_ids = jnp.broadcast_to(player_ids, (B, P, S_in))  # (B, P, S_in)
            per_player = per_player + player_embedding_table(player_ids)  # (B, P, S_in, E)

            # Flatten players × seeds → a single sequence
            x = per_player.reshape(B, P * S_in, E)  # (B, 6*S_in, E)

            # Me-aware anchor so “me vs others” is explicit
            me_anchor = nn.Dense(E, name="units_me_anchor")(me_token)[:, None, :]  # (B, 1, E)

            seq = jnp.concatenate([me_anchor, x], axis=1)  # (B, 1+6*S_in, E)

            # Let seeds interact across the fixed player order (1D RoPE)
            seq = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.units_players_num_heads,
                rope_query="1d",
                rope_key="1d",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.units_players_mlp_ratio,
                use_bias_qkv=False,
                use_bias_out=False,
                name="units_players_sab",
            )(
                query_tokens=seq, key_value_tokens=seq,
                deterministic=not training,
            )

            # Pool to a compact summary (S_out seeds)
            z_units_summary = PMA(
                hidden_dim=E,
                num_heads=self.units_players_pma_num_heads,
                num_seeds=self.units_players_pma_num_seeds,
                dropout_rate=0.0,
                name="units_players_pma",
            )(seq, deterministic=not training)  # (B, S_out, E)

            # Now let's handle the trade route information. We have now two masks
            # to help us out with that. One for caravan and one for active trade 
            # routes. These two carry overlapping, but not identical, information
            # Tri-state: 0=non-caravan, 1=caravan idle, 2=caravan active
            caravan_state = jnp.where(
                ~is_caravan_mask, 0,
                jnp.where(trade_yields_mask, 2, 1)
            ).astype(jnp.int32)  # (B,30)

            state_tok = nn.Embed(3, E, name="caravan_state_id")(caravan_state)  # (B,30,E)

            # Direction tags (to/from) for endpoint embeddings
            endpoint_dir_tags = nn.Embed(2, E, name="trade_endpoint_dir_id")(
                jnp.array([0, 1], dtype=jnp.int32)
            )  # (2,E)

            # Compose endpoints
            z_to = z_trade_to_player_int + z_trade_to_city_int + endpoint_dir_tags[0][None, None, :]
            z_from = z_trade_from_city_int + endpoint_dir_tags[1][None, None, :]

            # Per-unit yield pooling: add tiny IDs for (dir, yield-type), then SAB→PMA over 20 items
            # IDs: direction 0..1, yield 0..9
            dir_ids = jnp.arange(2, dtype=jnp.int32)[None, None, :, None]  # (1,1,2,1)
            yld_ids = jnp.arange(10, dtype=jnp.int32)[None, None, None, :]  # (1,1,1,10)

            dir_emb = nn.Embed(2, E, name="trade_yield_dir_id")(dir_ids)  # (1,1,2,1,E)
            yld_emb = nn.Embed(10, E, name="trade_yield_type_id")(yld_ids)  # (1,1,1,10,E)

            y = z_trade_yields + dir_emb + yld_emb  # (B,30,2,10,E)
            y = y.reshape(B * S, 2 * 10, E)  # (B*30, 20, E)

            y = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.trade_yield_num_heads,
                rope_query="none",
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.trade_yield_mlp_ratio,
                use_bias_qkv=False,
                use_bias_out=False,
                name="trade_yield_sab",
            )(y, y, deterministic=not training)

            y = PMA(
                hidden_dim=E,
                num_heads=self.trade_yield_pma_num_heads,
                num_seeds=self.trade_yield_pma_num_seeds, 
                dropout_rate=0.0,
                name="trade_yield_pma",
            )(y, deterministic=not training)  # (B*30, S_y, E)

            # collapse S_y seeds via mean if >1, then reshape to (B,30,E)
            y = y.mean(axis=1).reshape(B, S, E)  # (B,30,E)
            # Only contribute when active (keep state_tok to explain why missing)
            y = y * trade_yields_mask[..., None]

            # Per-unit trade subtoken (own units only)
            trade_unit = (
                state_tok +
                z_engaged_n_turns +
                z_action_cat +
                z_to + z_from +
                y
            )  # (B,30,E)

            trade_unit = nn.LayerNorm(epsilon=1e-6, name="trade_unit_ln")(trade_unit)

            # Pool my 30 trade-unit tokens → a few seeds
            trade_seq = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.trade_units_num_heads,
                rope_query="none",
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.trade_units_mlp_ratio,
                use_bias_qkv=False,
                use_bias_out=False,
                name="my_trade_units_sab",
            )(trade_unit, trade_unit, deterministic=not training)  # (B,30,E)

            my_trade_summary = PMA(
                hidden_dim=E,
                num_heads=self.trade_units_pma_num_heads,
                num_seeds=self.trade_units_pma_num_seeds, 
                dropout_rate=0.0,
                name="my_trade_units_pma",
            )(trade_seq, deterministic=not training)  # (B, S_trade, E)

            # Finally concat 
            # Order: [anchor | my_trade_summary | ypks]
            seq = jnp.concatenate([z_units_summary, my_trade_summary, 
                                   z_culture_ypk, z_culture_hf_ypk], axis=1)
            
            for _ in range(self.n_units_summary_selfattn):
                seq = TransformerCrossLayer(
                    hidden_dim=E,
                    num_heads=self.units_summary_num_heads,
                    rope_query="none", 
                    rope_key="none",
                    attn_dropout=0.0,
                    mlp_dropout=0.0,
                    mlp_ratio=self.units_summary_mlp_ratio,
                    use_bias_qkv=False,
                    use_bias_out=False,
                    name=f"units_summary_sab_{_}",
                )(seq, seq, deterministic=not training)

            z_units_summary = PMA(
                hidden_dim=E,
                num_heads=self.units_summary_pma_num_heads,
                num_seeds=self.units_summary_pma_num_seeds,
                dropout_rate=0.0,
                name="units_summary_pma",
            )(seq, deterministic=not training)   # (B, S_out, E)

            return z_units_summary


    class CityStateEncoder(nn.Module):
        # per-CS inner (religion)
        rel_num_heads: int = 4
        rel_mlp_ratio: int = 4
        rel_pma_num_heads: int = 4
        rel_pma_num_seeds: int = 1

        # across CS
        cs_num_heads: int = 4
        cs_mlp_ratio: int = 4
        cs_pma_num_heads: int = 4
        cs_pma_num_seeds: int = 2

        # global trackers
        global_num_heads: int = 4
        global_mlp_ratio: int = 4
        global_pma_num_heads: int = 4
        global_pma_num_seeds: int = 2

        # final fuse
        fuse_num_heads: int = 4
        fuse_mlp_ratio: int = 4
        fuse_pma_num_heads: int = 4
        fuse_pma_num_seeds: int = 2
        n_fuse_selfattn: int = 3

        param_dtype: any = jnp.float32
        activation: Literal["gelu","silu"] = "gelu"
        use_bias_qkv: bool = False
        use_bias_out: bool = False

        @nn.compact
        def __call__(
            self,
            z_religious_population,
            z_cs_relationships,
            z_influence_level,
            z_cs_type,
            z_cs_quest_type,
            z_culture_tracker,
            z_faith_tracker,
            z_tech_tracker,
            z_trade_tracker,
            z_religion_tracker,
            z_wonder_tracker,
            z_resource_tracker,
            z_culture_tracker_ratio,
            z_faith_tracker_ratio,
            z_tech_tracker_ratio,
            z_trade_tracker_ratio,
            z_religion_tracker_ratio,
            z_wonder_tracker_ratio,
            z_resource_tracker_ratio,
            training
        ):
        
            # ---------- per-CS inner: religion (set over 6 religions) ----------
            # (B, 12, 6, z_dim)
            B, C, R, E = z_religious_population.shape
            rel = z_religious_population.reshape(B*C, R, E)

            rel = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.rel_num_heads,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.rel_mlp_ratio,
                use_bias_qkv=self.use_bias_qkv, use_bias_out=self.use_bias_out,
                name="cs_religion_sab",
            )(rel, rel, deterministic=not training)

            rel = PMA(
                hidden_dim=E, num_heads=self.rel_pma_num_heads,
                num_seeds=self.rel_pma_num_seeds, dropout_rate=0.0,
                name="cs_religion_pma",
            )(rel, deterministic=not training)  # (B*C, S_rel, E)

            rel = rel.reshape(B, C, self.rel_pma_num_seeds, E)

            # ---------- per-CS base token + per-CS trade trackers ----------
            base = z_cs_relationships + z_influence_level + z_cs_type + z_cs_quest_type  # (B,12,E)
            base = nn.LayerNorm(epsilon=1e-6, name="cs_base_ln")(base)
            base = base[:, :, None, :]  # (B,12,1,E)

            tt = z_trade_tracker + z_trade_tracker_ratio  # (B,12,E)
            tt = nn.LayerNorm(epsilon=1e-6, name="cs_trade_ln")(tt)
            tt = tt[:, :, None, :]  # (B,12,1,E)

            # stack per-CS seeds: [base | religion_seeds | trade_seed]
            per_cs = jnp.concatenate([base, rel, tt], axis=2)  # (B,12,S_in,E)
            S_in = per_cs.shape[2]
            per_cs = per_cs.reshape(B, C*S_in, E)

            # across CS (fixed list of 12, give it 1D structure)
            per_cs = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.cs_num_heads,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.cs_mlp_ratio,
                use_bias_qkv=self.use_bias_qkv, use_bias_out=self.use_bias_out,
                name="cs_across_sab",
            )(per_cs, per_cs, deterministic=not training)

            z_cs_summary = PMA(
                hidden_dim=E, num_heads=self.cs_pma_num_heads,
                num_seeds=self.cs_pma_num_seeds, dropout_rate=0.0,
                name="cs_across_pma",
            )(per_cs, deterministic=not training)  # (B, S_cs, E)

            # ---------- global trackers (B,E each) → (B, S_global, E) ----------
            E = z_cs_summary.shape[-1]
            global_tokens = [
                z_culture_tracker,
                z_faith_tracker,
                z_tech_tracker,
                z_religion_tracker,
                z_wonder_tracker,
                z_resource_tracker,
                z_culture_tracker_ratio,
                z_faith_tracker_ratio,
                z_tech_tracker_ratio,
                z_religion_tracker_ratio,
                z_wonder_tracker_ratio,
                z_resource_tracker_ratio,
            ]
            N_g = len(global_tokens)
            global_seq = jnp.stack(global_tokens, axis=1)  # (B, N_g, E)

            # type id to separate raw vs ratio and tracker kinds
            type_ids = jnp.arange(N_g, dtype=jnp.int32)[None, :]
            global_seq = global_seq + nn.Embed(N_g, E, name="cs_global_type_id")(type_ids)

            global_seq = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.global_num_heads,
                rope_query="none",
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.global_mlp_ratio,
                use_bias_qkv=self.use_bias_qkv,
                use_bias_out=self.use_bias_out,
                name="cs_global_sab",
            )(global_seq, global_seq, deterministic=not training)

            z_global_summary = PMA(
                hidden_dim=E,
                num_heads=self.global_pma_num_heads,
                num_seeds=self.global_pma_num_seeds,
                dropout_rate=0.0,
                name="cs_global_pma",
            )(global_seq, deterministic=not training)  # (B, S_global, E)

            # ---------- fuse per-CS summary with global trackers ----------
            fuse_seq = jnp.concatenate([z_cs_summary, z_global_summary], axis=1)  # (B, S_cs+S_global, E)
            
            for _ in range(self.n_fuse_selfattn):
                fuse_seq = TransformerCrossLayer(
                    hidden_dim=E,
                    num_heads=self.fuse_num_heads,
                    rope_query="none",
                    rope_key="none",
                    attn_dropout=0.0,
                    mlp_dropout=0.0,
                    mlp_ratio=self.fuse_mlp_ratio,
                    use_bias_qkv=self.use_bias_qkv,
                    use_bias_out=self.use_bias_out,
                    name=f"cs_fuse_sab_{_}",
                )(fuse_seq, fuse_seq, deterministic=not training)

            z_cs_final = PMA(
                hidden_dim=E,
                num_heads=self.fuse_pma_num_heads,
                num_seeds=self.fuse_pma_num_seeds,
                dropout_rate=0.0,
                name="cs_fuse_pma",
            )(fuse_seq, deterministic=not training)  # (B, S_fuse, E)

            return z_cs_final
    
    class CityEncoder(nn.Module):
        pooling_num_heads: int
        pooling_mlp_ratio: int
        pooling_pma_seeds: int
        fuse_num_heads: int
        fuse_mlp_ratio: int
        fuse_pma_num_heads: int
        fuse_pma_num_seeds: int
        relation_num_heads: int
        relation_mlp_ratio: int
        relation_pma_num_heads: int
        relation_pma_num_seeds: int
        n_fuse_selfattn: int
        fuse_pma_num_seeds_final: int

        @nn.compact
        def __call__(
            self,
            z_city_ids,
            z_city_rowcols,
            z_city_yields,
            z_city_center_yields,
            z_building_yields,
            z_culture_building_yields,
            z_religion_building_yields,
            z_city_population,
            z_city_worked_slots,
            z_city_specialists,
            z_city_gws,
            z_city_food_reserves,
            z_city_growth_carryover,
            z_city_prod_reserves,
            z_city_prod_carryover,
            z_city_constructing,
            z_city_building_maintenance,
            z_city_defense,
            z_city_hp,
            z_city_my_buildings,
            z_city_my_resources,
            z_city_is_coastal,
            z_city_culture_reserves_for_border,
            z_city_gpps,
            z_city_religious_population,
            z_my_city_tenets,
            z_cs_perturn_influence_cumsum,
            z_player_perturn_influence_cumsum,
            training,
        ):

            # base city token: sum single-token features + learned tag, then LN → (B, C, 1, E)
            E = z_city_ids.shape[-1]
            B, C = z_city_ids.shape[:2]

            base = (
                z_city_ids
                + z_city_population
                + z_city_food_reserves
                + z_city_growth_carryover
                + z_city_prod_reserves
                + z_city_prod_carryover
                + z_city_building_maintenance
                + z_city_defense
                + z_city_hp
                + z_city_is_coastal
                + z_city_culture_reserves_for_border
            )

            # learned tag so the fuse can recognize this sub-block later
            tag = nn.Embed(1, E, name="city_base_tag")(jnp.zeros((1,), dtype=jnp.int32))  # (1, E)
            tag = jnp.broadcast_to(tag[None], (B, C, E))  # (B, C, E)
            base = base + tag

            base = nn.LayerNorm(epsilon=1e-6, name="city_base_ln")(base)  # (B, C, E)
            base = base[:, :, None, :]  # (B, C, 1, E)

            def pool_city_set(x, *, name: str, seeds: int = 1, heads: int = 4, mlp_ratio: int = 4, training: bool = False):
                B, C = x.shape[:2]
                E = x.shape[-1]
                y = x.reshape(B * C, -1, E)
                y = TransformerCrossLayer(
                    hidden_dim=E, num_heads=heads,
                    rope_query="none", rope_key="none",
                    attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=mlp_ratio,
                    use_bias_qkv=False, use_bias_out=False, name=f"{name}_sab",
                )(y, y, deterministic=not training)
                y = PMA(
                    hidden_dim=E, num_heads=heads,
                    num_seeds=seeds, dropout_rate=0.0, name=f"{name}_pma",
                )(y, deterministic=not training)  # (B*C, seeds, E)
                return y.reshape(B, C, seeds, E)
            
            rowcols_seed = pool_city_set(
                z_city_rowcols, name="city_rowcols", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            center_yields_seed = pool_city_set(
                z_city_center_yields, name="city_center_yields", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            yields_seed = pool_city_set(
                z_city_yields, name="city_yields", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            bldg_yields_seed = pool_city_set(
                z_building_yields, name="city_bldg_yields", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            cult_bldg_yields_seed = pool_city_set(
                z_culture_building_yields, name="city_cult_bldg_yields", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            relig_bldg_yields_seed = pool_city_set(
                z_religion_building_yields, name="city_relig_bldg_yields", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            worked_slots_seed = pool_city_set(
                z_city_worked_slots, name="city_worked_slots", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            specialists_seed = pool_city_set(
                z_city_specialists, name="city_specialists", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            gws_seed = pool_city_set(
                z_city_gws, name="city_gws", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            gpps_seed = pool_city_set(
                z_city_gpps, name="city_gpps", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            relpop_seed = pool_city_set(
                z_city_religious_population, name="city_relpop", seeds=self.pooling_pma_seeds,
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            tenets_seed = pool_city_set(
                z_my_city_tenets, name="city_tenets", seeds=self.pooling_pma_seeds,
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            my_bldgs_seed = pool_city_set(
                z_city_my_buildings, name="city_my_bldgs", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            my_resources_seed = pool_city_set(
                z_city_my_resources, name="city_my_resources", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            cs_inf_cumsum_seed = pool_city_set(
                z_cs_perturn_influence_cumsum, name="city_cs_inf_cumsum", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            city_constructing = pool_city_set(
                z_city_constructing, name="city_constructing", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )

            # flatten (B,C,6,10,E) → T=60 before pooling
            player_inf_cumsum_seed = pool_city_set(
                z_player_perturn_influence_cumsum, name="city_player_inf_cumsum", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )

            per_city = jnp.concatenate([
                base,
                rowcols_seed,
                center_yields_seed,
                yields_seed,
                bldg_yields_seed,
                cult_bldg_yields_seed,
                relig_bldg_yields_seed,
                worked_slots_seed,
                specialists_seed,
                gws_seed,
                gpps_seed,
                relpop_seed,
                tenets_seed,
                my_bldgs_seed,
                my_resources_seed,
                cs_inf_cumsum_seed,
                player_inf_cumsum_seed,
                city_constructing
            ], axis=2)  # (B, C, S_total, E)

            # per_city: (B, C, S_total, E) from Step 2
            B, C, S_total, E = per_city.shape

            # 3a) fuse within each city → a few seeds per city
            y = per_city.reshape(B * C, S_total, E)
            y = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.fuse_num_heads,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.fuse_mlp_ratio,
                use_bias_qkv=False, use_bias_out=False, name="city_inner_fuse_sab",
            )(y, y, deterministic=not training)

            # choose 1–2 seeds per city; 2 gives a little extra capacity
            city_seeds = PMA(
                hidden_dim=E, num_heads=self.fuse_pma_num_heads,
                num_seeds=self.fuse_pma_num_seeds, dropout_rate=0.0, name="city_inner_fuse_pma",
            )(y, deterministic=not training)  # (B*C, 2, E)
            city_seeds = city_seeds.reshape(B, C, self.fuse_pma_num_seeds, E)

            # 3b) tag city slot identity so slots are distinguishable
            city_ids = jnp.arange(C, dtype=jnp.int32)[None, :, None]
            city_ids = jnp.broadcast_to(city_ids, (B, C, self.fuse_pma_num_seeds))
            city_seeds = city_seeds + nn.Embed(C, E, name="city_slot_id")(city_ids)

            # 3c) mix across cities as a set (no RoPE), then pool to a global summary
            seq = city_seeds.reshape(B, C * self.fuse_pma_num_seeds, E)

            seq = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.fuse_num_heads,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.fuse_mlp_ratio,
                use_bias_qkv=False, use_bias_out=False, name="cities_across_sab",
            )(seq, seq, deterministic=not training)

            z_cities_summary = PMA(
                hidden_dim=E, num_heads=self.fuse_pma_num_heads,
                num_seeds=self.fuse_pma_num_seeds, dropout_rate=0.0, name="cities_across_pma",
            )(seq, deterministic=not training)  # (B, 2, E)

            # z_cs_perturn_influence_cumsum: (B, C, 12, E)
            B, C, M, E = z_cs_perturn_influence_cumsum.shape
            cs = z_cs_perturn_influence_cumsum.reshape(B * C, M, E)
            cs = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.relation_num_heads,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.relation_mlp_ratio,
                use_bias_qkv=False, use_bias_out=False, name="city_cs_rel_sab",
            )(cs, cs, deterministic=not training)
            cs = PMA(
                hidden_dim=E, num_heads=self.relation_pma_num_heads,
                num_seeds=self.relation_pma_num_seeds, dropout_rate=0.0, name="city_cs_rel_pma",
            )(cs, deterministic=not training)  # (B*C, 1, E)
            cs = cs.reshape(B, C, self.relation_pma_num_seeds, E)

            # z_player_perturn_influence_cumsum: (B, C, 6, 10, E)
            B, C, P, T, E = z_player_perturn_influence_cumsum.shape
            pt = z_player_perturn_influence_cumsum.reshape(B * C * P, T, E)
            # treat T as time → give it 1D structure
            pt = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.relation_num_heads,
                rope_query="1d", rope_key="1d",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.relation_mlp_ratio,
                use_bias_qkv=False, use_bias_out=False, name="city_player_time_sab",
            )(pt, pt, deterministic=not training)
            pt = PMA(
                hidden_dim=E, num_heads=self.relation_pma_num_heads,
                num_seeds=self.relation_pma_num_seeds, dropout_rate=0.0, name="city_player_time_pma",
            )(pt, deterministic=not training)  # (B*C*P, 1, E)
            pt = pt.reshape(B, C, P, E)

            # players as a small set per city
            pt = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.relation_num_heads,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.relation_mlp_ratio,
                use_bias_qkv=False, use_bias_out=False, name="city_players_rel_sab",
            )(pt.reshape(B * C, P, E), pt.reshape(B * C, P, E), deterministic=not training)
            pt = PMA(
                hidden_dim=E, num_heads=self.relation_pma_num_heads,
                num_seeds=self.relation_pma_num_seeds, dropout_rate=0.0, name="city_players_rel_pma",
            )(pt, deterministic=not training)  # (B*C, 1, E)
            pt = pt.reshape(B, C, self.relation_pma_num_seeds, E)

            # combine relation seeds per city
            rel_city = jnp.concatenate([cs, pt], axis=2)  # (B, C, 2, E)
            rel_city = rel_city.reshape(B, C * 2, E)

            # summarize relation tokens across cities
            rel_city = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.relation_num_heads,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.relation_mlp_ratio,
                use_bias_qkv=False, use_bias_out=False, name="city_rel_across_sab",
            )(rel_city, rel_city, deterministic=not training)
            z_relations_summary = PMA(
                hidden_dim=E, num_heads=self.relation_pma_num_heads,
                num_seeds=self.relation_pma_num_seeds, dropout_rate=0.0, name="city_rel_across_pma",
            )(rel_city, deterministic=not training)  # (B, 2, E)

            # final fuse with z_cities_summary from Step 3
            fuse_seq = jnp.concatenate([z_cities_summary, z_relations_summary], axis=1)  # (B, 4, E)
            for _ in range(self.n_fuse_selfattn):
                fuse_seq = TransformerCrossLayer(
                    hidden_dim=E, num_heads=self.fuse_num_heads,
                    rope_query="none", rope_key="none",
                    attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.fuse_mlp_ratio,
                    use_bias_qkv=False, use_bias_out=False, name=f"city_fuse_sab_{_}",
                )(fuse_seq, fuse_seq, deterministic=not training)

            z_city_final = PMA(
                hidden_dim=E, num_heads=self.fuse_pma_num_heads,
                num_seeds=self.fuse_pma_num_seeds_final, dropout_rate=0.0, name="city_fuse_pma",
            )(fuse_seq, deterministic=not training)  # (B, 2, E)

            return z_city_final

    
    class ValueFunction(nn.Module):
        
        use_stream_gates: bool
        n_selfattn: int
        num_heads: int
        mlp_ratio: int
        pma_num_heads: int
        pma_num_seeds: int
        head_hidden_mult: int

        param_dtype: any = jnp.bfloat16
        activation: Literal["gelu", "silu"] = "gelu"
        use_bias_qkv: bool = False
        use_bias_out: bool = False
        attention_bias: bool = False

        @nn.compact
        def __call__(
            self,
            z_map,
            z_gamestate,
            z_units_nonscatter,
            z_cs,
            z_my_city,
            training,
        ):
            E = z_map.shape[-1]
            streams = [z_map, z_gamestate, z_units_nonscatter, z_cs, z_my_city]
            B = streams[0].shape[0]

            # Tag tokens by source stream so the critic can learn different mixing for each.
            # stream ids: 0=map, 1=game, 2=units, 3=cs, 4=cities
            sizes = [x.shape[1] for x in streams]
            type_ids_1d = jnp.concatenate([jnp.full((n,), i, dtype=jnp.int32) for i, n in enumerate(sizes)], axis=0)
            type_ids = jnp.broadcast_to(type_ids_1d[None, :], (B, type_ids_1d.shape[0]))

            tokens = jnp.concatenate(streams, axis=1)  # (B, L, E)
            tokens = tokens + nn.Embed(5, E, name="vf_type_id")(type_ids)  # (B, L, E)       


            if self.use_stream_gates:
                # Learn one scalar gate per stream (softplus to keep positive, init ~1.0)
                gates = self.param("stream_gates", nn.initializers.constant(0.0), (5, 1), self.param_dtype)
                gate_vals = jax.nn.softplus(gates)  # (5,1)
                tokens = tokens * gate_vals[type_ids]  # broadcast per token

            # A couple of set-style SAB layers (no RoPE).
            for i in range(self.n_selfattn):
                tokens = TransformerCrossLayer(
                    hidden_dim=E,
                    num_heads=self.num_heads,
                    rope_query="none",
                    rope_key="none",
                    attn_dropout=0.0,
                    mlp_dropout=0.0,
                    mlp_ratio=self.mlp_ratio,
                    param_dtype=self.param_dtype,
                    out_dtype=tokens.dtype,
                    activation=self.activation,
                    use_bias_qkv=self.use_bias_qkv,
                    use_bias_out=self.use_bias_out,
                    name=f"vf_sab_{i}",
                )(
                    query_tokens=tokens,
                    key_value_tokens=tokens,
                    attention_bias=self.attention_bias,
                    deterministic=not training,
                )

            # Pool to a compact set of seeds
            seeds = PMA(
                hidden_dim=E,
                num_heads=self.pma_num_heads,
                num_seeds=self.pma_num_seeds,
                dropout_rate=0.0,
                param_dtype=self.param_dtype,
                out_dtype=tokens.dtype,
                name="vf_pma",
            )(tokens, deterministic=not training)  # (B, S_v, E)

            # Flatten seeds and run a tiny MLP head → scalar
            x = seeds.reshape(B, self.pma_num_seeds * E)  # (B, S_v*E)
            x = nn.LayerNorm(epsilon=1e-6, name="vf_head_ln")(x)
            x = nn.Dense(self.head_hidden_mult * E, name="vf_head_fc1")(x)
            x = (jax.nn.gelu if self.activation == "gelu" else jax.nn.silu)(x)
            x = nn.Dense(1, name="vf_head_fc2")(x)  # (B, 1)
            # final value is float32 for stability
            return x.astype(jnp.float32)


    class ActionHeadTradeDeals(nn.Module):
        
        use_stream_gates: bool
        n_selfattn: int
        num_heads: int
        mlp_ratio: int
        pma_num_heads: int
        pma_num_seeds: int
        num_heads_trade_offer: int
        mlp_ratio_trade_offer: int
        num_heads_trade_ledger: int
        mlp_ratio_trade_ledger: int
        pma_num_heads_trade_ledger: int
        pma_num_seeds_trade_ledger: int
        num_heads_res_adj: int
        mlp_ratio_res_adj: int
        pma_num_heads_res_adj: int
        pma_num_seeds_res_adj: int
        num_heads_trade_summary: int
        mlp_ratio_trade_summary: int
        pma_num_heads_trade_summary: int
        pma_num_seeds_trade_summary: int

        param_dtype: any = jnp.bfloat16
        activation: Literal["gelu", "silu"] = "gelu"
        use_bias_qkv: bool = False
        use_bias_out: bool = False
        attention_bias: any = None

        @nn.compact
        def __call__(
            self,
            z_map,
            z_gamestate,
            z_units_nonscatter,
            z_cs,
            z_my_city,
            z_trade_offers,
            z_trade_ledger,
            z_trade_length,
            z_trade_gpt,
            z_res_adj,
            z_have_met,
            z_treasury,
            z_resources_owned_percity,
            training,
        ):
            E = z_map.shape[-1]
            streams = [z_map, z_gamestate, z_units_nonscatter, z_cs, z_my_city]
            B = streams[0].shape[0]
            
            # Learned query for each opponent slot
            opp_queries = nn.Embed(6, E, name="opponent_embeddings")(jnp.arange(6, dtype=jnp.int32))[None]  # (1,6,E)
            opp_queries = jnp.broadcast_to(opp_queries, (B, *opp_queries.shape[1:]))

            # Tag tokens by source stream so the critic can learn different mixing for each.
            # stream ids: 0=map, 1=game, 2=units, 3=cs, 4=cities
            sizes = [x.shape[1] for x in streams]
            type_ids_1d = jnp.concatenate([jnp.full((n,), i, dtype=jnp.int32) for i, n in enumerate(sizes)], axis=0)
            type_ids = jnp.broadcast_to(type_ids_1d[None, :], (B, type_ids_1d.shape[0]))

            tokens = jnp.concatenate(streams, axis=1)  # (B, L, E)
            tokens = tokens + nn.Embed(5, E, name="vf_type_id")(type_ids)  # (B, L, E)       


            if self.use_stream_gates:
                # Learn one scalar gate per stream (softplus to keep positive, init ~1.0)
                gates = self.param("stream_gates", nn.initializers.constant(0.0), (5, 1), self.param_dtype)
                gate_vals = jax.nn.softplus(gates)  # (5,1)
                tokens = tokens * gate_vals[type_ids]  # broadcast per token

            # A couple of set-style SAB layers (no RoPE).
            for i in range(self.n_selfattn):
                tokens = TransformerCrossLayer(
                    hidden_dim=E,
                    num_heads=self.num_heads,
                    rope_query="none",
                    rope_key="none",
                    attn_dropout=0.0,
                    mlp_dropout=0.0,
                    mlp_ratio=self.mlp_ratio,
                    param_dtype=self.param_dtype,
                    out_dtype=tokens.dtype,
                    activation=self.activation,
                    use_bias_qkv=self.use_bias_qkv,
                    use_bias_out=self.use_bias_out,
                    name=f"vf_sab_{i}",
                )(
                    query_tokens=tokens,
                    key_value_tokens=tokens,
                    attention_bias=self.attention_bias,
                    deterministic=not training,
                )

            # Pool to a compact set of seeds
            global_summary_seeds = PMA(
                hidden_dim=E,
                num_heads=self.pma_num_heads,
                num_seeds=self.pma_num_seeds,
                dropout_rate=0.0,
                param_dtype=self.param_dtype,
                out_dtype=tokens.dtype,
                name="vf_pma",
            )(tokens, deterministic=not training)  # (B, S_v, E)

            # ---------- Trade: offers per opponent (B,6,2,E) → (B,6,E) ----------
            B, P, K, E = z_trade_offers.shape  # K=2 ask/offer
            L = P * K

            offers = z_trade_offers.reshape(B, L, E)   # canonical order: [p0.k0, p0.k1, p1.k0, p1.k1, ...]

            # Minimal identity: slot embedding so the model knows direction (ask vs offer)
            slot_ids = jnp.tile(jnp.arange(K, dtype=jnp.int32), reps=(P,))[None, :]  # (1, L)
            slot_ids = jnp.broadcast_to(slot_ids, (B, L))
            offers = offers + nn.Embed(K, E, name="offer_slot_id")(slot_ids)

            # Let offers interact globally with 1D RoPE (order is meaningful and stable)
            offers = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.num_heads_trade_offer,
                rope_query="1d",  # <-- use 1D RoPE here
                rope_key="1d",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.mlp_ratio_trade_offer,
                use_bias_qkv=False,
                use_bias_out=False,
                name="trade_offers_sab1",
            )(offers, offers, deterministic=not training)   # (B, L, E)

            z_trade_offers_tokens = offers  # keep as (B, P*K, E) so the head can pick among competing offers

            # ---------- Trade: active ledger per opponent (B,6,10,2,E) → (B,6,2,E) ----------
            # z_trade_ledger: (B, P=6, D=10, S=2, E)
            # z_trade_length: (B, D=10, E)  -- aligned by deal index d
            B, P, D, S, E = z_trade_ledger.shape

            # (0) Add explicit slot-id (direction) so tokens know offer vs ask
            slot_ids = jnp.arange(S, dtype=jnp.int32)[None, None, None, :, None]  # (1,1,1,S,1)
            slot_ids = jnp.broadcast_to(slot_ids, (B, P, D, S, 1)).squeeze(-1)   # (B,P,D,S)
            z_trade_ledger = z_trade_ledger + nn.Embed(S, E, name="trade_ledger_slot_id")(slot_ids)

            # (1) FiLM-modulate each (p, d, s) with its time-remaining token at index d
            # Broadcast lengths over player & slot: (B,1,D,1,E) -> (B,P,D,S,E)
            len_tok = jnp.broadcast_to(z_trade_length[:, None, :, None, :], (B, P, D, S, E))
            ledger_time_aware = film_modulate(
                z_trade_ledger, len_tok, name="trade_ledger_time_film",
                param_dtype=z_trade_ledger.dtype, out_dtype=z_trade_ledger.dtype
            )  # (B,P,D,S,E)

            # (2) Treat D as an unordered set per (player, slot) → SAB("none") + PMA over D
            x = ledger_time_aware.reshape(B * P * S, D, E)  # collapse (p,slot), keep the D-set

            x = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_trade_ledger,
                rope_query="none", rope_key="none",                 # D has no meaningful order; time info is in features
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_trade_ledger,
                use_bias_qkv=False, use_bias_out=False, name="trade_ledger_items_sab1",
            )(x, x, deterministic=not training)

            x = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_trade_ledger,
                num_seeds=self.pma_num_seeds_trade_ledger, dropout_rate=0.0,
                name="trade_ledger_items_pma",
            )(x, deterministic=not training)                               # (B*P*S, S_ledger, E)

            # (3) Sequence across P×S (canonical order) with 1D RoPE so items can interact globally
            S_ledger = self.pma_num_seeds_trade_ledger
            seq = x.reshape(B, P, S, S_ledger, E).reshape(B, P * S * S_ledger, E)   # (B, P*S*S_ledger, E)

            z_trade_ledger_tokens = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_trade_ledger,
                rope_query="1d", rope_key="1d",                # P and S are fixed-order axes
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_trade_ledger,
                use_bias_qkv=False, use_bias_out=False, name="trade_ledger_seq_sab1",
            )(seq, seq, deterministic=not training)                                  # (B, P*S*S_ledger, E)

            # ---------- Trade: resource adjustments (B,52,E) → (B,4,E) ----------
            B, R, E = z_res_adj.shape
            x = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_res_adj, rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_res_adj,
                use_bias_qkv=False, use_bias_out=False, name="res_adj_sab1",
            )(z_res_adj, z_res_adj, deterministic=not training)            # (B,52,E)

            z_res_adj_pooled = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_res_adj, 
                num_seeds=self.pma_num_seeds_res_adj, dropout_rate=0.0, name="res_adj_pma",
            )(x, deterministic=not training)                                # (B,4,E)

            # ---------- Trade: GPT adjustment stays scalar token (B,E) ----------
            z_trade_gpt_token = z_trade_gpt  # (B,E)

            # ---- Trade fuse: concat -> global SAB (no RoPE) -> PMA summary ----
            E = z_trade_offers_tokens.shape[-1]

            # 1) Concatenate along sequence axis
            trade_seq = jnp.concatenate([
                z_trade_offers_tokens,  # (B, L_offers, E)
                z_trade_ledger_tokens,  # (B, L_ledger, E)
                z_res_adj_pooled,  # (B, S_res, E)
                z_trade_gpt_token[:, None, :],  # (B, 1, E)
            ], axis=1)  # -> (B, L_total, E)

            # 2) Vectorized type IDs (offers=0, ledger=1, res=2, gpt=3)
            B, L_total, E = trade_seq.shape
            sizes = jnp.array([
                z_trade_offers_tokens.shape[1],
                z_trade_ledger_tokens.shape[1],
                z_res_adj_pooled.shape[1],
                1,
            ], dtype=jnp.int32)  # (4,)

            type_ids_1d = jnp.repeat(jnp.arange(4, dtype=jnp.int32), sizes)   # (L_total,)
            type_ids = jnp.broadcast_to(type_ids_1d[None, :], (B, L_total))   # (B, L_total)

            trade_seq = trade_seq + nn.Embed(4, E, name="trade_fuse_type_id")(type_ids)

            # 3) One global self-attn (set-style: no RoPE)
            trade_seq = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_trade_summary,  # add this field
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=4.0,
                use_bias_qkv=False, use_bias_out=False,
                name="trade_fuse_sab",
            )(trade_seq, trade_seq, deterministic=not training)  # (B, L_total, E)

            z_trade_summary = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=2,  # force to be divisible
                rope_query="none", 
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.mlp_ratio_trade_summary,
                use_bias_qkv=False,
                use_bias_out=False,
                name="trade_xattn",
            )(
                query_tokens=opp_queries,
                key_value_tokens=trade_seq,
                deterministic=not training,
            )  # (B,6,E)
            
            # We'll update the z_trade_summary via Q in QKV
            # Need to first summarize resources owned across all cities in to one global view
            res_by_r = z_resources_owned_percity.mean(axis=1)  # (B, R, E)

            # (Optional) add resource ID so the model can distinguish resource slots cleanly
            res_ids = jnp.arange(R, dtype=jnp.int32)[None, :]  # (1,R)
            res_by_r = res_by_r + nn.Embed(R, E, name="resource_id")(res_ids)

            # Light set mixing across resources (no RoPE), then pool to a few seeds
            res_by_r = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_trade_summary,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_trade_summary,
                use_bias_qkv=False, use_bias_out=False, name="res_mem_sab",
            )(res_by_r, res_by_r, deterministic=not training)  # (B,R,E)

            res_mem = PMA(
                hidden_dim=E, num_heads=self.pma_num_heads_trade_summary,
                num_seeds=self.pma_num_seeds_trade_summary, dropout_rate=0.0, name="res_mem_pma",
            )(res_by_r, deterministic=not training)  # (B, S_res, E)

            # Finally constructing the KV
            type_tokens = nn.Embed(3, E, name="ca_type_id")(
                jnp.concatenate([
                    jnp.zeros(shape=(res_mem.shape[1],), dtype=jnp.int32),
                    jnp.zeros(shape=(6,), dtype=jnp.int32) + 1,
                    jnp.zeros(shape=(1,), dtype=jnp.int32) + 2
                ])
            )
            z_current_affairs = jnp.concatenate([res_mem, z_have_met[:, :6], z_treasury[:, None]], axis=1)
            z_current_affairs = z_current_affairs + type_tokens[None]

            z_current_affairs = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_trade_summary,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_trade_summary,
                use_bias_qkv=False, use_bias_out=False, name="current_affairs_sab",
            )(z_current_affairs, z_current_affairs, deterministic=not training)  # (B,R,E)
            
            # TODO: this breaks slot semantics
            #z_current_affairs = PMA(
            #    hidden_dim=E, num_heads=self.pma_num_heads_trade_summary,
            #    num_seeds=self.pma_num_seeds_trade_summary, dropout_rate=0.0, name="current_affairs_pma",
            #)(z_current_affairs, deterministic=not training)  # (B, S_res, E)
            
            z_current_affairs = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=2,  # force to be divisible
                rope_query="none", 
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.mlp_ratio_trade_summary,
                use_bias_qkv=False,
                use_bias_out=False,
                name="trade_xattn2",
            )(
                query_tokens=opp_queries,
                key_value_tokens=z_current_affairs,
                deterministic=not training,
            )  # (B,6,E)

            # Cross attention updating our trade summary
            z_updated_trade_info = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_trade_summary,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_trade_summary,
                use_bias_qkv=False, use_bias_out=False, name="updated_xab",
            )(z_trade_summary, z_current_affairs, deterministic=not training)  # (B,R,E)
            
            z_updated_trade_info = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.num_heads_trade_summary,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.mlp_ratio_trade_summary,
                use_bias_qkv=False, use_bias_out=False, name="updated_xab2",
            )(z_updated_trade_info, global_summary_seeds, deterministic=not training)  # (B,R,E)
            
            
            # Cross-attend: queries = the 6 opponent slots, keys/values = trade summary tokens
            opp_ctx = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=2,  # force to be divisible
                rope_query="none", 
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.mlp_ratio_trade_summary,
                use_bias_qkv=False,
                use_bias_out=False,
                name="trade_xattn3",
            )(
                query_tokens=opp_queries,
                key_value_tokens=z_updated_trade_info,
                deterministic=not training,
            )  # (B,6,E)

            # (1) (B, 6, 2)
            action_acceptdeny = nn.Dense(2, name="acceptdeny_logits")(opp_ctx)  # (B,6,2)

            # (2) (B, 6, 56)
            action_ask = nn.Dense(len(ALL_RESOURCES) + 4)(opp_ctx)

            # (3) (B, 56)
            E = z_updated_trade_info.shape[-1]
            B = z_updated_trade_info.shape[0]

            # Learned global query token (1,1,E) -> (B,1,E)
            offer_q = nn.Embed(1, E, name="offer_query_token")(jnp.array([0], dtype=jnp.int32))[None, :, :]
            offer_q = jnp.broadcast_to(offer_q, (B, 1, E))
            

            # Cross-attend the global query to the trade-summary keys/values
            offer_ctx = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.num_heads_trade_summary,
                rope_query="none",
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.mlp_ratio_trade_summary,
                use_bias_qkv=False,
                use_bias_out=False,
                name="offer_xattn",
            )(
                query_tokens=offer_q,  # (B,1,E)
                key_value_tokens=z_updated_trade_info,  # (B,6,E) after PMA-to-6
                deterministic=not training,
            )  # -> (B,1,E)

            # Project to offer logits: (B, 56) = len(ALL_RESOURCES) + 4
            action_offer = nn.Dense(len(ALL_RESOURCES) + 4)(offer_ctx[:, 0, :])
            #action_offer = opp_ctx.reshape(opp_ctx.shape[0], -1)
            #action_offer = nn.Dense(len(ALL_RESOURCES) + 4)(action_offer)

            # (4) (B, 6)
            action_counterparty = nn.Dense(1)(opp_ctx).squeeze(-1)

            return (action_acceptdeny, action_ask, action_offer, action_counterparty)
    

    class ActionHeadSocialPolicies(nn.Module):
        use_stream_gates: bool
        n_selfattn: int
        num_heads: int
        mlp_ratio: int
        pma_num_heads: int
        pma_num_seeds: int
        num_heads_pols: int
        n_general_selfattn: int
        general_num_heads: int
        general_mlp_ratio: int
        
        param_dtype: any = jnp.bfloat16
        activation: Literal["gelu", "silu"] = "gelu"
        use_bias_qkv: bool = False
        use_bias_out: bool = False
        attention_bias: any = None


        @nn.compact
        def __call__(
            self,
            z_map,
            z_gamestate,
            z_units_nonscatter,
            z_cs,
            z_my_city,
            z_culture_reserves,
            z_free_policies,
            z_pols,
            z_pols_mask,
            training,
        ):
            E = z_map.shape[-1]
            streams = [z_map, z_gamestate, z_units_nonscatter, z_cs, z_my_city]
            B = streams[0].shape[0]

            # Tag tokens by source stream so the critic can learn different mixing for each.
            # stream ids: 0=map, 1=game, 2=units, 3=cs, 4=cities
            sizes = [x.shape[1] for x in streams]
            type_ids_1d = jnp.concatenate([jnp.full((n,), i, dtype=jnp.int32) for i, n in enumerate(sizes)], axis=0)
            type_ids = jnp.broadcast_to(type_ids_1d[None, :], (B, type_ids_1d.shape[0]))

            tokens = jnp.concatenate(streams, axis=1)  # (B, L, E)
            tokens = tokens + nn.Embed(5, E, name="vf_type_id")(type_ids)  # (B, L, E)       


            if self.use_stream_gates:
                # Learn one scalar gate per stream (softplus to keep positive, init ~1.0)
                gates = self.param("stream_gates", nn.initializers.constant(0.0), (5, 1), self.param_dtype)
                gate_vals = jax.nn.softplus(gates)  # (5,1)
                tokens = tokens * gate_vals[type_ids]  # broadcast per token

            # A couple of set-style SAB layers (no RoPE).
            for i in range(self.n_selfattn):
                tokens = TransformerCrossLayer(
                    hidden_dim=E,
                    num_heads=self.num_heads,
                    rope_query="none",
                    rope_key="none",
                    attn_dropout=0.0,
                    mlp_dropout=0.0,
                    mlp_ratio=self.mlp_ratio,
                    param_dtype=self.param_dtype,
                    out_dtype=tokens.dtype,
                    activation=self.activation,
                    use_bias_qkv=self.use_bias_qkv,
                    use_bias_out=self.use_bias_out,
                    name=f"vf_sab_{i}",
                )(
                    query_tokens=tokens,
                    key_value_tokens=tokens,
                    attention_bias=self.attention_bias,
                    deterministic=not training,
                )

            # Pool to a compact set of seeds
            global_summary_seeds = PMA(
                hidden_dim=E,
                num_heads=self.pma_num_heads,
                num_seeds=self.pma_num_seeds,
                dropout_rate=0.0,
                param_dtype=self.param_dtype,
                out_dtype=tokens.dtype,
                name="vf_pma",
            )(tokens, deterministic=not training)  # (B, S_v, E)
            
            def _pool_membership_tokens(tokens, present_mask, *, num_seeds=2, heads=4, name="set", training=False):
                """
                tokens: (B, N, E)  e.g., z_tech or z_pols (absent has its own embedding)
                present_mask: (B, N) 1=present, 0=absent (binary)
                returns: (B, num_seeds, E)
                """
                E = tokens.shape[-1]
                pm = present_mask.astype(jnp.int32)

                # Learned scalar gate per state; init present≈0.99, absent≈0.5 (soft down-weight, not zero)
                def _gate_init(key, shape, dtype):
                    # rows: [absent, present]
                    p_absent  = 0.5
                    p_present = 0.99
                    eps = 1e-6  # avoid inf at exactly 0 or 1
                    def logit(p):
                        p = jnp.clip(p, eps, 1.0 - eps)
                        return jnp.log(p / (1.0 - p))
                    return jnp.array([[logit(p_absent)], [logit(p_present)]], dtype=dtype)

                raw_gate = nn.Embed(2, 1, name=f"{name}_gate", embedding_init=_gate_init)(pm)  # (B,N,1)
                gate = jax.nn.sigmoid(raw_gate)  # (B,N,1)

                z = tokens * gate  # scale, don't erase semantics

                # Optional light SAB so present items can interact once
                z = TransformerCrossLayer(
                    hidden_dim=E, num_heads=heads, rope_query="1d", rope_key="1d",
                    attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=4.0,
                    use_bias_qkv=False, use_bias_out=False, name=f"{name}_sab1",
                )(z, z, deterministic=not training)
                
                return z
            
            # These should carry some relative-position semantics!
            slot_ids = jnp.arange(z_pols.shape[1], dtype=jnp.int32)[None, :]
            z_pols = z_pols + nn.Embed(z_pols.shape[1], E, name="policy_slot_id")(slot_ids)

            z_pols = _pool_membership_tokens(
                z_pols, z_pols_mask,
                num_seeds=None, heads=self.num_heads_pols,
                name="pols_set", training=training
            )

            culture_tokens = jnp.concatenate([
                global_summary_seeds,
                z_culture_reserves[:, None],
                z_free_policies[:, None],
            ], axis=1)
            
            z_pols_ctx = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.general_num_heads,
                rope_query="1d",
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.general_mlp_ratio,
                param_dtype=self.param_dtype,
                out_dtype=tokens.dtype,
                activation=self.activation,
                use_bias_qkv=self.use_bias_qkv,
                use_bias_out=self.use_bias_out,
            )(
                query_tokens=z_pols, key_value_tokens=culture_tokens,
                attention_bias=self.attention_bias,
                deterministic=not training,
            )
            
            for _ in range(self.n_general_selfattn):
                z_pols_ctx = TransformerCrossLayer(
                    hidden_dim=E,
                    num_heads=self.general_num_heads,
                    rope_query="1d",
                    rope_key="1d",
                    attn_dropout=0.0,
                    mlp_dropout=0.0,
                    mlp_ratio=self.general_mlp_ratio,
                    param_dtype=self.param_dtype,
                    out_dtype=z_pols_ctx.dtype,
                    activation=self.activation,
                    use_bias_qkv=self.use_bias_qkv,
                    use_bias_out=self.use_bias_out,
                )(
                    query_tokens=z_pols_ctx, key_value_tokens=z_pols_ctx,
                    attention_bias=self.attention_bias,
                    deterministic=not training,
                )
            
            actions_social_policies = nn.Dense(1)(z_pols_ctx).squeeze(-1)

            return actions_social_policies
    

    class ActionHeadReligion(nn.Module):
        use_stream_gates: bool
        n_selfattn: int
        num_heads: int
        mlp_ratio: int
        pma_num_heads: int
        pma_num_seeds: int
        pma_num_heads_tenets_inner: int
        pma_num_seeds_tenets_inner: int
        num_heads_tenets_inner: int
        mlp_ratio_tenets_inner: int
        general_num_heads: int
        general_mlp_ratio: int
        n_general_selfattn: int
        
        param_dtype: any = jnp.bfloat16
        activation: Literal["gelu", "silu"] = "gelu"
        use_bias_qkv: bool = False
        use_bias_out: bool = False
        attention_bias: any = None


        @nn.compact
        def __call__(
            self,
            z_map,
            z_gamestate,
            z_units_nonscatter,
            z_cs,
            z_my_city,
            z_rel_tenets,
            rel_tenets_mask,
            z_faith_reserves,
            me_idx,
            games_idx,
            player_embedding_table,
            training,
        ):
            E = z_map.shape[-1]
            streams = [z_map, z_gamestate, z_units_nonscatter, z_cs, z_my_city]
            B = streams[0].shape[0]

            # Tag tokens by source stream so the critic can learn different mixing for each.
            # stream ids: 0=map, 1=game, 2=units, 3=cs, 4=cities
            sizes = [x.shape[1] for x in streams]
            type_ids_1d = jnp.concatenate([jnp.full((n,), i, dtype=jnp.int32) for i, n in enumerate(sizes)], axis=0)
            type_ids = jnp.broadcast_to(type_ids_1d[None, :], (B, type_ids_1d.shape[0]))

            tokens = jnp.concatenate(streams, axis=1)  # (B, L, E)
            tokens = tokens + nn.Embed(5, E, name="vf_type_id")(type_ids)  # (B, L, E)       


            if self.use_stream_gates:
                # Learn one scalar gate per stream (softplus to keep positive, init ~1.0)
                gates = self.param("stream_gates", nn.initializers.constant(0.0), (5, 1), self.param_dtype)
                gate_vals = jax.nn.softplus(gates)  # (5,1)
                tokens = tokens * gate_vals[type_ids]  # broadcast per token

            # A couple of set-style SAB layers (no RoPE).
            for i in range(self.n_selfattn):
                tokens = TransformerCrossLayer(
                    hidden_dim=E,
                    num_heads=self.num_heads,
                    rope_query="none",
                    rope_key="none",
                    attn_dropout=0.0,
                    mlp_dropout=0.0,
                    mlp_ratio=self.mlp_ratio,
                    param_dtype=self.param_dtype,
                    out_dtype=tokens.dtype,
                    activation=self.activation,
                    use_bias_qkv=self.use_bias_qkv,
                    use_bias_out=self.use_bias_out,
                    name=f"vf_sab_{i}",
                )(
                    query_tokens=tokens,
                    key_value_tokens=tokens,
                    attention_bias=self.attention_bias,
                    deterministic=not training,
                )

            # Pool to a compact set of seeds
            global_summary_seeds = PMA(
                hidden_dim=E,
                num_heads=self.pma_num_heads,
                num_seeds=self.pma_num_seeds,
                dropout_rate=0.0,
                param_dtype=self.param_dtype,
                out_dtype=tokens.dtype,
                name="vf_pma",
            )(tokens, deterministic=not training)  # (B, S_v, E)
            
            # ---------- Religious tenets: per-player set -> player sequence -> summary ----------
            B, P, T, E = z_rel_tenets.shape  # P=6 players, T=91 tenets
            z_rel_tenets = (
                z_rel_tenets + 
                nn.Embed(2, E, name="religion_tenets_mask_bias")(rel_tenets_mask.astype(jnp.int32))
            )

            # (1) Per-player set pooling (tenets are unordered)  -- RoPE: none
            x = z_rel_tenets.reshape(B * P, T, E)  # collapse player axis
            x = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.num_heads_tenets_inner,
                rope_query="1d",
                rope_key="1d",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.mlp_ratio_tenets_inner,
                use_bias_qkv=False,
                use_bias_out=False,
                name="tenets_inner_sab",
            )(x, x, deterministic=not training)
            z_all_players_tenets = x.reshape(B, P, T, E)
            z_my_tenets = z_all_players_tenets[games_idx, me_idx]

            # Summarizing on a per-player level, then adding a per-player learned 
            # embedding, then flattening.
            x = PMA(
                hidden_dim=E,
                num_heads=self.pma_num_heads_tenets_inner,
                num_seeds=self.pma_num_seeds_tenets_inner,  # S_in (e.g., 1)
                dropout_rate=0.0,
                name="tenets_inner_pma",
            )(x, deterministic=not training)  # (B*P, S_in, E)

            S_in = self.pma_num_seeds_tenets_inner
            # Bring players back (canonical order), flatten seeds across players
            x = x.reshape(B, P, S_in, E)
            
            # Add player identity to each per-player seed using the passed-in table
            player_ids = jnp.arange(P, dtype=jnp.int32)[None, :, None]  # (1, P, 1)
            player_ids = jnp.broadcast_to(player_ids, (B, P, S_in))  # (B, P, S_in)
            per_player = x + player_embedding_table(player_ids)  # (B, P, S_in, E)
            per_player = per_player.reshape(B, P * S_in, E)  # (B, 6*S_in, E)

            # Updating our global information using the
            religion_tokens = jnp.concatenate([
                global_summary_seeds,
                z_faith_reserves[:, None]
            ], axis=1)
            
            z_tenets_ctx = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.general_num_heads,
                rope_query="1d",
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.general_mlp_ratio,
                param_dtype=self.param_dtype,
                out_dtype=tokens.dtype,
                activation=self.activation,
                use_bias_qkv=self.use_bias_qkv,
                use_bias_out=self.use_bias_out,
            )(
                query_tokens=z_my_tenets, key_value_tokens=religion_tokens,
                attention_bias=self.attention_bias,
                deterministic=not training,
            )
            
            z_tenets_ctx = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.general_num_heads,
                rope_query="1d",
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.general_mlp_ratio,
                param_dtype=self.param_dtype,
                out_dtype=tokens.dtype,
                activation=self.activation,
                use_bias_qkv=self.use_bias_qkv,
                use_bias_out=self.use_bias_out,
            )(
                query_tokens=z_tenets_ctx, key_value_tokens=per_player,
                attention_bias=self.attention_bias,
                deterministic=not training,
            )

            for _ in range(self.n_general_selfattn):
                z_tenets_ctx = TransformerCrossLayer(
                    hidden_dim=E,
                    num_heads=self.general_num_heads,
                    rope_query="1d",
                    rope_key="1d",
                    attn_dropout=0.0,
                    mlp_dropout=0.0,
                    mlp_ratio=self.general_mlp_ratio,
                    param_dtype=self.param_dtype,
                    out_dtype=z_tenets_ctx.dtype,
                    activation=self.activation,
                    use_bias_qkv=self.use_bias_qkv,
                    use_bias_out=self.use_bias_out,
                )(
                    query_tokens=z_tenets_ctx, key_value_tokens=z_tenets_ctx,
                    attention_bias=self.attention_bias,
                    deterministic=not training,
                )
            
            actions_religion = nn.Dense(1)(z_tenets_ctx).squeeze(-1)
            return actions_religion 


    class ActionHeadTechnology(nn.Module):
        use_stream_gates: bool
        n_selfattn: int
        num_heads: int
        mlp_ratio: int
        num_heads_tech: int
        pma_num_heads: int
        pma_num_seeds: int
        general_num_heads: int
        general_mlp_ratio: int
        n_general_selfattn: int
        
        param_dtype: any = jnp.bfloat16
        activation: Literal["gelu", "silu"] = "gelu"
        use_bias_qkv: bool = False
        use_bias_out: bool = False
        attention_bias: any = None


        @nn.compact
        def __call__(
            self,
            z_map,
            z_gamestate,
            z_units_nonscatter,
            z_cs,
            z_my_city,
            z_tech,
            z_tech_mask,
            z_is_researching,
            z_science_reserves,
            z_free_techs,
            training,
        ):
            E = z_map.shape[-1]
            streams = [z_map, z_gamestate, z_units_nonscatter, z_cs, z_my_city]
            B = streams[0].shape[0]

            # Tag tokens by source stream so the critic can learn different mixing for each.
            # stream ids: 0=map, 1=game, 2=units, 3=cs, 4=cities
            sizes = [x.shape[1] for x in streams]
            type_ids_1d = jnp.concatenate([jnp.full((n,), i, dtype=jnp.int32) for i, n in enumerate(sizes)], axis=0)
            type_ids = jnp.broadcast_to(type_ids_1d[None, :], (B, type_ids_1d.shape[0]))

            tokens = jnp.concatenate(streams, axis=1)  # (B, L, E)
            tokens = tokens + nn.Embed(5, E, name="vf_type_id")(type_ids)  # (B, L, E)       


            if self.use_stream_gates:
                # Learn one scalar gate per stream (softplus to keep positive, init ~1.0)
                gates = self.param("stream_gates", nn.initializers.constant(0.0), (5, 1), self.param_dtype)
                gate_vals = jax.nn.softplus(gates)  # (5,1)
                tokens = tokens * gate_vals[type_ids]  # broadcast per token

            # A couple of set-style SAB layers (no RoPE).
            for i in range(self.n_selfattn):
                tokens = TransformerCrossLayer(
                    hidden_dim=E,
                    num_heads=self.num_heads,
                    rope_query="none",
                    rope_key="none",
                    attn_dropout=0.0,
                    mlp_dropout=0.0,
                    mlp_ratio=self.mlp_ratio,
                    param_dtype=self.param_dtype,
                    out_dtype=tokens.dtype,
                    activation=self.activation,
                    use_bias_qkv=self.use_bias_qkv,
                    use_bias_out=self.use_bias_out,
                    name=f"vf_sab_{i}",
                )(
                    query_tokens=tokens,
                    key_value_tokens=tokens,
                    attention_bias=self.attention_bias,
                    deterministic=not training,
                )

            # Pool to a compact set of seeds
            global_summary_seeds = PMA(
                hidden_dim=E,
                num_heads=self.pma_num_heads,
                num_seeds=self.pma_num_seeds,
                dropout_rate=0.0,
                param_dtype=self.param_dtype,
                out_dtype=tokens.dtype,
                name="vf_pma",
            )(tokens, deterministic=not training)  # (B, S_v, E)
            
            def _pool_membership_tokens(tokens, present_mask, *, num_seeds=2, heads=4, name="set", training=False):
                """
                tokens: (B, N, E)  e.g., z_tech or z_pols (absent has its own embedding)
                present_mask: (B, N) 1=present, 0=absent (binary)
                returns: (B, num_seeds, E)
                """
                E = tokens.shape[-1]
                pm = present_mask.astype(jnp.int32)

                # Learned scalar gate per state; init present≈0.99, absent≈0.5 (soft down-weight, not zero)
                def _gate_init(key, shape, dtype):
                    # rows: [absent, present]
                    p_absent  = 0.5
                    p_present = 0.99
                    eps = 1e-6  # avoid inf at exactly 0 or 1
                    def logit(p):
                        p = jnp.clip(p, eps, 1.0 - eps)
                        return jnp.log(p / (1.0 - p))
                    return jnp.array([[logit(p_absent)], [logit(p_present)]], dtype=dtype)

                raw_gate = nn.Embed(2, 1, name=f"{name}_gate", embedding_init=_gate_init)(pm)  # (B,N,1)
                gate = jax.nn.sigmoid(raw_gate)  # (B,N,1)

                z = tokens * gate  # scale, don't erase semantics

                # Optional light SAB so present items can interact once
                z = TransformerCrossLayer(
                    hidden_dim=E, num_heads=heads, rope_query="1d", rope_key="1d",
                    attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=4.0,
                    use_bias_qkv=False, use_bias_out=False, name=f"{name}_sab1",
                )(z, z, deterministic=not training)
                
                return z
            
            # These should carry some relative-position semantics!
            slot_ids = jnp.arange(z_tech.shape[1], dtype=jnp.int32)[None, :]
            z_tech = z_tech + nn.Embed(z_tech.shape[1], E, name="tech_slot_id")(slot_ids)

            z_tech = _pool_membership_tokens(
                z_tech, z_tech_mask,
                num_seeds=None, heads=self.num_heads_tech,
                name="tech_set", training=training
            )
            
            tech_tokens = jnp.concatenate([
                global_summary_seeds,
                z_science_reserves[:, None],
                z_free_techs[:, None],
                z_is_researching[:, None],
            ], axis=1)
            
            z_techs_ctx = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.general_num_heads,
                rope_query="1d",
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.general_mlp_ratio,
                param_dtype=self.param_dtype,
                out_dtype=tokens.dtype,
                activation=self.activation,
                use_bias_qkv=self.use_bias_qkv,
                use_bias_out=self.use_bias_out,
            )(
                query_tokens=z_tech, key_value_tokens=tech_tokens,
                attention_bias=self.attention_bias,
                deterministic=not training,
            )
            
            for _ in range(self.n_general_selfattn):
                z_techs_ctx = TransformerCrossLayer(
                    hidden_dim=E,
                    num_heads=self.general_num_heads,
                    rope_query="1d",
                    rope_key="1d",
                    attn_dropout=0.0,
                    mlp_dropout=0.0,
                    mlp_ratio=self.general_mlp_ratio,
                    param_dtype=self.param_dtype,
                    out_dtype=z_techs_ctx.dtype,
                    activation=self.activation,
                    use_bias_qkv=self.use_bias_qkv,
                    use_bias_out=self.use_bias_out,
                )(
                    query_tokens=z_techs_ctx, key_value_tokens=z_techs_ctx,
                    attention_bias=self.attention_bias,
                    deterministic=not training,
                )
            
            actions_techs = nn.Dense(1)(z_techs_ctx).squeeze(-1)

            return actions_techs


    class ActionHeadUnits(nn.Module):
        use_stream_gates: bool
        n_selfattn: int
        num_heads: int
        mlp_ratio: int
        pma_num_heads: int
        pma_num_seeds: int
        units_inner_num_heads: int
        units_inner_mlp_ratio: int
        units_players_num_heads: int
        units_players_mlp_ratio: int
        trade_yield_num_heads: int
        trade_yield_mlp_ratio: int
        trade_yield_pma_num_heads: int
        trade_yield_pma_num_seeds: int
        trade_units_num_heads: int
        trade_units_mlp_ratio: int
        my_units_num_heads: int
        my_units_mlp_ratio: int
        n_my_units_selfattn: int
        
        param_dtype: any = jnp.bfloat16
        activation: Literal["gelu", "silu"] = "gelu"
        use_bias_qkv: bool = False
        use_bias_out: bool = False
        attention_bias: any = None


        @nn.compact
        def __call__(
            self,
            z_map,
            z_gamestate,
            z_units_nonscatter,
            z_cs,
            z_my_city,
            z_units_nonscatter_raw,
            cb_tok,
            hp_tok,
            ap_tok,
            pos_tok_units,
            z_engaged_n_turns,
            z_action_cat,
            z_trade_to_player_int,
            z_trade_from_city_int,
            z_trade_to_city_int,
            _trade_to_player_int,
            _trade_from_city_int,
            z_trade_yields,
            z_culture_ypk,
            z_culture_hf_ypk,
            trade_yields_mask,
            is_caravan_mask,
            me_token,
            player_embedding_table,
            games_idx,
            me_idx,
            training,
        ):
            E = z_map.shape[-1]
            streams = [z_map, z_gamestate, z_units_nonscatter, z_cs, z_my_city]
            B = streams[0].shape[0]

            # Tag tokens by source stream so the critic can learn different mixing for each.
            # stream ids: 0=map, 1=game, 2=units, 3=cs, 4=cities
            sizes = [x.shape[1] for x in streams]
            type_ids_1d = jnp.concatenate([jnp.full((n,), i, dtype=jnp.int32) for i, n in enumerate(sizes)], axis=0)
            type_ids = jnp.broadcast_to(type_ids_1d[None, :], (B, type_ids_1d.shape[0]))

            tokens = jnp.concatenate(streams, axis=1)  # (B, L, E)
            tokens = tokens + nn.Embed(5, E, name="vf_type_id")(type_ids)  # (B, L, E)       


            if self.use_stream_gates:
                # Learn one scalar gate per stream (softplus to keep positive, init ~1.0)
                gates = self.param("stream_gates", nn.initializers.constant(0.0), (5, 1), self.param_dtype)
                gate_vals = jax.nn.softplus(gates)  # (5,1)
                tokens = tokens * gate_vals[type_ids]  # broadcast per token

            # A couple of set-style SAB layers (no RoPE).
            for i in range(self.n_selfattn):
                tokens = TransformerCrossLayer(
                    hidden_dim=E,
                    num_heads=self.num_heads,
                    rope_query="none",
                    rope_key="none",
                    attn_dropout=0.0,
                    mlp_dropout=0.0,
                    mlp_ratio=self.mlp_ratio,
                    param_dtype=self.param_dtype,
                    out_dtype=tokens.dtype,
                    activation=self.activation,
                    use_bias_qkv=self.use_bias_qkv,
                    use_bias_out=self.use_bias_out,
                    name=f"vf_sab_{i}",
                )(
                    query_tokens=tokens,
                    key_value_tokens=tokens,
                    attention_bias=self.attention_bias,
                    deterministic=not training,
                )

            # Pool to a compact set of seeds
            global_summary_seeds = PMA(
                hidden_dim=E,
                num_heads=self.pma_num_heads,
                num_seeds=self.pma_num_seeds,
                dropout_rate=0.0,
                param_dtype=self.param_dtype,
                out_dtype=tokens.dtype,
                name="vf_pma",
            )(tokens, deterministic=not training)  # (B, S_v, E)

            global_summary_seeds = jnp.concatenate([
                global_summary_seeds,
                z_culture_ypk,
                z_culture_hf_ypk,
            ], axis=1)
            
            # First step: mixing the units. These variables are over all units 
            # and have already been  modulated for "known" vs "unknown"
            unit_mix = (
                z_units_nonscatter_raw +
                cb_tok +
                hp_tok +
                ap_tok + 
                pos_tok_units
            )
            unit_mix = nn.LayerNorm(epsilon=1e-6, name="unit_mix_ln")(unit_mix)
            
            # ---- Step 2: within-player attention (set over 30 slots) ----
            B, P, S, E = unit_mix.shape  # (B,6,30,E)

            x = unit_mix.reshape(B * P, S, E)

            x = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.units_inner_num_heads,
                rope_query="1d",
                rope_key="1d",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.units_inner_mlp_ratio,
                use_bias_qkv=False,
                use_bias_out=False,
                name="units_inner_sab",
            )(x, x, deterministic=not training)  # (B*P, S, E)
            
            per_player = x.reshape(B, P, S, E)  # (B,6,S_in,E)
            
            # ---- Step 3: across players (fixed order) + me-aware anchor ----
            B, P, S_in, E = per_player.shape

            # Add player identity to each per-player seed using the passed-in table
            player_ids = jnp.arange(P, dtype=jnp.int32)[None, :, None]  # (1, P, 1)
            player_ids = jnp.broadcast_to(player_ids, (B, P, S_in))  # (B, P, S_in)
            per_player = per_player + player_embedding_table(player_ids)  # (B, P, S_in, E)

            # Me-aware anchor so “me vs others” is explicit

            # Me-aware anchor: apply ONLY to the current player's slice
            me_anchor = nn.Dense(E, name="units_me_anchor")(me_token)  # (B, E)
            me_anchor = me_anchor[:, None, None, :]  # (B, 1, 1, E)
            me_mask = jax.nn.one_hot(me_idx, P, dtype=per_player.dtype)  # (B, P)
            me_mask = me_mask[:, :, None, None]  # (B, P, 1, 1)
            per_player = per_player + me_anchor * me_mask  # only affects (b, me_idx[b], :, :)
            
            seq = per_player.reshape(B * P, S_in, E)

            seq = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.units_players_num_heads,
                rope_query="1d",
                rope_key="1d",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.units_players_mlp_ratio,
                use_bias_qkv=False,
                use_bias_out=False,
                name="units_players_sab",
            )(
                query_tokens=seq, key_value_tokens=seq,
                deterministic=not training,
            )
            
            # Now let's handle the trade route information. We have now two masks
            # to help us out with that. One for caravan and one for active trade 
            # routes. These two carry overlapping, but not identical, information
            # Tri-state: 0=non-caravan, 1=caravan idle, 2=caravan active
            caravan_state = jnp.where(
                ~is_caravan_mask, 0,
                jnp.where(trade_yields_mask, 2, 1)
            ).astype(jnp.int32)  # (B,30)

            state_tok = nn.Embed(3, E, name="caravan_state_id")(caravan_state)  # (B,30,E)

            # Direction tags (to/from) for endpoint embeddings
            endpoint_dir_tags = nn.Embed(2, E, name="trade_endpoint_dir_id")(
                jnp.array([0, 1], dtype=jnp.int32)
            )  # (2,E)

            # Compose endpoints
            z_to = z_trade_to_player_int + z_trade_to_city_int + endpoint_dir_tags[0][None, None, :]
            z_from = z_trade_from_city_int + endpoint_dir_tags[1][None, None, :]

            # Per-unit yield pooling: add tiny IDs for (dir, yield-type)
            # IDs: direction 0..1, yield 0..9
            dir_ids = jnp.arange(2, dtype=jnp.int32)[None, None, :, None]  # (1,1,2,1)
            yld_ids = jnp.arange(10, dtype=jnp.int32)[None, None, None, :]  # (1,1,1,10)

            dir_emb = nn.Embed(2, E, name="trade_yield_dir_id")(dir_ids)  # (1,1,2,1,E)
            yld_emb = nn.Embed(10, E, name="trade_yield_type_id")(yld_ids)  # (1,1,1,10,E)

            y = z_trade_yields + dir_emb + yld_emb  # (B,30,2,10,E)
            y = y.reshape(B * S, 2 * 10, E)  # (B*30, 20, E)

            y = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.trade_yield_num_heads,
                rope_query="1d",
                rope_key="1d",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.trade_yield_mlp_ratio,
                use_bias_qkv=False,
                use_bias_out=False,
                name="trade_yield_sab",
            )(y, y, deterministic=not training)
            
            y = PMA(
                hidden_dim=E,
                num_heads=self.trade_yield_pma_num_heads,
                num_seeds=self.trade_yield_pma_num_seeds, 
                dropout_rate=0.0,
                name="trade_yield_pma",
            )(y, deterministic=not training)  # (B*30, S_y, E)

            # collapse S_y seeds via mean if >1, then reshape to (B,30,E)
            y = y.mean(axis=1).reshape(B, S, E)  # (B,30,E)
            # Only contribute when active (keep state_tok to explain why missing)
            y = y * trade_yields_mask[..., None]
            
            # Per-unit trade subtoken (own units only)
            trade_unit = (
                state_tok +
                z_engaged_n_turns +
                z_action_cat +
                z_to + z_from +
                y
            )  # (B,30,E)

            trade_unit = nn.LayerNorm(epsilon=1e-6, name="trade_unit_ln")(trade_unit)
            
            trade_seq = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.trade_units_num_heads,
                rope_query="1d",
                rope_key="1d",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.trade_units_mlp_ratio,
                use_bias_qkv=False,
                use_bias_out=False,
                name="my_trade_units_sab",
            )(trade_unit, trade_unit, deterministic=not training)  # (B,30,E)

            # Let's now separate **my** units from all units
            all_units = seq.reshape(trade_seq.shape[0], 6, trade_seq.shape[1], -1)
            my_units = all_units[games_idx, me_idx]
            all_units = all_units.reshape(B, P * S, E)

            # And now update the belief in my units based on all of the information thusfar!
            my_units = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.my_units_num_heads,
                rope_query="1d",
                rope_key="1d",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.my_units_mlp_ratio,
                use_bias_qkv=False,
                use_bias_out=False,
                name="my_units_all_units_xab",
            )(my_units, all_units, deterministic=not training)  # (B,30,E)
            
            my_units = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.my_units_num_heads,
                rope_query="1d",
                rope_key="1d",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.my_units_mlp_ratio,
                use_bias_qkv=False,
                use_bias_out=False,
                name="my_units_trade_seq_xab",
            )(my_units, trade_seq, deterministic=not training)  # (B,30,E)
            
            my_units = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.my_units_num_heads,
                rope_query="1d",
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.my_units_mlp_ratio,
                use_bias_qkv=False,
                use_bias_out=False,
                name="my_units_summary_xab",
            )(my_units, global_summary_seeds, deterministic=not training)  # (B,30,E)

            for _ in range(self.n_my_units_selfattn):
                my_units = TransformerCrossLayer(
                    hidden_dim=E,
                    num_heads=self.my_units_num_heads,
                    rope_query="1d",
                    rope_key="1d",
                    attn_dropout=0.0,
                    mlp_dropout=0.0,
                    mlp_ratio=self.my_units_mlp_ratio,
                    use_bias_qkv=False,
                    use_bias_out=False,
                    name=f"my_units_summary_{_}",
                )(my_units, my_units, deterministic=not training)  # (B,30,E)

            actions_my_units = nn.Dense(len(ALL_ACTION_FUNCTIONS))(my_units)
            actions_map = nn.Dense(2772)(my_units)

            return (actions_my_units, actions_map)

    class ActionHeadCities(nn.Module):
        use_stream_gates: bool
        n_selfattn: int
        num_heads: int
        mlp_ratio: int
        pma_num_heads: int
        pma_num_seeds: int
        pooling_num_heads: int
        pooling_pma_seeds: int
        pooling_mlp_ratio: int
        fuse_num_heads: int
        fuse_mlp_ratio: int
        fuse_pma_num_heads: int
        fuse_pma_num_seeds: int
        
        param_dtype: any = jnp.bfloat16
        activation: Literal["gelu", "silu"] = "gelu"
        use_bias_qkv: bool = False
        use_bias_out: bool = False
        attention_bias: any = None


        @nn.compact
        def __call__(
            self,
            z_map,
            z_gamestate,
            z_units_nonscatter,
            z_cs,
            z_my_city,
            z_city_ids,
            z_city_rowcols,
            z_city_yields,
            z_city_center_yields,
            z_building_yields,
            z_culture_building_yields,
            z_religion_building_yields,
            z_city_population,
            z_city_worked_slots,
            z_city_specialists,
            z_city_gws,
            z_city_food_reserves,
            z_city_growth_carryover,
            z_city_prod_reserves,
            z_city_prod_carryover,
            z_city_constructing,
            z_city_building_maintenance,
            z_city_defense,
            z_city_hp,
            z_city_my_buildings,
            z_city_my_resources,
            z_city_is_coastal,
            z_city_culture_reserves_for_border,
            z_city_gpps,
            z_city_religious_population,
            z_my_city_tenets,
            _ownership_map,
            games_idx,
            me_idx,
            training,
        ):

            E = z_map.shape[-1]
            streams = [z_map, z_gamestate, z_units_nonscatter, z_cs, z_my_city]
            B = streams[0].shape[0]

            # Tag tokens by source stream so the critic can learn different mixing for each.
            # stream ids: 0=map, 1=game, 2=units, 3=cs, 4=cities
            sizes = [x.shape[1] for x in streams]
            type_ids_1d = jnp.concatenate([jnp.full((n,), i, dtype=jnp.int32) for i, n in enumerate(sizes)], axis=0)
            type_ids = jnp.broadcast_to(type_ids_1d[None, :], (B, type_ids_1d.shape[0]))

            tokens = jnp.concatenate(streams, axis=1)  # (B, L, E)
            tokens = tokens + nn.Embed(5, E, name="vf_type_id")(type_ids)  # (B, L, E)       


            if self.use_stream_gates:
                # Learn one scalar gate per stream (softplus to keep positive, init ~1.0)
                gates = self.param("stream_gates", nn.initializers.constant(0.0), (5, 1), self.param_dtype)
                gate_vals = jax.nn.softplus(gates)  # (5,1)
                tokens = tokens * gate_vals[type_ids]  # broadcast per token

            # A couple of set-style SAB layers (no RoPE).
            for i in range(self.n_selfattn):
                tokens = TransformerCrossLayer(
                    hidden_dim=E,
                    num_heads=self.num_heads,
                    rope_query="none",
                    rope_key="none",
                    attn_dropout=0.0,
                    mlp_dropout=0.0,
                    mlp_ratio=self.mlp_ratio,
                    param_dtype=self.param_dtype,
                    out_dtype=tokens.dtype,
                    activation=self.activation,
                    use_bias_qkv=self.use_bias_qkv,
                    use_bias_out=self.use_bias_out,
                    name=f"vf_sab_{i}",
                )(
                    query_tokens=tokens,
                    key_value_tokens=tokens,
                    attention_bias=self.attention_bias,
                    deterministic=not training,
                )

            # Pool to a compact set of seeds
            global_summary_seeds = PMA(
                hidden_dim=E,
                num_heads=self.pma_num_heads,
                num_seeds=self.pma_num_seeds,
                dropout_rate=0.0,
                param_dtype=self.param_dtype,
                out_dtype=tokens.dtype,
                name="vf_pma",
            )(tokens, deterministic=not training)  # (B, S_v, E)
            
            # base city token: sum single-token features + learned tag, then LN → (B, C, 1, E)
            E = z_city_ids.shape[-1]
            B, C = z_city_ids.shape[:2]

            base = (
                z_city_ids
                + z_city_population
                + z_city_food_reserves
                + z_city_growth_carryover
                + z_city_prod_reserves
                + z_city_prod_carryover
                + z_city_building_maintenance
                + z_city_defense
                + z_city_hp
                + z_city_is_coastal
                + z_city_culture_reserves_for_border
            )

            base = nn.LayerNorm(epsilon=1e-6, name="city_base_ln")(base)  # (B, C, E)
            base = base[:, :, None, :]  # (B, C, 1, E)
            
            def pool_city_set(x, *, name: str, seeds: int = 1, heads: int = 4, mlp_ratio: int = 4, training: bool = False):
                B, C = x.shape[:2]
                E = x.shape[-1]
                y = x.reshape(B * C, -1, E)
                y = TransformerCrossLayer(
                    hidden_dim=E, num_heads=heads,
                    rope_query="none", rope_key="none",
                    attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=mlp_ratio,
                    use_bias_qkv=False, use_bias_out=False, name=f"{name}_sab",
                )(y, y, deterministic=not training)
                y = PMA(
                    hidden_dim=E, num_heads=heads,
                    num_seeds=seeds, dropout_rate=0.0, name=f"{name}_pma",
                )(y, deterministic=not training)  # (B*C, seeds, E)
                return y.reshape(B, C, seeds, E)
            
            rowcols_seed = pool_city_set(
                z_city_rowcols, name="city_rowcols", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            center_yields_seed = pool_city_set(
                z_city_center_yields, name="city_center_yields", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            yields_seed = pool_city_set(
                z_city_yields, name="city_yields", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            bldg_yields_seed = pool_city_set(
                z_building_yields, name="city_bldg_yields", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            cult_bldg_yields_seed = pool_city_set(
                z_culture_building_yields, name="city_cult_bldg_yields", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            relig_bldg_yields_seed = pool_city_set(
                z_religion_building_yields, name="city_relig_bldg_yields", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            worked_slots_seed = pool_city_set(
                z_city_worked_slots, name="city_worked_slots", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            specialists_seed = pool_city_set(
                z_city_specialists, name="city_specialists", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            gws_seed = pool_city_set(
                z_city_gws, name="city_gws", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            gpps_seed = pool_city_set(
                z_city_gpps, name="city_gpps", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            relpop_seed = pool_city_set(
                z_city_religious_population, name="city_relpop", seeds=self.pooling_pma_seeds,
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            tenets_seed = pool_city_set(
                z_my_city_tenets, name="city_tenets", seeds=self.pooling_pma_seeds,
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            my_bldgs_seed = pool_city_set(
                z_city_my_buildings, name="city_my_bldgs", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            my_resources_seed = pool_city_set(
                z_city_my_resources, name="city_my_resources", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            city_constructing = pool_city_set(
                z_city_constructing, name="city_constructing", seeds=self.pooling_pma_seeds, 
                heads=self.pooling_num_heads, mlp_ratio=self.pooling_mlp_ratio, training=training
            )
            
            per_city = jnp.concatenate([
                base,
                rowcols_seed,
                center_yields_seed,
                yields_seed,
                bldg_yields_seed,
                cult_bldg_yields_seed,
                relig_bldg_yields_seed,
                worked_slots_seed,
                specialists_seed,
                gws_seed,
                gpps_seed,
                relpop_seed,
                tenets_seed,
                my_bldgs_seed,
                my_resources_seed,
                city_constructing
            ], axis=2)  # (B, C, S_total, E)

            # per_city: (B, C, S_total, E) from Step 2
            B, C, S_total, E = per_city.shape

            # 3a) fuse within each city → a few seeds per city
            y = per_city.reshape(B * C, S_total, E)
            y = TransformerCrossLayer(
                hidden_dim=E, num_heads=self.fuse_num_heads,
                rope_query="none", rope_key="none",
                attn_dropout=0.0, mlp_dropout=0.0, mlp_ratio=self.fuse_mlp_ratio,
                use_bias_qkv=False, use_bias_out=False, name="city_inner_fuse_sab",
            )(y, y, deterministic=not training)

            # choose 1–2 seeds per city; 2 gives a little extra capacity
            city_seeds = PMA(
                hidden_dim=E, num_heads=self.fuse_pma_num_heads,
                num_seeds=self.fuse_pma_num_seeds, dropout_rate=0.0, name="city_inner_fuse_pma",
            )(y, deterministic=not training)  # (B*C, 2, E)
            city_seeds = city_seeds.reshape(B, C, self.fuse_pma_num_seeds, E)

            # 3b) tag city slot identity so slots are distinguishable
            # (B, max_num_cities, pma_seeds, E)
            city_ids = jnp.arange(C, dtype=jnp.int32)[None, :, None]
            city_ids = jnp.broadcast_to(city_ids, (B, C, self.fuse_pma_num_seeds))
            city_seeds = city_seeds + nn.Embed(C, E, name="city_slot_id")(city_ids)

            # We ultimately need z_city_worked_slots, z_city_my_buildings
            my_ownership = _ownership_map[games_idx, me_idx] >= 2

            B, C, H, W = my_ownership.shape
            E = per_city.shape[-1]

            # Learned 2D coordinate embeddings (no map features needed)
            E_row = E // 2
            E_col = E - E_row

            row_ids = jnp.arange(H, dtype=jnp.int32)  # (H,)
            col_ids = jnp.arange(W, dtype=jnp.int32)  # (W,)

            row_emb = nn.Embed(H, E_row, name="own_row_emb")(row_ids)  # (H, E_row)
            col_emb = nn.Embed(W, E_col, name="own_col_emb")(col_ids)  # (W, E_col)

            # Build (H, W, E) position field by concatenating row/col halves
            row_field = jnp.broadcast_to(row_emb[:, None, :], (H, W, E_row))  # (H, W, E_row)
            col_field = jnp.broadcast_to(col_emb[None, :, :], (H, W, E_col))  # (H, W, E_col)
            pos_hw  = jnp.concatenate([row_field, col_field], axis=-1)  # (H, W, E)

            # Broadcast to (B, C, H, W, E) and mask
            pos_bc = jnp.broadcast_to(pos_hw[None, None, ...], (B, C, H, W, E))  # (B, C, H, W, E)
            mask = my_ownership[..., None].astype(pos_bc.dtype)  # (B, C, H, W, 1)

            owned_sum = (mask * pos_bc).sum(axis=(2, 3))  # (B, C, E)
            owned_cnt = mask.sum(axis=(2, 3)).clip(min=1e-6)  # (B, C, 1)
            owned_mean = owned_sum / owned_cnt  # (B, C, E)

            # Optional: tag “has any tiles” vs padded city
            #has_tiles = (owned_cnt[..., 0] > 0).astype(jnp.int32)                       # (B, C)
            #owned_mean = owned_mean + nn.Embed(2, E, name="own_has_tiles_id")(has_tiles)  # (B, C, E)

            ownership_seed = owned_mean[:, :, None, :]                                   # (B, C, 1, E)

            # Worked slots (B, max_num_cities, 36, E)
            # X: global, city seeds, ownership_seed
            B, C, S, E = z_city_worked_slots.shape
            city_ids = jnp.arange(C, dtype=jnp.int32)[None, :, None]  # (1,C,1)
            slot_ids = jnp.arange(S, dtype=jnp.int32)[None, None, :]  # (1,1,S)
            
            worked_city_id = nn.Embed(C, E, name="worked_city_id")(city_ids)  # (B,C,S,E) via broadcast
            worked_slot_id = nn.Embed(S, E, name="worked_slot_id")(slot_ids)  # (B,C,S,E) via broadcast
            z_city_worked_slots = nn.LayerNorm(1e-6)(
                z_city_worked_slots
                + worked_city_id 
                + worked_slot_id
            )

            z_city_worked_slots = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.num_heads,
                rope_query="none",
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.mlp_ratio,
                param_dtype=self.param_dtype,
                out_dtype=tokens.dtype,
                activation=self.activation,
                use_bias_qkv=self.use_bias_qkv,
                use_bias_out=self.use_bias_out,
                name="worked_slots_global_xab",
            )(
                query_tokens=z_city_worked_slots.reshape(B, C * S, E),
                key_value_tokens=global_summary_seeds,
                attention_bias=self.attention_bias,
                deterministic=not training,
            )
            z_city_worked_slots = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.num_heads,
                rope_query="none",
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.mlp_ratio,
                param_dtype=self.param_dtype,
                out_dtype=tokens.dtype,
                activation=self.activation,
                use_bias_qkv=self.use_bias_qkv,
                use_bias_out=self.use_bias_out,
                name="worked_slots_city_xab",
            )(
                query_tokens=z_city_worked_slots,
                key_value_tokens=nn.LayerNorm(1e-6)(city_seeds + worked_city_id).reshape(B, -1, E),
                attention_bias=self.attention_bias,
                deterministic=not training,
            )
            z_city_worked_slots = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.num_heads,
                rope_query="none",
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.mlp_ratio,
                param_dtype=self.param_dtype,
                out_dtype=tokens.dtype,
                activation=self.activation,
                use_bias_qkv=self.use_bias_qkv,
                use_bias_out=self.use_bias_out,
                name="worked_slots_ownership_xab",
            )(
                query_tokens=z_city_worked_slots,
                key_value_tokens=nn.LayerNorm(1e-6)(ownership_seed + worked_city_id).reshape(B, -1, E),
                attention_bias=self.attention_bias,
                deterministic=not training,
            )
            z_city_worked_slots = z_city_worked_slots.reshape(B, C, S, E)
            actions_city_worked_slots = nn.Dense(36)(z_city_worked_slots)

            # Building (B, max_num_cities, 148, E)
            # X; global, city seeds
            B, C, Bl, E = z_city_my_buildings.shape
            city_ids = jnp.arange(C, dtype=jnp.int32)[None, :, None]  # (1,C,1)
            bl_ids = jnp.arange(Bl, dtype=jnp.int32)[None, None, :]  # (1,1,Bl)

            bldg_city_id = nn.Embed(C, E, name="bldg_city_id")(city_ids)  # (B,C,Bl,E) via broadcast
            bldg_slot_id = nn.Embed(Bl, E, name="bldg_slot_id")(bl_ids)  # (B,C,Bl,E) via broadcast
            z_city_my_buildings = nn.LayerNorm(1e-6)(
                z_city_my_buildings
                + bldg_city_id 
                + bldg_slot_id
            )

            z_city_my_buildings = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.num_heads,
                rope_query="none",
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.mlp_ratio,
                param_dtype=self.param_dtype,
                out_dtype=tokens.dtype,
                activation=self.activation,
                use_bias_qkv=self.use_bias_qkv,
                use_bias_out=self.use_bias_out,
                name="my_bldgs_global_xab",
            )(
                query_tokens=z_city_my_buildings.reshape(B, C * Bl, E),
                key_value_tokens=global_summary_seeds,
                attention_bias=self.attention_bias,
                deterministic=not training,
            )
            z_city_my_buildings = TransformerCrossLayer(
                hidden_dim=E,
                num_heads=self.num_heads,
                rope_query="none",
                rope_key="none",
                attn_dropout=0.0,
                mlp_dropout=0.0,
                mlp_ratio=self.mlp_ratio,
                param_dtype=self.param_dtype,
                out_dtype=tokens.dtype,
                activation=self.activation,
                use_bias_qkv=self.use_bias_qkv,
                use_bias_out=self.use_bias_out,
                name="my_buildings_city_xab",
            )(
                query_tokens=z_city_my_buildings,
                key_value_tokens=nn.LayerNorm(1e-6)(city_seeds + bldg_city_id).reshape(B, -1, E),
                attention_bias=self.attention_bias,
                deterministic=not training,
            )
            z_city_my_buildings = z_city_my_buildings.reshape(B, C, Bl, E)
            actions_city_my_buildings = nn.Dense(1)(z_city_my_buildings)

            return (actions_city_worked_slots, actions_city_my_buildings.squeeze(-1))


    class TerraNovaModel(nn.Module):
        """
        The expectation is that this module is only ever `call`ed from a per-device perspective. Ergo,
        all functions containing calls to this model should be under a shard_map'ed context
        """
        embedding_dim: int = 64

        @nn.compact
        def __call__(self, observation, training):
            def _make_embeddings(num_embeddings, features, _dtype, name, initializer, indexing_data, return_table=False):
                """
                NOTE: ensure that `num_embeddings` is the number of actual feature integers + 1. This is because
                "unknown" or "not available" information (i.e., -1) will be mapped to it's own row in the table.

                Automatic construction of embedding table. 
                """
                indexing_data = jnp.where(indexing_data == -1, num_embeddings - 1, indexing_data)
                table = nn.Embed(
                    num_embeddings=num_embeddings, features=features, 
                    dtype=_dtype, param_dtype=_dtype,
                    embedding_init=INITIALIZER_LOOKUP[initializer], name=name
                )
                z = table(indexing_data.astype(jnp.int32))

                if return_table:
                    return z, table
                else:
                    return z
            
            # First, some useful arrays for indexing
            games_idx = jnp.arange(observation.player_id.shape[0])
            

            ### Embedding Tables ###
            def embed_player_or_cs(player_table: nn.Embed, cs_table: nn.Embed, ids: jnp.ndarray) -> jnp.ndarray:
                """
                Look up embeddings where:
                  - ids in [0 .. P-1] index `player_table`
                  - ids in [P .. P+C-1] index `cs_table` (with offset P)
                Assumes there is no unknown; returns shape ids.shape + (E,).
                """
                # Shapes: players (P, E), city-states (C, E)
                players = player_table.embedding
                citystates = cs_table.embedding
                # sanity: same feature dim
                assert players.shape[1] == citystates.shape[1], "feature dim mismatch between tables"

                combined = jnp.concatenate([players, citystates], axis=0)  # (P+C, E)
                return combined[ids.astype(jnp.int32)]

            # (0) Learnable player tokens
            player_embedding_table = nn.Embed(
                num_embeddings=6, features=self.embedding_dim,
                dtype=embedding_dtype, param_dtype=embedding_dtype,
                embedding_init=INITIALIZER_LOOKUP[embedding_init],
                name="player_id_embeddings"
            )
            cs_embedding_table = nn.Embed(
                num_embeddings=12, features=self.embedding_dim,
                dtype=embedding_dtype, param_dtype=embedding_dtype,
                embedding_init=INITIALIZER_LOOKUP[embedding_init],
                name="cs_id_embeddings"
            )
            city_int_table = nn.Embed(
                num_embeddings=10, features=self.embedding_dim,
                dtype=embedding_dtype, param_dtype=embedding_dtype,
                embedding_init=INITIALIZER_LOOKUP[embedding_init],
                name="city_int_embeddings"
            )
            resources_table = nn.Embed(
                num_embeddings=len(ALL_RESOURCES), features=self.embedding_dim,
                dtype=embedding_dtype, param_dtype=embedding_dtype,
                embedding_init=INITIALIZER_LOOKUP[embedding_init],
                name="resources_embeddings"
            )

            # (6, 64), (100, 64)
            z_all_players = player_embedding_table(jnp.arange(6).astype(jnp.int32))[None, :, None, :]  # (1,6,1,z_dim)


            # (2) Units
            # Units are actually coming in +1. -1 = don't know, 0 = I don't have a unit in this slot, 1 = actual unit...
            z_units, units_embedding_table = _make_embeddings(
                num_embeddings=len(GameUnits) + 2, features=self.embedding_dim, _dtype=embedding_dtype,
                name="unit_embeddings", initializer=embedding_init, indexing_data=observation.units.unit_type,
                return_table=True
            )

            # Map-related embeddings
            # (1) Elevation [sea, flat, hill, mountain]
            z_elevation = _make_embeddings(
                num_embeddings=5, features=self.embedding_dim, _dtype=embedding_dtype, name="elevation_embeddings",
                initializer=embedding_init, indexing_data=observation.elevation_map
            )
            
            # (2) Terrain [ocean, grass, plains, desert, tundra, snow]
            z_terrain = _make_embeddings(
                num_embeddings=7, features=self.embedding_dim, _dtype=embedding_dtype, name="terrain_embeddings",
                initializer=embedding_init, indexing_data=observation.terrain_map
            )
            
            # (3) Feature [forest, jungle, marsh, oasis, floodplains, ice]
            z_features = _make_embeddings(
                num_embeddings=7, features=self.embedding_dim, _dtype=embedding_dtype, name="feature_embeddings",
                initializer=embedding_init, indexing_data=observation.feature_map
            )
            
            # (4) Natural Wonder
            z_nw = _make_embeddings(
                num_embeddings=len(ALL_NATURAL_WONDERS) + 1, features=self.embedding_dim, _dtype=embedding_dtype, name="nw_embeddings",
                initializer=embedding_init, indexing_data=observation.nw_map
            )
            
            # (5) Resource
            z_resource = _make_embeddings(
                num_embeddings=len(ALL_RESOURCES) + 1, features=self.embedding_dim, _dtype=embedding_dtype, name="resource_embeddings",
                initializer=embedding_init, indexing_data=observation.visible_resources_map_players
            )
            
            # (6) Improvement
            z_improvement = _make_embeddings(
                num_embeddings=len(Improvements) + 1, features=self.embedding_dim, _dtype=embedding_dtype, name="improvement_embeddings",
                initializer=embedding_init, indexing_data=observation.improvement_map
            )
            
            # (7) Road
            z_road = _make_embeddings(
                num_embeddings=3, features=self.embedding_dim, _dtype=embedding_dtype, name="road_embeddings",
                initializer=embedding_init, indexing_data=observation.road_map
            )

            # (8) Edge rivers
            z_edge_river = _make_embeddings(
                num_embeddings=3, features=self.embedding_dim, _dtype=embedding_dtype, name="edge_river_embeddings",
                initializer=embedding_init, indexing_data=observation.edge_river_map
            )

            # (9) CS ownership
            z_cs_ownership = _make_embeddings(
                num_embeddings=13, features=self.embedding_dim, _dtype=embedding_dtype, name="cs_ownership_embeddings",
                initializer=embedding_init, indexing_data=observation.cs_ownership_map
            )

            # (10) Player ownership (deferred to later?)
            # [unowned, could own, does own, city center]

            ### Other map-related information (non-categorical)
            yield_map = observation.yield_map_players
            improvement_yield_map = observation.improvement_additional_yield_map
            visibility_map = observation.visibility_map
            culture_yield_map = observation.culture_info.additional_yield_map
            city_yield_map = observation.player_cities.additional_yield_map  # (B, cities)
            religion_yield_map = observation.player_cities.religion_info.additional_yield_map  # (B, cities)

            ### Patchifying all map elements ###
            # First, the elements using learned  embeddings
            # (B, 42, 66, embedding_dim) -> (B, T, patch_dim)
            z_elevation = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(z_elevation)

            z_terrain = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(z_terrain)
            
            z_features = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(z_features)
            
            z_nw = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(z_nw)

            z_resource = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(z_resource)
            
            z_improvement = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(z_improvement)

            z_road = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(z_road)
            
            z_edge_river = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(fuse_dir6_into_channels(z_edge_river))

            z_cs_ownership = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(z_cs_ownership)
            
            # (10) Player ownership map -- deferred to here!
            # This one is a bit weird. We need to keep two bits of info entact:
            # (1) Who has what ownership status over each tile
            # (2) Which city from player (1) retains that status
            own_code = observation.player_cities.ownership_map  # (B,6,10,H,W) int
            owns_mask = (own_code >= 2).astype(jnp.float32)  # (B,6,10,H,W)
            center_mask = (own_code == 3).astype(jnp.float32)  # (B,6,10,H,W)
            could_mask = (own_code == 1).astype(jnp.float32)  # (B,6,10,H,W)
            
            # (1) Who? (B,42,66,z_dim) 
            own_p = owns_mask.max(axis=2)  # (B,6,H,W)
            tile_player_emb = jnp.einsum('bphw,pd->bhwd', own_p, player_embedding_table.embedding)

            # (2) Which city? 
            # Most of this information is in the form (6, max_num_cities), so we want to "scatter" these into 
            # the game map. We also need to be careful to only scatter in for cities we know about...
            mask_pos_known = (observation.player_cities.city_rowcols[..., 0] != -1) & (observation.player_cities.city_rowcols[..., 1] != -1)
            mask_pop_known = (observation.player_cities.population != -1)
            mask_def_known = (observation.player_cities.defense != -1.0)
            mask_hp_known = (observation.player_cities.hp != -1.0)
            
            # The two categorical pieces of information
            # (a) city kind: 0=no city, 1=capital, 2=non-capital, -1 unknown
            city_kind_emb, city_kind_embedding_table = _make_embeddings(
                num_embeddings=4,  # 3 real (0..2) + 1 unknown
                features=self.embedding_dim, _dtype=embedding_dtype, name="city_kind_embeddings",
                initializer=embedding_init, indexing_data=observation.player_cities.city_ids,
                return_table=True
            )  # (B,6,max_num_cities,z_dim)

            # (f) is_coastal: 0/1, -1 unknown → 2 real + 1 unknown
            coastal_emb = _make_embeddings(
                num_embeddings=3, features=self.embedding_dim, _dtype=embedding_dtype, name="is_coastal_embeddings",
                initializer=embedding_init, indexing_data=observation.player_cities.is_coastal
            )  # (B,6,Cmax,E)
            
            # Now the continuous pieces of information. For these, we scale them, project them into the 
            # appropriate space, and use a learned known/unknown bias (adds to the projection)
            pop_tok = gate_cont(
                observation.player_cities.population, mask_pop_known, scale=50.0, 
                feature_dim=self.embedding_dim, name="population"
            )
            def_tok = gate_cont(
                observation.player_cities.defense, mask_def_known, scale=6000.0, 
                feature_dim=self.embedding_dim, name="defense"
            )
            hp_tok = gate_cont(
                observation.player_cities.hp, mask_hp_known, scale=2.0, 
                feature_dim=self.embedding_dim, name="hp"
            )

            # pos: (B,6,max_num_cities,2) with -1 unknown; gate it
            pos_known = mask_pos_known.astype(jnp.float32)[..., None]  # (B,6,max_num_cities,1)
            pos_xy = jnp.clip(observation.player_cities.city_rowcols.astype(jnp.float32), 0, 1e9)  # safe
            pos_xy = pos_xy / jnp.array([42.0, 66.0], dtype=pos_xy.dtype)  # normalize rows/cols to [0,1]
            pos_xy = pos_xy * pos_known

            pos_tok = (
                nn.Dense(self.embedding_dim, name="city_pos_proj", use_bias=False)(pos_xy) 
                + nn.Embed(2, self.embedding_dim, name="pos_known_emb", 
                           dtype=embedding_dtype, param_dtype=embedding_dtype)(mask_pos_known.astype(jnp.int32))
            )

            city_tok = (
                city_kind_emb
                + coastal_emb
                + pop_tok + def_tok + hp_tok
                + z_all_players
                + pos_tok
            )  # (B,6,max_num_cities,z_dim)
            
            # (B,H,W,dim_c)
            z_city_owns = scatter_cities(owns_mask, city_tok)
            z_city_couldown = scatter_cities(could_mask, city_tok)
            z_cities = nn.Conv(city_tok.shape[-1], kernel_size=(1,1), use_bias=True, name="city_fuse")(
                    jnp.concatenate([tile_player_emb, z_city_owns, z_city_couldown], -1)
            )
            if norm_scatters:
                z_cities = nn.LayerNorm(epsilon=1e-6, name="city_fuse_ln")(z_cities)
            
            z_cities = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(z_cities)


            # City-state locations → per-tile feature map
            cs_rowcol = observation.citystate_info.city_rowcols.astype(jnp.int32)  # (B, 12, 2)
            cs_known = ((cs_rowcol[..., 0] >= 0) & (cs_rowcol[..., 1] >= 0)).astype(jnp.int32)  # (B, 12)

            # Position token (normalize to [0,1], gate by known, add known/unknown bias)
            cs_pos_xy = jnp.clip(cs_rowcol.astype(jnp.float32), 0.0, 1e9)
            cs_pos_xy = cs_pos_xy / jnp.array([42.0, 66.0], dtype=cs_pos_xy.dtype)
            cs_pos_xy = cs_pos_xy * cs_known[..., None].astype(jnp.float32)  # (B, 12, 2)

            cs_pos_tok = (
                nn.Dense(self.embedding_dim, name="cs_pos_proj", use_bias=False)(cs_pos_xy)
                + nn.Embed(2, self.embedding_dim, name="cs_pos_known_emb",
                           dtype=embedding_dtype, param_dtype=embedding_dtype)(cs_known)
            )  # (B, 12, E)

            # Add CS identity so tiles know *which* CS is at that position
            cs_ids = jnp.arange(12, dtype=jnp.int32)[None, :]  # (1, 12)
            cs_id_tok = cs_embedding_table(cs_ids)  # (1, 12, E)
            cs_id_tok = jnp.broadcast_to(cs_id_tok, cs_pos_tok.shape)  # (B, 12, E)
            cs_tok = cs_pos_tok + cs_id_tok  # (B, 12, E)

            # Scatter into a per-tile map using existing scatter (reuse unit scatter with singleton axes)
            z_cs_centers_map = scatter_units(
                unit_tokens=cs_tok[:, None, :, :],          # (B, 1, 12, E)
                unit_rowcol=cs_rowcol[:, None, :, :],       # (B, 1, 12, 2)
                unit_exists_mask=(cs_known[:, None, :].astype(bool)),  # (B, 1, 12)
                map_hw=(42, 66),
                mean_normalize=False,
            )  # -> (B, 42, 66, E)

            if norm_scatters:
                z_cs_centers_map = nn.LayerNorm(epsilon=1e-6, name="cs_centers_ln")(z_cs_centers_map)

            # Patchify to join the map token stream
            z_cs_centers_map = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings,
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(z_cs_centers_map)
            
            # Next the map information not with learned embeddings
            z_yield_map = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(yield_map)

            z_improvement_yield_map = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(improvement_yield_map)
            
            # Does not have a native "channel" dimension, so adding singleton
            z_visibility_map = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(visibility_map[..., None])

            z_culture_yield_map = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(culture_yield_map)
            
            z_city_yield_map = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(city_yield_map)
            
            # (B, max_num_cities, 42, 66, 7) -> (B, 42, 66, 8 * max_num_cities)
            # (B, max_num_cities)
            # Here, we likely need some information exposed to the model on whether the
            # city exists. This is a slightly different type of "unknown" than the 
            # "I have not seen this tile yet" type of unknown.
            x = jnp.transpose(religion_yield_map, (0, 2, 3, 1, 4))   # (B, 42, 66, max_num_cities, 7)
            city_exists = observation.player_cities.city_ids > 0
            city_exists = city_exists[jnp.arange(city_exists.shape[0], dtype=jnp.int32),
                                      observation.player_id.astype(jnp.int32), :]
            exists = jnp.broadcast_to(city_exists[:, None, None, :, None], x.shape[:-1] + (1,))
            x = jnp.concatenate([x, exists.astype(jnp.float32)], axis=-1)
            z_religion_yield_map = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * x.shape[4])  # (B, 42, 66, max_num_cities*8)
            z_religion_yield_map = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(z_religion_yield_map)
            
            # Scattering the units onto the map
            pos_known = ((observation.units.unit_rowcol[..., 0] != -1) &
                         (observation.units.unit_rowcol[..., 1] != -1))
            type_known = (observation.units.unit_type != -1)
            cb_known = (observation.units.combat_bonus_accel != -1)
            hp_known = (observation.units.health != -1)
            ap_known = (observation.units.unit_ap != -1)


            rc = observation.units.unit_rowcol.astype(jnp.float32)
            rc = jnp.maximum(rc, 0.0) 
            rc = rc / jnp.array([42.0, 66.0]) 
            rc = rc * pos_known.astype(jnp.float32)[..., None]
            pos_tok_units = (
                nn.Dense(self.embedding_dim, name="unit_pos_proj", use_bias=False)(rc)
                + nn.Embed(2, self.embedding_dim, name="unit_pos_known_emb",
                           dtype=embedding_dtype, param_dtype=embedding_dtype)(pos_known.astype(jnp.int32))
            )

            cb_tok = gate_cont(observation.units.combat_bonus_accel, cb_known, 3.0, self.embedding_dim, "unit_cb")
            hp_tok = gate_cont(observation.units.health, hp_known, 1.0, self.embedding_dim, "unit_hp")
            ap_tok = gate_cont(observation.units.unit_ap, ap_known, 10.0, self.embedding_dim, "unit_ap")
            
            # (B,6,30,z_dim)
            unit_tok = (z_units + z_all_players + pos_tok_units + cb_tok + hp_tok + ap_tok) * type_known[..., None]

            # scatter to per-tile embedding
            # This scattering loses infomation w.r.t. "these are all my unit slots, but I don't have units in slot xyz..."
            # due to the mask  unit_type > 0. So we need a 2nd pathway to ensuer we capture this infortmation
            z_units = scatter_units(
                unit_tokens=unit_tok,
                unit_rowcol=observation.units.unit_rowcol.astype(jnp.int32),
                unit_exists_mask=observation.units.unit_type > 0,
                map_hw=(42, 66),
                mean_normalize=True,
            )  # (B,H,W,z_dim)
            if norm_scatters:
                z_units = nn.LayerNorm(epsilon=1e-6, name="units_fuse_ln")(z_units)
            
            z_units = PatchEmbed(
                mode=patch_embedding_mode, dim=patch_dim, patch=patch_size, add_pos=patch_position_embeddings, 
                param_dtype=patch_proj_dtype, out_dtype=patch_dtype
            )(z_units)
            ### Producing the map embedding ###
            # Catalog of all information:
            # Generic map information: z_elevation, z_terrain, z_features, z_nw, z_resource, z_improvement, z_road, 
            # z_edge_river, z_cs_ownership 

            # "Can see" information about all players: z_cities, z_units 

            # 

            twod_rope_nhnw = (42 // patch_size[0], 66 // patch_size[1])

            z_map = MapEmbedder(
                n_map_selfattn=me_n_map_selfattn,
                attn_num_heads=me_attn_num_heads,
                rope_query=me_rope_query,
                rope_key=me_rope_key,
                rope_base=me_rope_base,
                attn_dropout=me_attn_dropout,
                mlp_dropout=me_mlp_dropout,
                mlp_ratio=me_mlp_ratio,
                param_dtype=me_dtype,
                use_bias_qkv=me_use_bias_qkv,
                use_bias_out=me_use_bias_out,
                norm_streams_pre=me_norm_streams_pre,
                n_map_crossattn=me_n_map_crossattn,
                n_pma_seeds=me_n_pma_seeds,
                pma_dropout=me_pma_dropout,
                out_dim=self.embedding_dim,
            )(
                z_elevation,
                z_terrain,
                z_features,
                z_nw,
                z_resource,
                z_improvement,
                z_road,
                z_edge_river,
                z_cs_ownership,
                z_cs_centers_map,
                z_yield_map,
                z_improvement_yield_map,
                z_culture_yield_map,
                z_city_yield_map,
                z_religion_yield_map,
                # Visibility map
                z_visibility_map,
                # Player-specific information
                z_cities,
                z_units,
                twod_rope_nhnw,
                training=training,
                attention_bias=None,
            )

            ### Empire-wide information (my empire) ###
            # Technologies, policies, science_reserves, culture_reserves, faith_reserves, is_researching, num_trade_routes, 
            # num_delegates, free_techs, free_policies, great_works, gpps, gp_threshold, golden_age_turns, trade_offers, 
            # trade_ledger, trade_length_ledger, trade_gpt_adjustment, trade_resource_adjustment, have_met, treasury, 
            # happiness, most_population, least_population, most_crop_yield, least_crop_yield, most_manufactured_goods,
            # least_manufactured_goods, most_gnp, least_gnp, most_land, lest_land, most_soldiers, least_soldiers, 
            # most_approval, least_approval, most_literacy, least_literacy, player_id, current_turn
            my_techs = jnp.arange(len(Technologies))[None] * observation.technologies
            my_techs = jnp.where(my_techs == 0, -1, my_techs)
            z_tech, z_tech_table = _make_embeddings(
                num_embeddings=len(Technologies) + 1, features=self.embedding_dim, _dtype=embedding_dtype,
                name="tech_embeddings", initializer=embedding_init, indexing_data=my_techs,
                return_table=True
            )

            my_pols = jnp.arange(len(SocialPolicies))[None] * observation.policies
            my_pols = jnp.where(my_pols == 0, -1, my_pols)
            z_pols = _make_embeddings(
                num_embeddings=len(SocialPolicies) + 1, features=self.embedding_dim, _dtype=embedding_dtype,
                name="policy_embeddings", initializer=embedding_init, indexing_data=my_pols
            )

            
            my_is_researching = jnp.where(observation.is_researching == -1,
                                          len(Technologies), 
                                          observation.is_researching)
            z_is_researching = z_tech_table(my_is_researching.astype(jnp.int32))
            
            # ... all reserves*
            # For these, we need some way to meaningfully establish the values instead of just 
            # "letting the number drift" without anchoring. E.g., for science reserves, it could be 
            # a percentage of the cost of the thing being researched!
            # Some of the reserves don't have a hard cap. For these, we'll form a ratio of 
            # reserves / max(1, per turn)
            z_sci_reserves = encode_single_value(
                observation.science_reserves, denom=ALL_TECH_COST[my_is_researching],
                dim=self.embedding_dim, name="science_reserves"
            ) 
            
            z_culture_reserves = encode_single_value(
                observation.culture_reserves, denom=observation.culture_threshold,
                dim=self.embedding_dim, name="culture_reserves"
            ) 

            z_faith_reserves = encode_single_value(
                observation.faith_reserves,
                dim=self.embedding_dim, name="faith_reserves"
            )
            
            z_num_trade_routes = encode_single_value(
                observation.num_trade_routes, denom=jnp.array([12], dtype=jnp.float32),
                dim=self.embedding_dim, name="num_trade_routes"
            )

            z_cs_perturn_influence = encode_vector_values(
                observation.cs_perturn_influence,
                dim=self.embedding_dim, name="cs_perturn_influence"
            )
            
            z_num_delegates = encode_vector_values(
                observation.num_delegates, denom=jnp.array([12], dtype=jnp.float32),
                dim=self.embedding_dim, name="num_delegates"
            )

            
            all_tenets = jnp.arange(1, len(ReligiousTenets) + 1)[None, None] * observation.religious_tenets
            all_tenets = all_tenets - 1
            z_rel_tenets, religious_tenets_table = _make_embeddings(
                num_embeddings=len(ReligiousTenets) + 1, features=self.embedding_dim, _dtype=embedding_dtype,
                name="religious_tenets_embeddings", initializer=embedding_init, indexing_data=all_tenets,
                return_table=True
            )

            z_free_techs = encode_single_value(
                observation.free_techs, denom=jnp.array([5], dtype=jnp.float32),
                dim=self.embedding_dim, name="free_techs"
            )
            
            z_free_policies = encode_single_value(
                observation.free_policies, denom=jnp.array([5], dtype=jnp.float32),
                dim=self.embedding_dim, name="free_policies"
            )

            z_gws = encode_vector_values(
                observation.great_works,
                dim=self.embedding_dim, name="great_works"
            )
            
            z_gpps = encode_vector_values(
                observation.gpps, denom=observation.gp_threshold[:, None],
                dim=self.embedding_dim, name="great_person_points"
            )
            
            z_golden_age_turns = encode_single_value(
                observation.golden_age_turns, denom=jnp.array([20], dtype=jnp.float32),
                dim=self.embedding_dim, name="golden_age_turns"
            )
            
            z_trade_offers = _make_embeddings(
                num_embeddings=len(ALL_RESOURCES) + 4, features=self.embedding_dim, _dtype=embedding_dtype,
                name="trade_offer_embeddings", initializer=embedding_init, indexing_data=observation.trade_offers
            )
            
            z_trade_ledger = _make_embeddings(
                num_embeddings=len(ALL_RESOURCES) + 4, features=self.embedding_dim, _dtype=embedding_dtype,
                name="trade_ledger_embeddings", initializer=embedding_init, indexing_data=observation.trade_ledger
            )

            z_trade_length = encode_vector_values(
                observation.trade_length_ledger, denom=jnp.array([15]),
                dim=self.embedding_dim, name="trade_length"
            )

            z_trade_gpt = encode_single_value(
                observation.trade_gpt_adjustment,
                dim=self.embedding_dim, name="trade_gpt_adjustment"
            )
            
            # Resource adj is technically a count, but we can signifiy "already have/don't have" with 
            # categories and that's basically all of the info you need 
            all_res_adj = jnp.arange(1, len(ALL_RESOURCES) + 1)[None] * (observation.trade_resource_adjustment > 0)
            all_res_adj = all_res_adj - 1
            z_res_adj = _make_embeddings(
                num_embeddings=len(ALL_RESOURCES) + 1, features=self.embedding_dim, _dtype=embedding_dtype,
                name="res_adj_embeddings", initializer=embedding_init, indexing_data=all_res_adj
            )
            
            z_have_met = _make_embeddings(
                num_embeddings=2, features=self.embedding_dim, _dtype=embedding_dtype,
                name="have_met_embeddings", initializer=embedding_init, indexing_data=observation.have_met
            )
            
            z_treasury = encode_single_value(
                observation.treasury,
                dim=self.embedding_dim, name="treasury"
            )
            
            z_tourism_total = encode_vector_values(
                observation.tourism_total,
                dim=self.embedding_dim, name="tourism_total"
            ) 
            
            z_culture_total = encode_vector_values(
                observation.culture_total,
                dim=self.embedding_dim, name="culture_total"
            ) 

            z_happiness = encode_single_value(
                observation.happiness,
                dim=self.embedding_dim, name="happiness"
            )
            
            z_at_war = _make_embeddings(
                num_embeddings=2 + 1, features=self.embedding_dim, _dtype=embedding_dtype,
                name="at_war_embeddings", initializer=embedding_init, indexing_data=observation.at_war
            )
            
            z_has_sacked = _make_embeddings(
                num_embeddings=2 + 1, features=self.embedding_dim, _dtype=embedding_dtype,
                name="has_sacked_embeddings", initializer=embedding_init, indexing_data=observation.has_sacked
            )

            z_most_pop = player_embedding_table(observation.most_population.astype(jnp.int32))
            z_least_pop = player_embedding_table(observation.least_population.astype(jnp.int32))
            
            z_most_crop = player_embedding_table(observation.most_crop_yield.astype(jnp.int32))
            z_least_crop = player_embedding_table(observation.least_crop_yield.astype(jnp.int32))
            
            z_most_prod = player_embedding_table(observation.most_manufactured_goods.astype(jnp.int32))
            z_least_prod = player_embedding_table(observation.least_manufactured_goods.astype(jnp.int32))

            z_most_gnp = player_embedding_table(observation.most_gnp.astype(jnp.int32))
            z_least_gnp = player_embedding_table(observation.least_gnp.astype(jnp.int32))

            z_most_land = player_embedding_table(observation.most_land.astype(jnp.int32))
            z_least_land = player_embedding_table(observation.least_land.astype(jnp.int32))

            z_most_army = player_embedding_table(observation.most_soldiers.astype(jnp.int32))
            z_least_army = player_embedding_table(observation.least_soldiers.astype(jnp.int32))

            z_most_approval = player_embedding_table(observation.most_approval.astype(jnp.int32))
            z_least_approval = player_embedding_table(observation.least_approval.astype(jnp.int32))

            z_most_literacy = player_embedding_table(observation.most_literacy.astype(jnp.int32))
            z_least_literacy = player_embedding_table(observation.least_literacy.astype(jnp.int32))
            
            me_token = player_embedding_table(observation.player_id.astype(jnp.int32))
            
            z_current_turn = encode_single_value(
                observation.current_turn, denom=jnp.array([330]),
                dim=self.embedding_dim, name="current_turn"
            )
            
            z_culture_cs_resting_influence = encode_single_value(
                observation.culture_info.cs_resting_influence,
                dim=self.embedding_dim, name="culture_cs_resting_influence"
            )

            z_gamestate = GamestateEmbedder(
                num_seeds_tech=ge_num_seeds_tech,
                num_heads_tech=ge_num_heads_tech,
                num_seeds_pols=ge_num_seeds_pols,
                num_heads_pols=ge_num_heads_pols,
                num_heads_trade_offer=ge_num_heads_trade_offer,
                mlp_ratio_trade_offer=ge_mlp_ratio_trade_offer,
                pma_num_heads_trade_offer=ge_pma_num_heads_trade_offer,
                pma_num_seeds_trade_offer=ge_pma_num_seeds_trade_offer,
                num_heads_trade_ledger=ge_num_heads_trade_ledger,
                mlp_ratio_trade_ledger=ge_mlp_ratio_trade_ledger,
                pma_num_heads_trade_ledger=ge_pma_num_heads_trade_ledger,
                pma_num_seeds_trade_ledger=ge_pma_num_seeds_trade_ledger,
                num_heads_trade_length=ge_num_heads_trade_length,
                mlp_ratio_trade_length=ge_mlp_ratio_trade_length,
                pma_num_heads_trade_length=ge_pma_num_heads_trade_length,
                pma_num_seeds_trade_length=ge_pma_num_seeds_trade_length,
                num_heads_res_adj=ge_num_heads_res_adj,
                mlp_ratio_res_adj=ge_mlp_ratio_res_adj,
                pma_num_heads_res_adj=ge_pma_num_heads_res_adj,
                pma_num_seeds_res_adj=ge_pma_num_seeds_res_adj,
                num_heads_trade_summary=ge_num_heads_trade_summary,
                pma_num_heads_trade_summary=ge_pma_num_heads_trade_summary,
                pma_num_seeds_trade_summary=ge_pma_num_seeds_trade_summary,
                num_heads_tenets_inner=ge_num_heads_tenets_inner,
                mlp_ratio_tenets_inner=ge_mlp_ratio_tenets_inner,
                pma_num_heads_tenets_inner=ge_pma_num_heads_tenets_inner,
                pma_num_seeds_tenets_inner=ge_pma_num_seeds_tenets_inner,
                num_heads_tenets_players=ge_num_heads_tenets_players,
                mlp_ratio_tenets_players=ge_mlp_ratio_tenets_players,
                pma_num_heads_tenets_players=ge_pma_num_heads_tenets_players,
                pma_num_seeds_tenets_players=ge_pma_num_seeds_tenets_players,
                num_heads_cs=ge_num_heads_cs,
                mlp_ratio_cs=ge_mlp_ratio_cs,
                pma_num_heads_cs=ge_pma_num_heads_cs,
                pma_num_seeds_cs=ge_pma_num_seeds_cs,
                num_heads_delegates=ge_num_heads_dels,
                mlp_ratio_delegates=ge_mlp_ratio_dels,
                pma_num_heads_delegates=ge_pma_num_heads_dels,
                pma_num_seeds_delegates=ge_pma_num_seeds_dels,
                num_heads_gws=ge_num_heads_gws,
                mlp_ratio_gws=ge_mlp_ratio_gws,
                pma_num_heads_gws=ge_pma_num_heads_gws,
                pma_num_seeds_gws=ge_pma_num_seeds_gws,
                num_heads_gpps=ge_num_heads_gpps,
                mlp_ratio_gpps=ge_mlp_ratio_gpps,
                pma_num_heads_gpps=ge_pma_num_heads_gpps,
                pma_num_seeds_gpps=ge_pma_num_seeds_gpps,
                num_heads_have_met=ge_num_heads_have_met,
                mlp_ratio_have_met=ge_mlp_ratio_have_met,
                pma_num_heads_have_met=ge_pma_num_heads_have_met,
                pma_num_seeds_have_met=ge_pma_num_seeds_have_met,
                num_heads_tourism_inner=ge_num_heads_tourism_inner,
                mlp_ratio_tourism_inner=ge_mlp_ratio_tourism_inner,
                pma_num_heads_tourism_inner=ge_pma_num_heads_tourism_inner,
                pma_num_seeds_tourism_inner=ge_pma_num_seeds_tourism_inner,
                num_heads_tourism_players=ge_num_heads_tourism_players,
                mlp_ratio_tourism_players=ge_mlp_ratio_tourism_players,
                pma_num_heads_tourism_players=ge_pma_num_heads_tourism_players,
                pma_num_seeds_tourism_players=ge_pma_num_seeds_tourism_players,
                param_dtype=ge_param_dtype,
                n_general_selfattn=ge_n_general_selfattn,
                general_attention_bias=ge_general_attention_bias,
                general_num_heads=ge_general_num_heads,
                general_mlp_ratio=ge_general_mlp_ratio,
                general_pma_num_heads=ge_general_pma_num_heads,
                general_pma_num_seeds=ge_general_pma_num_seeds,
                n_aggregate_selfattn=ge_n_aggregate_selfattn,
                aggregate_attention_bias=ge_aggregate_attention_bias,
                aggregate_num_heads=ge_aggregate_num_heads,
                aggregate_mlp_ratio=ge_aggregate_mlp_ratio,
                aggregate_pma_num_heads=ge_aggregate_pma_num_heads,
                aggregate_pma_num_seeds=ge_aggregate_pma_num_seeds,
            )(
                z_tech,
                observation.technologies,
                z_pols,
                observation.policies,
                z_is_researching,
                z_sci_reserves,
                z_culture_reserves,
                z_faith_reserves,
                z_num_trade_routes,
                z_cs_perturn_influence,
                z_num_delegates,
                z_rel_tenets,
                z_free_techs,
                z_free_policies,
                z_gws,
                z_gpps,
                z_golden_age_turns,
                z_trade_offers,
                z_trade_ledger,
                z_trade_length,
                z_trade_gpt,
                z_res_adj,
                z_have_met,
                z_treasury,
                z_tourism_total,
                z_culture_total,
                z_happiness,
                z_most_pop,
                z_least_pop,
                z_most_crop,
                z_least_crop,
                z_most_prod,
                z_least_prod,
                z_most_gnp,
                z_least_gnp,
                z_most_land,
                z_least_land,
                z_most_army,
                z_least_army,
                z_most_approval,
                z_least_approval,
                z_most_literacy,
                z_least_literacy,
                z_culture_cs_resting_influence,
                z_at_war,
                z_has_sacked,
                me_token,
                z_current_turn,
                training    
            )
            
            ### Unit Information ###
            # we already encoded z_units above. However these were 
            # subsequently scattered. So let's do a non-scattered version
            shifted_units = jnp.where(
                (observation.units.unit_type + 1) == 0,
                len(GameUnits) + 2,
                observation.units.unit_type + 1
            ) - 1
            z_units_nonscatter_raw = units_embedding_table(
                shifted_units.astype(jnp.int32)
            )
            
            # For many of the other unit-related variables, we can borrow from earlier
            # preprocessing steps
            # cb_tok, hp_tok, ap_tok, pos_tok_units
            z_engaged_n_turns = encode_vector_values(
                observation.units.engaged_for_n_turns, denom=jnp.array([14], dtype=jnp.float32),
                dim=self.embedding_dim, name="engaged_n_turns"
            )

            z_action_cat = _make_embeddings(
                num_embeddings=len(ALL_ACTION_FUNCTIONS), features=self.embedding_dim, 
                _dtype=embedding_dtype, name="action_cat_embeddings", 
                initializer=embedding_init, indexing_data=observation.units.engaged_action_id
            )

            z_trade_to_player_int = embed_player_or_cs(
                player_embedding_table, cs_embedding_table, 
                observation.units.trade_to_player_int
            )
            
            z_trade_to_city_int = city_int_table(
                observation.units.trade_to_city_int.astype(jnp.int32)
            )

            z_trade_from_city_int = city_int_table(
                observation.units.trade_from_city_int.astype(jnp.int32)
            )

            z_trade_yields = encode_vector_values(
                observation.units.trade_yields, denom=jnp.array([20]),
                dim=self.embedding_dim, name="trade_yields"
            )
            
            z_culture_ypk = encode_vector_values(
                observation.culture_info.yields_per_kill, 
                dim=self.embedding_dim, name="culture_yields_per_kill"
            )
            z_culture_hf_ypk = encode_vector_values(
                observation.culture_info.honor_finisher_yields_per_kill, 
                dim=self.embedding_dim, name="culture_honor_finisher_yields_per_kill"
            )
            z_culture_cs_trade_yields = encode_vector_values(
                observation.culture_info.cs_trade_route_yields, denom=jnp.array([20]),
                dim=self.embedding_dim, name="culture_cs_trade_yields"
            )

            # (B, max_num_units, 2, 10) -> (B, max_num_units)
            trade_yields_mask = (observation.units.trade_yields > 0).any((-1, -2))

            # (B, max_num_units)
            is_caravan_mask = (observation.units.unit_type[games_idx, observation.player_id] == GameUnits["caravan"]._value_)

            z_units_nonscatter = UnitEncoder(
                units_inner_num_heads=ue_units_inner_num_heads,
                units_inner_mlp_ratio=ue_units_inner_mlp_ratio,
                units_inner_pma_num_heads=ue_units_inner_pma_num_heads,
                units_inner_pma_num_seeds=ue_units_inner_pma_num_seeds,
                units_players_num_heads=ue_units_players_num_heads,
                units_players_mlp_ratio=ue_units_players_mlp_ratio,
                units_players_pma_num_heads=ue_units_players_pma_num_heads,
                units_players_pma_num_seeds=ue_units_players_pma_num_seeds,
                trade_yield_num_heads=ue_trade_yield_num_heads,
                trade_yield_mlp_ratio=ue_trade_yield_mlp_ratio,
                trade_yield_pma_num_heads=ue_trade_yield_pma_num_heads,
                trade_yield_pma_num_seeds=ue_trade_yield_pma_num_seeds,
                trade_units_num_heads=ue_trade_units_num_heads,
                trade_units_mlp_ratio=ue_trade_units_mlp_ratio,
                trade_units_pma_num_heads=ue_trade_units_pma_num_heads,
                trade_units_pma_num_seeds=ue_trade_units_pma_num_seeds,
                units_summary_num_heads=ue_units_summary_num_heads,
                units_summary_mlp_ratio=ue_units_summary_mlp_ratio,
                units_summary_pma_num_heads=ue_units_summary_pma_num_heads,
                units_summary_pma_num_seeds=ue_units_summary_pma_num_seeds,
                n_units_summary_selfattn=ue_n_units_summary_selfattn,
            )(
                z_units_nonscatter_raw,
                cb_tok,
                hp_tok,
                ap_tok,
                pos_tok_units,
                z_engaged_n_turns,
                z_action_cat,
                z_trade_to_player_int,
                z_trade_to_city_int,
                z_trade_from_city_int,
                z_trade_yields,
                z_culture_ypk,
                z_culture_hf_ypk,
                z_culture_cs_trade_yields,
                observation.units.trade_to_player_int,
                observation.units.trade_from_city_int,
                me_token,
                player_embedding_table,
                trade_yields_mask,
                is_caravan_mask,
                training,
            )

            ### City state information ###
            z_religious_population = encode_vector_values(
                observation.citystate_info.religious_population,
                denom=jnp.array([30]),
                dim=self.embedding_dim, name="cs_religious_population"
            )
            z_cs_relationships = _make_embeddings(
                num_embeddings=6 + 1, features=self.embedding_dim, _dtype=embedding_dtype,
                name="cs_relationship_embeddings", initializer=embedding_init,
                indexing_data=observation.citystate_info.relationships
            )
            z_influence_level = encode_vector_values(
                observation.citystate_info.influence_level,
                denom=jnp.array([400]),
                dim=self.embedding_dim, name="cs_influence_level"
            )
            z_cs_type = _make_embeddings(
                num_embeddings=6 + 1, features=self.embedding_dim, _dtype=embedding_dtype,
                name="cs_type_embeddings", initializer=embedding_init, 
                indexing_data=observation.citystate_info.cs_type
            )
            z_cs_quest_type = _make_embeddings(
                num_embeddings=8 + 1, features=self.embedding_dim, _dtype=embedding_dtype,
                name="cs_quest_type_embeddings", initializer=embedding_init, 
                indexing_data=observation.citystate_info.quest_type
            )
            z_culture_tracker = encode_single_value(
                observation.citystate_info.culture_tracker_mine,
                dim=self.embedding_dim, name="cs_culture_tracker"
            ) 
            z_faith_tracker = encode_single_value(
                observation.citystate_info.faith_tracker_mine,
                dim=self.embedding_dim, name="cs_faith_tracker"
            )
            z_tech_tracker = encode_single_value(
                observation.citystate_info.tech_tracker_mine,
                dim=self.embedding_dim, name="cs_tech_tracker"
            )
            z_trade_tracker = encode_vector_values(
                observation.citystate_info.trade_tracker_mine,
                dim=self.embedding_dim, name="cs_trade_tracker"
            )
            z_religion_tracker = encode_single_value(
                observation.citystate_info.religion_tracker_mine,
                dim=self.embedding_dim, name="cs_religion_tracker"
            )
            z_wonder_tracker = encode_single_value(
                observation.citystate_info.wonder_tracker_mine,
                dim=self.embedding_dim, name="cs_wonder_tracker"
            )
            z_resource_tracker = encode_single_value(
                observation.citystate_info.resource_tracker_mine,
                dim=self.embedding_dim, name="cs_resource_tracker"
            )
            z_culture_tracker_ratio = encode_single_value(
                observation.citystate_info.culture_tracker_mine,
                denom=observation.citystate_info.culture_tracker_lead,
                dim=self.embedding_dim, name="cs_culture_tracker_ratio"
            ) 
            z_faith_tracker_ratio = encode_single_value(
                observation.citystate_info.faith_tracker_mine,
                denom=observation.citystate_info.faith_tracker_lead,
                dim=self.embedding_dim, name="cs_faith_tracker_ratio"
            )
            z_tech_tracker_ratio = encode_single_value(
                observation.citystate_info.tech_tracker_mine,
                denom=observation.citystate_info.tech_tracker_lead,
                dim=self.embedding_dim, name="cs_tech_tracker_ratio"
            )
            z_trade_tracker_ratio = encode_vector_values(
                observation.citystate_info.trade_tracker_mine,
                denom=observation.citystate_info.trade_tracker_lead,
                dim=self.embedding_dim, name="cs_trade_tracker_ratio"
            )
            z_religion_tracker_ratio = encode_single_value(
                observation.citystate_info.religion_tracker_mine,
                denom=observation.citystate_info.religion_tracker_lead,
                dim=self.embedding_dim, name="cs_religion_tracker_ratio"
            )
            z_wonder_tracker_ratio = encode_single_value(
                observation.citystate_info.wonder_tracker_mine,
                denom=observation.citystate_info.wonder_tracker_lead,
                dim=self.embedding_dim, name="cs_wonder_tracker_ratio"
            )
            z_resource_tracker_ratio = encode_single_value(
                observation.citystate_info.resource_tracker_mine,
                denom=observation.citystate_info.resource_tracker_lead,
                dim=self.embedding_dim, name="cs_resource_tracker_ratio"
            )

            z_cs = CityStateEncoder(
                rel_num_heads=cse_rel_num_heads,
                rel_mlp_ratio=cse_rel_mlp_ratio,
                rel_pma_num_heads=cse_rel_pma_num_heads,
                rel_pma_num_seeds=cse_rel_pma_num_seeds,
                cs_num_heads=cse_cs_num_heads,
                cs_mlp_ratio=cse_cs_mlp_ratio,
                cs_pma_num_heads=cse_cs_pma_num_heads,
                cs_pma_num_seeds=cse_cs_pma_num_seeds,
                global_num_heads=cse_global_num_heads,
                global_mlp_ratio=cse_global_mlp_ratio,
                global_pma_num_heads=cse_global_pma_num_heads,
                global_pma_num_seeds=cse_global_pma_num_seeds,
                fuse_num_heads=cse_fuse_num_heads,
                fuse_mlp_ratio=cse_fuse_mlp_ratio,
                fuse_pma_num_heads=cse_fuse_pma_num_heads,
                fuse_pma_num_seeds=cse_fuse_pma_num_seeds,
                n_fuse_selfattn=cse_n_fuse_selfattn
            )(
                z_religious_population,
                z_cs_relationships,
                z_influence_level,
                z_cs_type,
                z_cs_quest_type,
                z_culture_tracker,
                z_faith_tracker,
                z_tech_tracker,
                z_trade_tracker,
                z_religion_tracker,
                z_wonder_tracker,
                z_resource_tracker,
                z_culture_tracker_ratio,
                z_faith_tracker_ratio,
                z_tech_tracker_ratio,
                z_trade_tracker_ratio,
                z_religion_tracker_ratio,
                z_wonder_tracker_ratio,
                z_resource_tracker_ratio,
                training
            )

            ### City information ###
            # Should somehow bringin the tiles that this city can own vs can work (does own)
            # This will include both city and religion info (as this is on a per-city basis)
            z_city_ids = observation.player_cities.city_ids[games_idx, observation.player_id]
            z_city_ids = city_kind_embedding_table(z_city_ids.astype(jnp.int32))

            z_city_rowcols = observation.player_cities.city_rowcols[games_idx, observation.player_id]
            z_city_rowcols = encode_vector_values(
                z_city_rowcols, denom=jnp.array([42, 66], dtype=jnp.float32)[None, None],
                dim=self.embedding_dim, name="my_city_rowcols"
            )
           
            z_city_yields = encode_vector_values(
                observation.player_cities.yields, 
                dim=self.embedding_dim, name="my_city_yields"
            )
            z_city_center_yields = encode_vector_values(
                observation.player_cities.city_center_yields, 
                dim=self.embedding_dim, name="my_city_center_yields"
            )
            z_building_yields = encode_vector_values(
                observation.player_cities.building_yields, 
                dim=self.embedding_dim, name="my_city_building_yields"
            )
            z_culture_building_yields = encode_vector_values(
                observation.culture_info.building_yields, 
                dim=self.embedding_dim, name="culture_building_yields"
            )
            z_religion_building_yields = encode_vector_values(
                observation.player_cities.religion_info.building_yields, 
                dim=self.embedding_dim, name="religion_building_yields"
            )

            z_city_population = observation.player_cities.population[games_idx, observation.player_id]
            z_city_population = encode_vector_values(
                z_city_population, 
                dim=self.embedding_dim, name="my_city_population"
            )
            
            # In the encoder, we need to cross-attn this information with the 
            # numerous layers of the map?
            # {0,1} -> {unworked, worked}
            z_city_worked_slots = _make_embeddings(
                num_embeddings=2, features=self.embedding_dim, _dtype=embedding_dtype, 
                name="worked_slots_embedding",
                initializer=embedding_init, indexing_data=observation.player_cities.worked_slots
            )
            
            z_city_specialists = encode_vector_values(
                observation.player_cities.specialist_slots, 
                dim=self.embedding_dim, name="my_city_specialists"
            )
            z_city_gws = encode_vector_values(
                observation.player_cities.gw_slots, 
                dim=self.embedding_dim, name="my_city_gws"
            )
            z_city_food_reserves = encode_vector_values(
                observation.player_cities.food_reserves, 
                dim=self.embedding_dim, name="my_city_food_reserves"
            )
            z_city_growth_carryover = encode_vector_values(
                observation.player_cities.growth_carryover, 
                dim=self.embedding_dim, name="my_city_growth_carryover"
            )
            z_city_prod_reserves = encode_vector_values(
                observation.player_cities.prod_reserves, 
                dim=self.embedding_dim, name="my_city_prod_reserves"
            )
            z_city_prod_carryover = encode_vector_values(
                observation.player_cities.prod_carryover, 
                dim=self.embedding_dim, name="my_city_prod_carryover"
            )

            z_city_constructing, constructing_embedding_table = _make_embeddings(
                num_embeddings=len(GameBuildings) + len(GameUnits) + 1, 
                features=self.embedding_dim, _dtype=embedding_dtype, 
                name="is_constructing_embedding",
                initializer=embedding_init, indexing_data=observation.player_cities.is_constructing,
                return_table=True
            )
            
            z_city_building_maintenance = encode_vector_values(
                observation.player_cities.bldg_maintenance, 
                dim=self.embedding_dim, name="my_city_bldg_maintenance"
            )

            z_city_defense = observation.player_cities.defense[games_idx, observation.player_id]
            z_city_defense = encode_vector_values(
                z_city_defense, 
                dim=self.embedding_dim, name="my_city_defense"
            )
            z_city_hp = observation.player_cities.hp[games_idx, observation.player_id]
            z_city_hp = encode_vector_values(
                z_city_hp, 
                dim=self.embedding_dim, name="my_city_hp"
            )
            
            # (B, max_num_cities, E)
            city_exists_table = nn.Embed(2, z_city_hp.shape[-1], name="city_exists_embedding")
            z_city_exists = city_exists_table(
                (observation.player_cities.city_ids[games_idx, observation.player_id] > 0).astype(jnp.int32)
            )

            # (1, 1, num_bldgs, E) -> (B, max_num_cities, num_bldgs, E)
            my_buildings = jnp.arange(len(GameBuildings), dtype=jnp.int32)[None, None] 
            my_buildings = constructing_embedding_table(my_buildings)
            my_buildings = jnp.broadcast_to(
                my_buildings, 
                (z_city_hp.shape[0], z_city_hp.shape[1], my_buildings.shape[2], my_buildings.shape[3])
            )
            z_city_my_buildings = film_modulate(
                my_buildings, observation.player_cities.buildings_owned[..., None].astype(jnp.float32),
                name="my_city_buildings_owned", param_dtype=my_buildings.dtype,
                out_dtype=my_buildings.dtype
            ) 
            
            # (1, 1, num_resources, E)
            my_resources = jnp.arange(len(ALL_RESOURCES))[None, None]
            my_resources = resources_table(my_resources)
            my_resources = jnp.broadcast_to(
                my_resources,
                (z_city_hp.shape[0], z_city_hp.shape[1], my_resources.shape[2], my_resources.shape[3])
            )

            z_city_my_resources = film_modulate(
                my_resources, (observation.player_cities.resources_owned[..., None] > 0).astype(jnp.float32),
                name="my_city_resources_owned", param_dtype=my_resources.dtype,
                out_dtype=my_resources.dtype
            )

            is_coastal = observation.player_cities.is_coastal[games_idx, observation.player_id]
            z_city_is_coastal = encode_vector_values(
                is_coastal, 
                dim=self.embedding_dim, name="my_city_is_coastal"
            )
            z_city_culture_reserves_for_border = encode_vector_values(
                observation.player_cities.culture_reserves_for_border, 
                dim=self.embedding_dim, name="my_city_culture_reserves_for_border"
            )
            z_city_gpps = encode_vector_values(
                observation.player_cities.great_person_points, 
                dim=self.embedding_dim, name="my_city_culture_great_person_points"
            )
            
            z_city_religious_population = encode_vector_values(
                observation.player_cities.religion_info.religious_population, 
                dim=self.embedding_dim, name="my_city_religious_population"
            )

            my_city_tenets = jnp.arange(1, len(ReligiousTenets) + 1)[None, None] * observation.player_cities.religion_info.religious_tenets_per_city
            my_city_tenets = my_city_tenets - 1
            my_city_tenets = jnp.where(my_city_tenets == -1, len(ReligiousTenets), my_city_tenets)
            z_my_city_tenets = religious_tenets_table(my_city_tenets)

            z_cs_perturn_influence_cumsum = encode_vector_values(
                observation.player_cities.religion_info.cs_perturn_influence_cumulative, 
                dim=self.embedding_dim, name="my_city_cs_perturn_influence_cumulative"
            )
            
            z_player_perturn_influence_cumsum = encode_vector_values(
                observation.player_cities.religion_info.player_perturn_influence_cumulative, 
                dim=self.embedding_dim, name="my_city_player_perturn_influence_cumulative"
            )
            
            z_my_city = CityEncoder(
                pooling_num_heads=ce_pooling_num_heads,
                pooling_mlp_ratio=ce_pooling_mlp_ratio,
                pooling_pma_seeds=ce_pooling_pma_num_seeds,
                fuse_num_heads=ce_fuse_num_heads,
                fuse_mlp_ratio=ce_fuse_mlp_ratio,
                fuse_pma_num_heads=ce_fuse_pma_num_heads,
                fuse_pma_num_seeds=ce_fuse_pma_num_seeds,
                relation_num_heads=ce_relation_num_heads,
                relation_mlp_ratio=ce_relation_mlp_ratio,
                relation_pma_num_heads=ce_relation_pma_num_heads,
                relation_pma_num_seeds=ce_relation_pma_num_seeds,
                n_fuse_selfattn=ce_n_fuse_selfattn,
                fuse_pma_num_seeds_final=ce_fuse_pma_num_seeds_final
            )(
                z_city_ids,
                z_city_rowcols + z_city_exists[:, :, None],
                z_city_yields + z_city_exists[:, :, None],
                z_city_center_yields + z_city_exists[:, :, None],
                z_building_yields + z_city_exists[:, :, None],
                z_culture_building_yields + z_city_exists[:, :, None],
                z_religion_building_yields + z_city_exists[:, :, None],
                z_city_population + z_city_exists,
                z_city_worked_slots + z_city_exists[:, :, None],
                z_city_specialists + z_city_exists[:, :, None],
                z_city_gws + z_city_exists[:, :, None],
                z_city_food_reserves + z_city_exists,
                z_city_growth_carryover + z_city_exists,
                z_city_prod_reserves + z_city_exists,
                z_city_prod_carryover + z_city_exists,
                z_city_constructing + z_city_exists,
                z_city_building_maintenance + z_city_exists,
                z_city_defense + z_city_exists,
                z_city_hp + z_city_exists,
                z_city_my_buildings + z_city_exists[:, :, None],
                z_city_my_resources + z_city_exists[:, :, None],
                z_city_is_coastal + z_city_exists,
                z_city_culture_reserves_for_border + z_city_exists,
                z_city_gpps + z_city_exists[:, :, None],
                z_city_religious_population + z_city_exists[:, :, None],
                z_my_city_tenets + z_city_exists[:, :, None],
                z_cs_perturn_influence_cumsum + z_city_exists[:, :, None],
                z_player_perturn_influence_cumsum + z_city_exists[:, :, None, None],
                training,
            )
            
            z_V = ValueFunction(
                use_stream_gates=V_use_stream_gates,
                n_selfattn=V_n_selfattn,
                num_heads=V_num_heads,
                mlp_ratio=V_mlp_ratio,
                param_dtype=V_param_dtype,
                activation=V_activation,
                use_bias_qkv=V_use_bias_qkv,
                use_bias_out=V_use_bias_out,
                attention_bias=V_attention_bias,
                pma_num_heads=V_pma_num_heads,
                pma_num_seeds=V_pma_num_seeds,
                head_hidden_mult=V_head_hidden_mult,
            )(
                z_map,
                z_gamestate,
                z_units_nonscatter,
                z_cs,
                z_my_city,
                training,
            )

            actions_trade_deals = ActionHeadTradeDeals(
                use_stream_gates=Atd_use_stream_gates,
                n_selfattn=Atd_n_selfattn,
                num_heads=Atd_num_heads,
                mlp_ratio=Atd_mlp_ratio,
                param_dtype=Atd_param_dtype,
                activation=Atd_activation,
                use_bias_qkv=Atd_use_bias_qkv,
                use_bias_out=Atd_use_bias_out,
                attention_bias=Atd_attention_bias,
                pma_num_heads=Atd_pma_num_heads,
                pma_num_seeds=Atd_pma_num_seeds,
                num_heads_trade_offer=Atd_num_heads_trade_offer,
                mlp_ratio_trade_offer=Atd_mlp_ratio_trade_offer,
                num_heads_trade_ledger=Atd_num_heads_trade_ledger,
                mlp_ratio_trade_ledger=Atd_mlp_ratio_trade_ledger,
                pma_num_heads_trade_ledger=Atd_pma_num_heads_trade_ledger,
                pma_num_seeds_trade_ledger=Atd_pma_num_seeds_trade_ledger,
                num_heads_res_adj=Atd_num_heads_res_adj,
                mlp_ratio_res_adj=Atd_mlp_ratio_res_adj,
                pma_num_heads_res_adj=Atd_pma_num_heads_res_adj,
                pma_num_seeds_res_adj=Atd_pma_num_seeds_res_adj,
                num_heads_trade_summary=Atd_num_heads_trade_summary,
                mlp_ratio_trade_summary=Atd_mlp_ratio_trade_summary,
                pma_num_heads_trade_summary=Atd_pma_num_heads_trade_summary,
                pma_num_seeds_trade_summary=Atd_pma_num_seeds_trade_summary,
            )(
                z_map,
                z_gamestate,
                z_units_nonscatter,
                z_cs,
                z_my_city,
                z_trade_offers,
                z_trade_ledger,
                z_trade_length,
                z_trade_gpt,
                z_res_adj,
                z_have_met,
                z_treasury,
                z_city_my_resources + z_city_exists[:, :, None],
                training,
            )

            actions_social_policies = ActionHeadSocialPolicies(
                use_stream_gates=Asp_use_stream_gates,
                n_selfattn=Asp_n_selfattn,
                num_heads=Asp_num_heads,
                mlp_ratio=Asp_mlp_ratio,
                pma_num_heads=Asp_pma_num_heads,
                pma_num_seeds=Asp_pma_num_seeds,
                param_dtype=Asp_param_dtype,
                activation=Asp_activation,
                use_bias_qkv=Asp_use_bias_qkv,
                use_bias_out=Asp_use_bias_out,
                attention_bias=Asp_attention_bias,
                num_heads_pols=Asp_num_heads_pols,
                n_general_selfattn=Asp_n_general_selfattn,
                general_num_heads=Asp_general_num_heads,
                general_mlp_ratio=Asp_general_mlp_ratio,
            )(
                z_map,
                z_gamestate,
                z_units_nonscatter,
                z_cs,
                z_my_city,
                z_culture_reserves,
                z_free_policies,
                z_pols,
                observation.policies,
                training,
            )

            actions_religion = ActionHeadReligion(
                use_stream_gates=Ar_use_stream_gates,
                n_selfattn=Ar_n_selfattn,
                num_heads=Ar_num_heads,
                mlp_ratio=Ar_mlp_ratio,
                param_dtype=Ar_param_dtype,
                activation=Ar_activation,
                use_bias_qkv=Ar_use_bias_qkv,
                use_bias_out=Ar_use_bias_out,
                attention_bias=Ar_attention_bias,
                pma_num_heads=Ar_pma_num_heads,
                pma_num_seeds=Ar_pma_num_seeds,
                pma_num_heads_tenets_inner=Ar_pma_num_heads_tenets_inner,
                pma_num_seeds_tenets_inner=Ar_pma_num_seeds_tenets_inner,
                num_heads_tenets_inner=Ar_num_heads_tenets_inner,
                mlp_ratio_tenets_inner=Ar_mlp_ratio_tenets_inner,
                general_num_heads=Ar_general_num_heads,
                general_mlp_ratio=Ar_general_mlp_ratio,
                n_general_selfattn=Ar_n_general_selfattn,
            )(
                z_map,
                z_gamestate,
                z_units_nonscatter,
                z_cs,
                z_my_city,
                z_rel_tenets,
                observation.religious_tenets,
                z_faith_reserves,
                observation.player_id,
                games_idx,
                player_embedding_table,
                training
            )

            actions_technology = ActionHeadTechnology(
                use_stream_gates=At_use_stream_gates,
                n_selfattn=At_n_selfattn,
                num_heads=At_num_heads,
                mlp_ratio=At_mlp_ratio,
                param_dtype=At_param_dtype,
                activation=At_activation,
                use_bias_qkv=At_use_bias_qkv,
                use_bias_out=At_use_bias_out,
                attention_bias=At_attention_bias,
                num_heads_tech=At_num_heads_tech,
                general_num_heads=At_general_num_heads,
                general_mlp_ratio=At_general_mlp_ratio,
                n_general_selfattn=At_n_general_selfattn,
                pma_num_heads=At_pma_num_heads,
                pma_num_seeds=At_pma_num_seeds,
            )(
                z_map,
                z_gamestate,
                z_units_nonscatter,
                z_cs,
                z_my_city,
                z_tech,
                observation.technologies,
                z_is_researching,
                z_sci_reserves,
                z_free_techs,
                training
            )

            actions_units = ActionHeadUnits(
                use_stream_gates=Au_use_stream_gates,
                n_selfattn=Au_n_selfattn,
                num_heads=Au_num_heads,
                mlp_ratio=Au_mlp_ratio,
                param_dtype=Au_param_dtype,
                activation=Au_activation,
                use_bias_qkv=Au_use_bias_qkv,
                use_bias_out=Au_use_bias_out,
                attention_bias=Au_attention_bias,
                pma_num_heads=Au_pma_num_heads,
                pma_num_seeds=Au_pma_num_seeds,
                units_inner_num_heads=Au_units_inner_num_heads,
                units_inner_mlp_ratio=Au_units_inner_mlp_ratio,
                units_players_num_heads=Au_units_players_num_heads,
                units_players_mlp_ratio=Au_units_players_mlp_ratio,
                trade_yield_num_heads=Au_trade_yields_num_heads,
                trade_yield_mlp_ratio=Au_trade_yields_mlp_ratio,
                trade_yield_pma_num_heads=Au_trade_yield_pma_num_heads,
                trade_yield_pma_num_seeds=Au_trade_yield_pma_num_seeds,
                trade_units_num_heads=Au_trade_units_num_heads,
                trade_units_mlp_ratio=Au_trade_units_mlp_ratio,
                my_units_num_heads=Au_my_units_num_heads,
                my_units_mlp_ratio=Au_my_units_mlp_ratio,
                n_my_units_selfattn=Au_n_my_units_selfattn
            )(
                z_map,
                z_gamestate,
                z_units_nonscatter,
                z_cs,
                z_my_city,
                z_units_nonscatter_raw,
                cb_tok,
                hp_tok,
                ap_tok,
                pos_tok_units,
                z_engaged_n_turns,
                z_action_cat,
                z_trade_to_player_int,
                z_trade_from_city_int,
                z_trade_to_city_int,
                observation.units.trade_to_player_int,
                observation.units.trade_from_city_int,
                z_trade_yields,
                z_culture_ypk,
                z_culture_hf_ypk,
                trade_yields_mask,
                is_caravan_mask,
                me_token,
                player_embedding_table,
                games_idx,
                observation.player_id,
                training,
            )
            
            actions_cities = ActionHeadCities(
                use_stream_gates=Ac_use_stream_gates,
                n_selfattn=Ac_n_selfattn,
                num_heads=Ac_num_heads,
                mlp_ratio=Ac_mlp_ratio,
                param_dtype=Ac_param_dtype,
                activation=Ac_activation,
                use_bias_qkv=Ac_use_bias_qkv,
                use_bias_out=Ac_use_bias_out,
                attention_bias=Ac_attention_bias,
                pma_num_heads=Ac_pma_num_heads,
                pma_num_seeds=Ac_pma_num_seeds,
                pooling_num_heads=Ac_pooling_num_heads,
                pooling_pma_seeds=Ac_pooling_pma_seeds,
                pooling_mlp_ratio=Ac_pooling_mlp_ratio,
                fuse_num_heads=Ac_fuse_num_heads,
                fuse_mlp_ratio=Ac_fuse_mlp_ratio,
                fuse_pma_num_heads=Ac_fuse_pma_num_heads,
                fuse_pma_num_seeds=Ac_fuse_pma_num_seeds,
            )(
                z_map,
                z_gamestate,
                z_units_nonscatter,
                z_cs,
                z_my_city,
                z_city_ids,
                z_city_rowcols + z_city_exists[:, :, None],
                z_city_yields + z_city_exists[:, :, None],
                z_city_center_yields + z_city_exists[:, :, None],
                z_building_yields + z_city_exists[:, :, None],
                z_culture_building_yields + z_city_exists[:, :, None],
                z_religion_building_yields + z_city_exists[:, :, None],
                z_city_population + z_city_exists,
                z_city_worked_slots + z_city_exists[:, :, None],
                z_city_specialists + z_city_exists[:, :, None],
                z_city_gws + z_city_exists[:, :, None],
                z_city_food_reserves + z_city_exists,
                z_city_growth_carryover + z_city_exists,
                z_city_prod_reserves + z_city_exists,
                z_city_prod_carryover + z_city_exists,
                z_city_constructing + z_city_exists,
                z_city_building_maintenance + z_city_exists,
                z_city_defense + z_city_exists,
                z_city_hp + z_city_exists,
                z_city_my_buildings + z_city_exists[:, :, None],
                z_city_my_resources + z_city_exists[:, :, None],
                z_city_is_coastal + z_city_exists,
                z_city_culture_reserves_for_border + z_city_exists,
                z_city_gpps + z_city_exists[:, :, None],
                z_city_religious_population + z_city_exists[:, :, None],
                z_my_city_tenets + z_city_exists[:, :, None],
                observation.player_cities.ownership_map,
                games_idx, 
                observation.player_id,
                training,
            )

            return (
                actions_trade_deals,
                actions_social_policies,
                actions_religion,
                actions_technology,
                actions_units,
                actions_cities
            ), z_V
            
    return TerraNovaModel(embedding_dim=embedding_dim)
