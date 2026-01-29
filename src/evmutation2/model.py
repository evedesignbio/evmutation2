"""
Custom modules for ML model
"""

from functools import partial, wraps
from typing import Tuple, Sequence, Literal

import torch
from torch.nn.functional import one_hot, pad, cross_entropy
from torch.nn import Module, ModuleList, Parameter, Sequential, LayerNorm, Linear
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import einx
from einops import rearrange, repeat, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange
import lightning as L
from lightning.pytorch.utilities import grad_norm
from loguru import logger
import numpy as np
import pandas as pd

from evmutation2.alphafold3_pytorch.alphafold3 import (
    LinearNoBias, LinearNoBiasThenOuterSum, OuterProductMean, MSAModule, TemplateEmbedder,
    PreLayerNorm, MSAPairWeightedAveraging, Transition, PairwiseBlock, PairformerStack, Dropout,
    default, exists, max_neg_value, pack_one, should_checkpoint, identity
)
from evmutation2.alphafold3_pytorch.attention import Attention
from evmutation2.alphafold3_pytorch.tensor_typing import Float, Bool, Int, typecheck

from evmutation2 import features
from evmutation2 import utils


# constants for encoder
DIM_SINGLE = 384
DIM_PAIRWISE = 128
DIM_PROFILE = utils.NUM_CLASSES
DIM_DELETION_MEAN = 1
DIM_SINGLE_INPUT = DIM_PROFILE + DIM_DELETION_MEAN
BINS_POSITION_ENCODING = 32

DIM_S_INPUT = 25
MSA_LAYER_DEPTH = 4
DIM_MSA = 64
DIM_MSA_INPUT = utils.NUM_CLASSES
DIM_ADDITIONAL_MSA_FEATS = 2  # + 2 for has_deletion and deletion_value
DIM_FULL_MSA_INPUT = DIM_MSA_INPUT + DIM_ADDITIONAL_MSA_FEATS

MSA_NUM_SAMPLES = 1024  # TODO: fix a reasonable number here, not specified in AF3 supplement
MSA_DIM_OUTER_PRODUCT_HIDDEN = 32
# PWA = pair-weighted averaging
MSA_PWA_DROPOUT_ROW_PROB = 0.15
MSA_PWA_HEADS = 8
MSA_PWA_DIM_HEAD = 32

# Pairformer
PAIRFORMER_DEPTH = 48
PAIRFORMER_DIM_HEAD = 64
PAIRFORMER_HEADS = 16
PAIRFORMER_DROPOUT = 0.25

# constants for decoder
DIM_SINGLE_DECODER = DIM_SINGLE  # TODO: find a reasonable choice, for now same as DIM_SINGLE

# ... decoder
DECODER_LAYER_DEPTH = 4  # TODO: find a reasonable choice, for now same as MSAModule depth
DECODER_EDGE_HEADS = 8  # TODO: find a reasonable choice, for now same as MSAModule depth
DECODER_NODE_HEADS = 16  # TODO: find a reasonable choice, for now same as AttentionPairBias depth
DECODER_SELF_HEADS = 16  # TODO: find a reasonable choice, for now same as AttentionPairBias depth
DECODER_EDGE_HEAD_DIM = 32  # TODO: find a reasonable choice, for now same as MSAModule depth
DECODER_NODE_HEAD_DIM = 64  # TODO: find a reasonable choice, for now same as AttentionPairBias depth
DECODER_SELF_HEAD_DIM = 64  # TODO: find a reasonable choice, for now same as AttentionPairBias depth
DECODER_SELF_DROPOUT = 0.25  # TODO: find a reasonable choice, for now same as AttentionPairBias depth
DECODER_EDGE_DROPOUT = 0.25  # TODO: find a reasonable choice, for now same as AttentionPairBias depth
DECODER_NODE_DROPOUT = 0.25  # TODO: find a reasonable choice, for now same as AttentionPairBias depth

# note the above are AF3 settings except for decoder, here were set specific params for our encoder-decoder
# architecture
DEFAULTS_PAIRWISE_BLOCK = {
    "tri_attn_dim_head": 16,
    "tri_mult_dim_hidden": None,
    "tri_attn_heads": 4,
    "dropout_row_prob": 0.25,
    "dropout_col_prob": 0.25,
    "tri_attn_kwargs": {
        "legacy_gating": True,  # to match AF3, this should be False
        "query_bias": True,  # to match AF3, this should be False
    }
}

DEFAULT_DIM_TEMPLATE_FEATS = 92

ENCODER_DEFAULT_SETTINGS = {
    "dim_single": 192,
    "dim_pairwise": 64,
    "msa_module_kwargs": {
        "depth": 4,
        "checkpoint": True,
        # "checkpoint_segments": 4*4,
        "dim_msa_input": DIM_MSA_INPUT,
        "dim_additional_msa_feats": DIM_ADDITIONAL_MSA_FEATS,
        "max_num_msa": 1024,
        # ----
        "pairwise_block_kwargs": DEFAULTS_PAIRWISE_BLOCK,
        "dim_msa": 32,
        "outer_product_mean_dim_hidden": 16,
        "msa_pwa_dropout_row_prob": 0.15,
        "msa_pwa_heads": 8,
        "msa_pwa_dim_head": 16,
        "layerscale_output": True,  # note this appears to be custom lucidrains addition to algorithm
    },
    "template_embedder_kwargs": {
        "dim_template_feats": DEFAULT_DIM_TEMPLATE_FEATS,
        "dim": 32,  # reduced from 64
        "pairformer_stack_depth": 2,
        "pairwise_block_kwargs": DEFAULTS_PAIRWISE_BLOCK,
        "layerscale_output": True,  # note this appears to be custom lucidrains addition to algorithm
        "checkpoint": True,
        # "checkpoint_segments": 2,
    },
    "pairformer_stack_kwargs": {
        "depth": 12,
        "checkpoint": True,
        # "checkpoint_segments": 3*12,
        # ----
        "pairwise_block_kwargs": DEFAULTS_PAIRWISE_BLOCK,
        "recurrent_depth": 1,
        "dropout_row_prob": 0.25,
        "pair_bias_attn_dim_head": 16,  # to get 8*24==192
        "pair_bias_attn_heads": 12,  # TODO: original value is 16
        "pair_bias_attn_kwargs": {
            "legacy_gating": True,  # to match AF3, this should be False
            "query_bias": True,  # okay as is, same as AF3
        }
    },
}

DECODER_DEFAULT_SETTINGS = {
    "dim_single": 192,
    "dim_single_decoder": 192,
    "dim_pairwise": 64,
    "dim_input_sequence": utils.NUM_CLASSES,
    "decoder_layer_params": {
        "attend_to_pairwise": True,
        "edge_heads": 4,
        "node_heads": 8,
        "self_heads": 8,
        "dim_edge_head": 16,
        "dim_node_head": 24,
        "dim_self_head": 24,
        "self_attention_dropout": 0.25,
        "node_attention_dropout": 0.25,
        "edge_attention_dropout": 0.25,
    },
    "depth": 6,
    "checkpoint": True,
    "checkpoint_segments": 2*6,
}

# Adam as used in AF3 paper but without learning rate scheduling for now
OPTIMIZER_DEFAULT_SETTINGS = dict(
    lr=3e-4,
    # lr=1.8e-3,
    # betas=(0.9, 0.95),
    # eps=1e-08
)


# LR scheduler from alphafold3_pytorch
# https://github.com/lucidrains/alphafold3-pytorch/blob/b766fcb717b602395c677eb33f8714765d500bb4/alphafold3_pytorch/trainer.py#L262C1-L271C33
def default_lambda_lr_fn(steps):
    # 1000 step warmup

    if steps < 1000:
        return steps / 1000

    # decay 0.95 every 5e4 steps

    steps -= 1000
    return 0.95 ** (steps / 5e4)


class InputFeatureEmbedder(Module):
    """
    Embedding of input features (simplified version that only considers protein monomers,
    in parts more similar to AlphaFold2 input embedder)
    """
    @typecheck
    def __init__(
        self,
        dim_single=DIM_SINGLE,
        dim_pairwise=DIM_PAIRWISE,
        dim_single_input=DIM_SINGLE_INPUT,
        r_max=BINS_POSITION_ENCODING,
    ):
        super().__init__()
        # for Alg 1 line 2
        self.single_input_to_single_init = LinearNoBias(dim_single_input, dim_single)

        # for Alg 1 line 3
        self.single_input_to_pairwise_init = LinearNoBiasThenOuterSum(dim_single_input, dim_pairwise)

        # for Alg 3 line 10 (or correspondingly AlphaFold2 Alg 4 line 2
        self.r_max = r_max
        # note we only need 2 * r_max + 1 (not + 2) since no other chains are encoded in this version
        self.embed_relative_position_encoding = LinearNoBias(2 * r_max + 1, dim_pairwise)

    @typecheck
    def forward(
        self,
        input_features: features.InputFeatureBatch,
        pair_mask: Bool["b n n"],
    ) -> Tuple[
        Float["b n ds"],  # single_inputs
        Float["b n ds"],  # single_init,
        Float["b n n dp"],  # pairwise_init  # Float["b n n dp"],  # relative_position_encoding
    ]:
        """
        Parameters
        ----------
        input_features:
            Batched model input features prepared with features.batch_features()
        pair_mask:
            Boolean mask for pairwise representations based on different sequence lengths
            (derived from input_features.pos_mask)
        
        Returns
        -------
        single_inputs
            Concatenated single input features (profile, deletion_mean)
        single_init
            Single representation initialization
        pairwise_init
            Pairwise representation initialization
        relative_position_encoding
            Relative position encoding (position in monomer)
        """
        # stack input features
        # 1) Alg. 2: {s_i^inputs} = concat restype, profile, deletion_mean. Here: exclude restype, atom features a_i
        single_inputs = einx.rearrange(
            "b n d, b n -> b n (d + 1)", input_features.profile, input_features.deletion_mean
        )
        # equivalent to:
        # single_inputs = torch.cat(
        #     (input_features.profile, input_features.deletion_mean.unsqueeze(-1)),
        #     dim=-1
        # )

        # 2) Alg. 1 line 2: s_i^init = LinearNoBias(s_i^inputs) ... s_i^init in R^(c_s) (single rep)
        # output: Float["b n ds"]
        single_init = self.single_input_to_single_init(single_inputs)

        # 3) Alg. 1 line 3: z_ij^init = LinearNoBias(s_i^inputs) + LinearNoBias(s_j^input). z_ij^init in R^cz (pair rep)
        # output: Float["b n n dp"]
        pairwise_init_raw = self.single_input_to_pairwise_init(single_inputs)
        pairwise_init = einx.where(
            "b i j, b i j d, -> b i j d", pair_mask, pairwise_init_raw, 0.
        )

        # 4) Alg. 1 (Main) z_ij^init += RelativePositionEncoding({f*} and Algorithm 3 line 4/6
        # compute sequence distances for all pairs
        d_ij = einx.subtract(
            "b i, b j -> b i j", input_features.token_index, input_features.token_index
        )
        
        # Apply clipping - this should actually be enough to define the bin since we are dealing with integers here;
        # Note that original AF2 implementation does not clip - their one_hot function seems to achieve the 
        # same functionality for -32 to +32 range. By using the computation in AF3, we should be able to map to
        # bins directly. Also note that upper end of range is *inclusive* for torch.clip
        d_ij_binned = torch.clip(d_ij + self.r_max, 0, 2 * self.r_max)
        
        # turn into one-hot encoding
        a_ij = one_hot(d_ij_binned, num_classes=2 * self.r_max + 1).float()

        # mask missing values to be more explicit (one hot should be blanked out then in multiplication)
        # create pair mask (currently: create this outside of function to avoid recreation?)
        # pair_mask = einx.logical_and(
        #     "b i, b j -> b i j", input_features.pos_mask, input_features.pos_mask
        # )

        a_ij_masked = einx.where(
            "b i j, b i j d, -> b i j d", pair_mask, a_ij, 0.
        )
        
        # apply linear layer to *binned* distance to obtain final relative position encoding;
        # note that this will be not be added to pairwise_init here but in outer calling function
        # Output: Float["b n n dp"]
        # changed: add relative position encoding right away to avoid returning explicit
        # pairwise tensors (increased memory usage)
        relative_position_encoding = self.embed_relative_position_encoding(a_ij_masked)
        pairwise_init = pairwise_init + relative_position_encoding

        return single_inputs, single_init, pairwise_init  #, relative_position_encoding


# alphafold3-pytorch implementation of MSA module adapted for our setting
# class MSAModuleCustom(Module):
#     """ Algorithm 8 """
#
#     def __init__(
#         self,
#         *,
#         # dim_single = 384,
#         dim_single_input=DIM_SINGLE_INPUT,  # changed to use {s_i^inputs}
#         dim_pairwise=DIM_PAIRWISE,
#         depth=MSA_LAYER_DEPTH,
#         dim_msa=DIM_MSA,
#         dim_msa_input=DIM_MSA_INPUT,
#         outer_product_mean_dim_hidden=MSA_DIM_OUTER_PRODUCT_HIDDEN,
#         msa_pwa_dropout_row_prob=MSA_PWA_DROPOUT_ROW_PROB,
#         msa_pwa_heads=MSA_PWA_HEADS,
#         msa_pwa_dim_head=MSA_PWA_DIM_HEAD,
#         pairwise_block_kwargs: dict | None = None,
#         max_num_msa: int | None = MSA_NUM_SAMPLES,
#         layerscale_output: bool = True
#     ):
#         super().__init__()
#
#         self.max_num_msa = default(max_num_msa, float('inf'))  # cap the number of MSAs, will do sample without replacement if exceeds
#
#         # removed else case
#         self.msa_init_proj = LinearNoBias(dim_msa_input, dim_msa) # if exists(dim_msa_input) else nn.Identity()
#         self.dim_msa_input = dim_msa_input  # store depth for checking in forward function
#
#         # changed: use actual input s_i^input
#         # self.single_to_msa_feats = LinearNoBias(dim_single, dim_msa)
#         self.single_to_msa_feats = LinearNoBias(dim_single_input, dim_msa)
#
#         layers = ModuleList([])
#
#         for _ in range(depth):
#             msa_pre_ln = partial(PreLayerNorm, dim=dim_msa)
#
#             outer_product_mean = OuterProductMean(
#                 dim_msa=dim_msa,
#                 dim_pairwise=dim_pairwise,
#                 dim_hidden=outer_product_mean_dim_hidden
#             )
#
#             msa_pair_weighted_avg = MSAPairWeightedAveraging(
#                 dim_msa=dim_msa,
#                 dim_pairwise=dim_pairwise,
#                 heads=msa_pwa_heads,
#                 dim_head=msa_pwa_dim_head,
#                 dropout=msa_pwa_dropout_row_prob,
#                 dropout_type='row'
#             )
#
#             msa_transition = Transition(dim=dim_msa)
#
#             pairwise_block = PairwiseBlock(
#                 dim_pairwise=dim_pairwise,
#                 **(pairwise_block_kwargs if pairwise_block_kwargs is not None else {})
#             )
#
#             layers.append(ModuleList([
#                 outer_product_mean,
#                 msa_pair_weighted_avg,
#                 msa_pre_ln(msa_transition),
#                 pairwise_block
#             ]))
#
#         self.layers = layers
#
#         self.layerscale_output = Parameter(torch.zeros(dim_pairwise)) if layerscale_output else 1.
#
#     @typecheck
#     def forward(
#         self,
#         *,
#         # single_repr: Float['b n ds'],  # replace with actual single inputs (cf. line below)
#         single_inputs: Float['b n d'],
#         pairwise_repr: Float['b n n dp'],
#         msa: Float['b s n dm'],
#         mask: Bool['b n'] | None = None,
#         pairwise_mask: Bool['b n n'] | None = None,
#         msa_mask: Bool['b s'] | None = None,
#     ) -> Float['b n n dp']:
#
#         batch, num_msa, device = *msa.shape[:2], msa.device
#         assert msa.shape[-1] == self.dim_msa_input
#
#         # sample without replacement
#
#         # TODO: could be replaced with new features.sample_sequences() function derived from the following code
#         if num_msa > self.max_num_msa:
#             # sample random numbers
#             rand = torch.randn((batch, num_msa), device=device)
#
#             # effectively move masked entries (no sequence) to end of top-k ranking, so they are selected last
#             if exists(msa_mask):
#                 # inverted mask -> fill where no sequence present
#                 rand.masked_fill_(~msa_mask, max_neg_value(msa))
#
#             # retrieve indices, masked out entries will be retrieved last
#             indices = rand.topk(self.max_num_msa, dim=-1).indices
#
#             # select reduced alignment (will include padded sequences if not enough seqs available)
#             msa = einx.get_at('b [s] n dm, b sampled -> b sampled n dm', msa, indices)
#
#             # also sub-select MSA mask (in cases with not enough sequences, will include "False" entries at end)
#             if exists(msa_mask):
#                 msa_mask = einx.get_at('b [s], b sampled -> b sampled', msa_mask, indices)
#         # account for no msa
#
#         if exists(msa_mask):
#             # one bool entry per sample in batch if MSA present or not
#             has_msa = reduce(msa_mask, 'b s -> b', 'any')
#
#         # process msa
#
#         # project MSA into single-pos representation (Float["b s n dmsa"])
#         msa = self.msa_init_proj(msa)
#
#         # single_msa_feats = self.single_to_msa_feats(single_repr)
#
#         # project, reshape and add to MSA representation
#         single_msa_feats = self.single_to_msa_feats(single_inputs)
#         msa = rearrange(single_msa_feats, 'b n d -> b 1 n d') + msa
#
#         # mask entries where no sequence is present - not strictly necessary probably due to handling inside functions
#         # (as broadcasted addition in line above will also affect empty seqs)
#         if exists(msa_mask):
#             # TODO: replace this with multiplication of mask?
#             msa = einx.where(
#                 "b sampled, b sampled n dmsa, -> b sampled n dmsa",
#                 msa_mask, msa, 0.
#             )
#
#         for (
#             outer_product_mean,
#             msa_pair_weighted_avg,
#             msa_transition,
#             pairwise_block
#         ) in self.layers:
#
#             # communication between msa and pairwise rep;
#             # note this creates pairwise mask internally in function, however, bias of to_pairwise_repr will
#             # add constant to zeroed out outer product mean - this then shows up in pairwise_repr
#             pairwise_repr = outer_product_mean(msa, mask=mask, msa_mask=msa_mask) + pairwise_repr
#
#             # mask out constant terms introduced by bias in outer_product_mean
#             # TODO: ultimately, masking inside outer_product_mean function should be *after* to_pairwise_repr,
#             # so this extra masking step is not needed here
#             pairwise_repr = pairwise_repr * rearrange(pairwise_mask, "b i j -> b i j 1")
#
#             # communication from pairwise representation to MSA; this respects masked sequences,
#             # but not masked positions, so apply mask in second step (appears to be due to folding of j axis onto i,
#             # which is not excluding j that are not covered by position mask but set to small value - cf. masked_fill)
#             msa = msa_pair_weighted_avg(msa=msa, pairwise_repr=pairwise_repr, mask=mask) + msa
#             # apply pos mask to clean up (msa shape: b s n d)
#             msa = msa * rearrange(mask, "b n -> b 1 n 1")
#
#             # no extra masking needed after transition
#             msa = msa_transition(msa) + msa
#
#             # pairwise block.
#             # note that triangle multiplication respects mask, but triangle attention does not (cf. line below)
#             pairwise_repr = pairwise_block(pairwise_repr=pairwise_repr, mask=mask)
#
#             # need to blank out pairwise_repr after here, since attention does not mask in "primary" row position, only
#             # in secondary/tertiary position
#             pairwise_repr = pairwise_repr * rearrange(pairwise_mask, "b i j -> b i j 1")
#
#         if exists(msa_mask):
#             pairwise_repr = einx.where(
#                 'b, b ..., -> b ...',
#                 has_msa, pairwise_repr, 0.
#             )
#
#         return pairwise_repr * self.layerscale_output


class Encoder(Module):
    """
    Monomer protein sequence encoder based on simplified AlphaFold3 "trunk" part of network
    Implementation derived from https://github.com/lucidrains/alphafold3-pytorch/blob/main/alphafold3_pytorch/alphafold3.py
    """

    @typecheck
    def __init__(
        self,
        *,
        dim_single=DIM_SINGLE,
        dim_pairwise=DIM_PAIRWISE,
        dim_single_input=DIM_SINGLE_INPUT,
        msa_module_kwargs: dict | None = None,
        pairformer_stack_kwargs: dict | None = None,
        template_embedder_kwargs: dict | None = None,
        detach_when_recycling: bool = True,
        force_templates_to_loss: bool = False,
        # msa_module_kwargs: dict = dict(
        #     max_num_msa=MSA_NUM_SAMPLES,
        #     depth=MSA_LAYER_DEPTH,
        #     dim_msa=DIM_MSA,
        #     dim_msa_input=DIM_MSA_INPUT,
        #     dim_additional_msa_feats=DIM_ADDITIONAL_MSA_FEATS,
        #     outer_product_mean_dim_hidden=MSA_DIM_OUTER_PRODUCT_HIDDEN,
        #     msa_pwa_dropout_row_prob=MSA_PWA_DROPOUT_ROW_PROB,
        #     msa_pwa_heads=MSA_PWA_HEADS,
        #     msa_pwa_dim_head=MSA_PWA_DIM_HEAD,
        #     pairwise_block_kwargs=dict(),
        #     layerscale_output=True,  # TODO: revisit if this makes sense
        #     checkpoint=True,
        # ),
        # pairformer_stack: dict = dict(
        #     depth=PAIRFORMER_DEPTH,
        #     pair_bias_attn_dim_head=PAIRFORMER_DIM_HEAD,
        #     pair_bias_attn_heads=PAIRFORMER_HEADS,
        #     dropout_row_prob=PAIRFORMER_DROPOUT,
        #     pairwise_block_kwargs=dict(),
        #     checkpoint=True,
        # ),
        # checkpoint_trunk_pairformer=False,
    ):
        super().__init__()

        if msa_module_kwargs is None:
            msa_module_kwargs = {}

        if pairformer_stack_kwargs is None:
            pairformer_stack_kwargs = {}

        if template_embedder_kwargs is None:
            template_embedder_kwargs = {}

        # store params passed to constructor
        self.dim_single = dim_single
        self.dim_pairwise = dim_pairwise

        # instantiate different model components
        self.input_embedder = InputFeatureEmbedder(
            dim_single=dim_single,
            dim_pairwise=dim_pairwise,
            dim_single_input=dim_single_input
        )

        self.template_embedder = TemplateEmbedder(
            # dim_template_feats=dim_template_feats,
            # dim=dim_template_model,
            dim_pairwise=dim_pairwise,
            **template_embedder_kwargs
        )
        self.force_templates_to_loss = force_templates_to_loss
        self.dim_template_feats = template_embedder_kwargs.get(
            "dim_template_feats", DEFAULT_DIM_TEMPLATE_FEATS
        )

        self.msa_module = MSAModule(
            dim_single=dim_single,
            dim_pairwise=dim_pairwise,
            # disable sequence subsampling, we will do this ourselves based on dense representation MSA
            **{
                **msa_module_kwargs,
                "max_num_msa": None,
            }
        )

        # instead record number of sequences to sample here
        self._max_num_msa = msa_module_kwargs.get("max_num_msa", MSA_NUM_SAMPLES)

        # pairformer
        self.pairformer = PairformerStack(
            dim_single=dim_single,
            dim_pairwise=dim_pairwise,
            **pairformer_stack_kwargs
        )

        # self.checkpoint_trunk_pairformer = checkpoint_trunk_pairformer

        # single representation recycling (Alg. 1 line 8)
        self.detach_when_recycling = detach_when_recycling

        self.recycle_single = Sequential(
            LayerNorm(dim_single),
            LinearNoBias(dim_single, dim_single)
        )

        # pair representation recycling (Alg. 1 line 11)
        self.recycle_pairwise = Sequential(
            LayerNorm(dim_pairwise),
            LinearNoBias(dim_pairwise, dim_pairwise)
        )

    @typecheck
    def forward(
        self,
        input_features: features.InputFeatureBatch,
        num_recycling_steps: int = 1,
        max_num_msa: int | None = None
    ):
        """
        # TODO: document parameters
        # TODO: add return type
        """
        # prepare masks to handle different input sequence lengths
        single_mask = input_features.pos_mask
        pair_mask = einx.logical_and(
            "b i, b j -> b i j", input_features.pos_mask, input_features.pos_mask
        )

        # embed input features, and also create relative position encoding
        # single_inputs, single_init, pairwise_init, relative_position_encoding = self.input_embedder(
        single_inputs, single_init, pairwise_init = self.input_embedder(
            input_features, pair_mask=pair_mask
        )

        # add relative position encoding to pairwise representation initialization
        # changed: already add this inside input embedder to potentially save memory
        # pairwise_init = pairwise_init + relative_position_encoding

        single = pairwise = None

        for current_step in range(num_recycling_steps):
            if torch.cuda.is_available():
                cuda_allocated = torch.cuda.memory_allocated() / 1024 ** 2
                logger.info(
                    f"recycling iteration {current_step} before detach, {cuda_allocated} MB allocated"
                )

            # if representations exist from previous iteration, apply recycling operationg
            if exists(single):
                # contributes to Alg. 1 line 11
                if self.detach_when_recycling:
                    single = single.detach()

                recycled_single = self.recycle_single(single)
            else:
                recycled_single = 0.

            if exists(pairwise):
                # contributes to Alg. 1 line 8
                if self.detach_when_recycling:
                    pairwise = pairwise.detach()

                recycled_pairwise = self.recycle_pairwise(pairwise)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache")
            else:
                recycled_pairwise = 0.

            if torch.cuda.is_available():
                cuda_allocated = torch.cuda.memory_allocated() / 1024 ** 2
                logger.info(
                    f"recycling iteration {current_step} after detach, {cuda_allocated} MB allocated"
                )

            # Alg. 1 line 11
            single = single_init + recycled_single

            # Alg. 1 line 8
            pairwise = pairwise_init + recycled_pairwise

            # handle templates;
            # allow to force empty template for multi-device training syncing (cf. #31)
            if self.force_templates_to_loss and input_features.templates is None:
                batch_size, seq_len = input_features.pos_mask.shape
                logger.info("forcing template contribution to loss")
                templates = torch.zeros(
                    (batch_size, 1, seq_len, seq_len, self.dim_template_feats),
                    dtype=pairwise.dtype,
                    device=pairwise.device,
                )
                template_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=pairwise.device)
            else:
                # otherwise simply pass through current value
                templates = input_features.templates
                template_mask = input_features.template_mask

            if templates is not None:
                pairwise = pairwise + self.template_embedder(
                    templates=templates,
                    template_mask=template_mask,
                    pairwise_repr=pairwise,
                    mask=input_features.pos_mask,
                )

            # process multiple sequence alignment and add to pair representation
            # Alg. 1 line 10;
            # note that alphafold3-pytorch MSAmodule does not add single input features s_i,
            # we run our own version that starts as listed in AF3 paper
            # msa_feat = einx.rearrange(
            #     "b s n d, b s n, b s n -> b s n (d + 1 + 1)",
            #     input_features.msa, input_features.has_deletion, input_features.deletion_value
            # )
            additional_msa_feats = einx.rearrange(
                "b s n, b s n -> b s n (1 + 1)",
                input_features.has_deletion, input_features.deletion_value
            )

            # allow to override number of sequences sampled from MSA (e.g. at inference time)
            if max_num_msa is None:
                msa_num_samples = self._max_num_msa
            else:
                msa_num_samples = max_num_msa

            # perform sequence subsampling outside of MSA module so we can store full MSA in dense representation
            sampled_msa, sampled_msa_mask, sampled_additional_msa_feats = features.sample_sequences_dense(
                msa_num_samples,
                msa=input_features.msa,
                msa_mask=input_features.msa_mask,
                pos_mask=input_features.pos_mask,
                additional_msa_feats=additional_msa_feats,
                num_classes=utils.NUM_CLASSES,
            )

            pairwise = pairwise + self.msa_module(
                single_repr=single,
                pairwise_repr=pairwise,
                msa=sampled_msa,
                mask=input_features.pos_mask,
                msa_mask=sampled_msa_mask,
                additional_msa_feats=sampled_additional_msa_feats
            )

            # get rid of temporary tensors
            del additional_msa_feats
            del sampled_msa
            del sampled_msa_mask
            del sampled_additional_msa_feats

            # process single and pairwise representations with Pairformer
            # pairformer = self.pairformer
            # if should_checkpoint(self, (single, pairwise), "checkpoint_trunk_pairformer"):
            #     pairformer = partial(checkpoint, pairformer, use_reentrant=False)

            single, pairwise = self.pairformer(
                single_repr=single,
                pairwise_repr=pairwise,
                mask=input_features.pos_mask,
            )

            if torch.cuda.is_available():
                cuda_allocated = torch.cuda.memory_allocated() / 1024 ** 2
                logger.info(
                    f"recycling iteration {current_step} end of loop, {cuda_allocated} MB allocated"
                )

        # return representations for use in decoder
        return (
            single * rearrange(input_features.pos_mask, "b i -> b i 1"),
            pairwise * rearrange(pair_mask, "b i j -> b i j 1")
        )


class DecoderLayer(Module):
    """
    Single decoder transformer layer with pair bias
    """
    @typecheck
    def __init__(
        self,
        dim_single=DIM_SINGLE,
        dim_pairwise=DIM_PAIRWISE,
        dim_single_decoder=DIM_SINGLE_DECODER,
        edge_heads=DECODER_EDGE_HEADS,
        node_heads=DECODER_NODE_HEADS,
        self_heads=DECODER_SELF_HEADS,
        dim_edge_head=DECODER_EDGE_HEAD_DIM,
        dim_node_head=DECODER_NODE_HEAD_DIM,
        dim_self_head=DECODER_SELF_HEAD_DIM,
        self_attention_dropout=DECODER_SELF_DROPOUT,
        node_attention_dropout=DECODER_NODE_DROPOUT,
        edge_attention_dropout=DECODER_EDGE_DROPOUT,
        attend_to_pairwise=True,
        legacy_gating=True,
        query_bias=True,
    ):
        """
        # TODO: document parameters
        """
        super().__init__()

        self.dim_single = dim_single
        self.dim_pairwise = dim_pairwise
        self.dim_single_decoder = dim_single_decoder

        # number of heads for different attention types
        # cross attention to encoder representations (single = node, pairwise = edge)
        self.edge_heads = edge_heads
        self.node_heads = node_heads
        self.self_heads = self_heads
        self.dim_edge_head = dim_edge_head
        self.dim_node_head = dim_node_head
        self.dim_self_head = dim_self_head

        self.self_attention_dropout = self_attention_dropout
        self.node_attention_dropout = node_attention_dropout
        self.edge_attention_dropout = edge_attention_dropout

        self.attend_to_pairwise = attend_to_pairwise

        # below: attention and pairwise representation projection to attention maps (2x cross attention single/pairs,
        # 1x self-attention); use separate projection for each to promote adaption to most relevant features for
        # each attention type

        # ------------------------------------------
        # self-attention for autoregression

        # Pre-LayerNorm for self-attention
        self.norm_self = LayerNorm(self.dim_single_decoder)

        # projection for biased self - attention; merge forward and backward information from decoder
        # to use maximal available information by stacking (i.e. 2 * dim_pairwise as input size);
        # re init cf. https://github.com/lucidrains/alphafold3-pytorch/blob/5b944ddf975d02e30e0eb3cd33fd8e8ed92d300f/alphafold3_pytorch/alphafold3.py#L624
        pairwise_repr_to_attn_self_linear = LinearNoBias(self.dim_pairwise * 2, self.self_heads)
        # torch.nn.init.zeros_(pairwise_repr_to_attn_self_linear.weight)

        self.pairwise_repr_to_attn_self = Sequential(
            pairwise_repr_to_attn_self_linear,
            Rearrange('b i j h -> b h i j')
        )

        self.attn_self = Attention(
            dim=self.dim_single_decoder,
            dim_head=self.dim_self_head,
            heads=self.self_heads,
            dropout=self.self_attention_dropout,
            legacy_gating=legacy_gating,
            query_bias=query_bias,
        )

        # ------------------------------------------
        # cross-attention to single representation from encoder

        # Pre-LayerNorm step for node cross-attention
        self.norm_node = LayerNorm(self.dim_single_decoder)

        pairwise_repr_to_attn_node_linear = LinearNoBias(self.dim_pairwise, self.node_heads)
        # torch.nn.init.zeros_(pairwise_repr_to_attn_node_linear.weight)

        self.pairwise_repr_to_attn_node = Sequential(
            pairwise_repr_to_attn_node_linear,
            Rearrange('b i j h -> b h i j')
        )

        # note: this requires dim_single == dim_single_decoder to work due to alphafold3-pytorch implementation
        assert self.dim_single_decoder == self.dim_single, "Current implementation needs same dimensions"
        self.attn_node = Attention(
            dim=self.dim_single_decoder,
            dim_head=self.dim_node_head,
            heads=self.node_heads,
            dropout=self.node_attention_dropout,
            legacy_gating=legacy_gating,
            query_bias=query_bias,
        )

        # ------------------------------------------
        # cross-attention to pairwise representation from encoder
        # projection for cross-attention to edges / pairwise representation itself
        if self.attend_to_pairwise:
            # Pre-LayerNorm for edge cross-attention (used in gate)
            self.norm_edge = LayerNorm(self.dim_single_decoder)

            pairwise_repr_to_attn_edge_linear = LinearNoBias(self.dim_pairwise, self.edge_heads)
            # torch.nn.init.zeros_(pairwise_repr_to_attn_edge_linear.weight)
            self.pairwise_repr_to_attn_edge = Sequential(
                pairwise_repr_to_attn_edge_linear,
                Rearrange("b i j h -> b h i j")
            )

            self.pairwise_repr_to_values_edge = Sequential(
                LinearNoBias(self.dim_pairwise, self.edge_heads * self.dim_edge_head),
                Rearrange("b i j (h d) -> b h i j d", h=self.edge_heads)
            )

            # use conventional Dropout for edge attention, similar to alphafold3_pytorch
            # implementation of AttentionPairBias (also cf. commented out version of
            # rowwise Dropout in self.edge_attn_to_out)
            self.dropout_edge_attn = torch.nn.Dropout(self.edge_attention_dropout)

            # note that gate projects from decoder sequence representation to edge attention inner dimension
            self.single_repr_decoder_to_gate = Sequential(
                LinearNoBias(self.dim_single_decoder, self.edge_heads * self.dim_edge_head),
                # reshape to output of edge -> node summation... order used here follows
                # MSAPairWeightedAveraging but unclear why this would be beneficial swapping heads and sequences
                Rearrange("b s n (h d) -> b h s n d", h=self.edge_heads)
            )

            # projection from stacked heads to output in decoder representation space;
            # following MSAPairWeightedAveraging, use LinearNoBias; following alphafold3_pytorch;
            # do not initialize weights
            self.edge_attn_to_out = Sequential(
                Rearrange("b h s n d -> b s n (h d)"),
                LinearNoBias(self.edge_heads * self.dim_edge_head, self.dim_single_decoder),
                # TODO: for now do not use rowwise dropout, this raises some questions
                # TODO: about a) column permutation b) autoregressive model... instead
                # TODO use attention dropout as is used in alphafold3_pytorch AttentionPairBias
                # TODO: via Attention/Attend classes
                # Dropout(self.edge_attention_dropout, dropout_type="row")
            )
        else:
            self.norm_edge = None
            self.pairwise_repr_to_attn_edge = None
            self.pairwise_repr_to_values_edge = None
            self.dropout_edge_attn = None
            self.single_repr_decoder_to_gate = None
            self.edge_attn_to_out = None

    @typecheck
    def forward(
        self,
        *,
        single_repr_decoder: Float["b s n dsd"],
        seq_order: Int["b s n"],
        single: Float["b n ds"],
        pairwise: Float["b n n dp"],
        pos_mask: Bool["b n"] | None,
        # seq_mask: Bool["b s"] | None,
    ):
        """
        # TODO: document parameters
        # TODO: refactor, split different attention types into individual methods or individual modules
        """
        n = pairwise.shape[-2]
        s = seq_order.shape[-2]

        # pack sequences into batch dimension, so we can supply tensor of shape "b i d" to Attention class
        # (cf. TriangleAttention implementation)
        single_repr_decoder_packed, unpack_single_repr_decoder = pack_one(single_repr_decoder, "* n dsd")

        # ------------------------------------------------------------------------------------------
        # FIRST, autoregressive self-attention (follow ordering in DeepMind tutorial Alg. 8);
        # project pair weights, merge forward and backward edges from decoder by stacking tensor to its transpose
        # before projecting down to attention bias
        pairwise_stacked = einx.rearrange(
            "b i j dp, b j i dp -> b i j (dp + dp)", pairwise, pairwise
        )
        bias_self = self.pairwise_repr_to_attn_self(pairwise_stacked)  # output shape: b h n n

        # prepare attention bias including causal mask in appropriate shape;
        # first, reorder according to decoding order along both axis
        # - need to stack so each sequence has its own bias ordering!

        # cannot use due to incompatibility with gradient checkpointing
        # bias_self_reordered_old = einx.get_at(
        #     "b h [i j], b s i_o, b s j_o -> b s h i_o j_o", bias_self, seq_order, seq_order
        # )

        # expand bias to target dimensions by creating one layer per sequence in each batch
        bias_self_repeated = repeat(
            bias_self, "b h i j -> b s h i j", s=s
        )

        seq_order_idx_i = repeat(
            seq_order, "b s io -> b s h io n", h=bias_self.shape[-3], n=n,
        )

        seq_order_idx_j = repeat(
            seq_order, "b s io -> b s h n io", h=bias_self.shape[-3], n=n,
        )

        bias_self_reordered = bias_self_repeated.gather(
            -1, seq_order_idx_j
        ).gather(
            -2, seq_order_idx_i
        )

        # note assertion needs weight init disabled to give meaningful check
        # assert (bias_self_reordered_old == bias_self_reordered).all()

        # then, apply causal mask *and* position mask by setting to -inf; position masking is for defensiveness
        # only as these positions j should be already covered by causal mask for all i that are inside pos_mask
        causal_mask = torch.tril(torch.ones(n, n, device=bias_self_reordered.device)).bool()
        bias_self_reordered = einx.where(
            "i j, b s h i j, ", causal_mask, bias_self_reordered, -torch.inf
        )
        bias_self_reordered = einx.where(
            "b j, b s h i j,", pos_mask, bias_self_reordered, -torch.inf
        )

        # pad bias to accommodate for additional BOS token used by decoder along dimensions i and j
        bias_self_reordered = pad(bias_self_reordered, (0, 1, 0, 1), value=-torch.inf)
        # add valid values on dimension i to avoid nan issues by copying values from previous (actual last) i,
        # (this part of tensor corresponding to last pos/EOS will be ignored in loss anyways)
        bias_self_reordered[:, :, :, -1, :] = bias_self_reordered[:, :, :, -2, :]

        # also pack bias term (so batch and sequences are in one dimension), resulting shape is "(b s) h n n"
        bias_self_packed, unpack_bias_self = pack_one(bias_self_reordered, "* h i j")

        # note: do not forward pos_mask into function, as this is included in attention bias via bias_self_packed;
        # apply pre-LayerNorm to decoder single representation input
        single_repr_decoder_packed = single_repr_decoder_packed + self.attn_self(
            self.norm_self(single_repr_decoder_packed),
            attn_bias=bias_self_packed,
        )

        # ------------------------------------------------------------------------------------------
        # SECOND, attend to full single representation via pair-biased cross-attention

        # project pair bias weights; unlike for self-attention, we can use both forward and reverse edges
        # i->j and j->i separately, no need to merge
        bias_node = self.pairwise_repr_to_attn_node(pairwise)

        # reorder on query axis (i) on a per-decoder-sequence basis; unlike self-attention,
        # keep key axis (j) indexing the encoder single representation as is

        # cannot use get_at due to incompatibility with checkpointing
        # bias_node_reordered_old = einx.get_at(
        #     "b h [i] j, b s i_o -> b s h i_o j", bias_node, seq_order
        # )

        bias_node_repeated = repeat(
            bias_node, "b h i j -> b s h i j", s=s
        )

        seq_order_idx_i_node = repeat(
            seq_order, "b s io -> b s h io n", h=bias_node.shape[-3], n=n,
        )

        bias_node_reordered = bias_node_repeated.gather(
            -2, seq_order_idx_i_node
        )

        # note assertion needs weight init disabled to give meaningful check
        # assert (bias_node_reordered == bias_node_reordered_old).all()

        # apply masking based on sequence position; unlike for self-attention where positions j
        # are already masked by causal mask, applying the mask is crucial here
        bias_node_reordered = einx.where(
            "b j, b s h i j,", pos_mask, bias_node_reordered, -torch.inf
        )

        # pad bias to accommodate for additional BOS token used by decoder along dimensions i and j (note that strictly
        # we would not need to pad on context positions j, but alphafold3-pytorch attention implementation checks
        # for square shape to use full attention instead of windowed attention, so reformat input accordingly)
        bias_node_reordered = pad(bias_node_reordered, (0, 1, 0, 1), value=-torch.inf)

        # again, copy last real attention bias row to last tensor row (EOS) to provide valid values; but will
        # be ignored in output/loss
        bias_node_reordered[:, :, :, -1, :] = bias_node_reordered[:, :, :, -2, :]

        # pack bias for attention class
        bias_node_packed, bias_node_packed_shape = pack_one(bias_node_reordered, "* h i j")

        # repeat single representation for use as context sequence
        num_seqs = single_repr_decoder.shape[1]
        single_repeated = repeat(
            single, "b ... -> (b repeat) ...", repeat=num_seqs
        )
        # also apply padding so context length is same as query length (required by alphafold3-pytorch implementation);
        # this position will never be actually attended to in calculation as last column in bias above is set to -inf
        single_repeated = pad(single_repeated, (0, 0, 0, 1), value=0.)

        # compute attention and add to representation; apply pre-LN before
        single_repr_decoder_packed = single_repr_decoder_packed + self.attn_node(
            seq=self.norm_node(single_repr_decoder_packed),
            context=single_repeated,
            attn_bias=bias_node_packed,
        )

        # pack back into shape "b s n dsd", we do not need this particular shape for
        # our own implementation of edge cross attention
        single_repr_decoder = unpack_single_repr_decoder(single_repr_decoder_packed)

        # ------------------------------------------------------------------------------------------
        # THIRD, attend to pairwise representation (i.e. edge features); this is simplified to use
        # same type of attention as MSAPairWeightedAveraging in encoder
        if self.attend_to_pairwise:
            # project edge bias for each head, dimensions "b h n n"
            bias_edge = self.pairwise_repr_to_attn_edge(pairwise)

            # apply mask in j dimension
            bias_edge = bias_edge.masked_fill(
                rearrange(~pos_mask, "b j -> b 1 1 j"),
                max_neg_value(bias_edge)
            )

            # compute weights for folding edges ij onto i, taking position mask into account for valid j;
            # note this will leave small value in unspecified positions (like in encoder)
            weights_edge = bias_edge.softmax(dim=-1)

            # apply dropout (also see comments in constructor)
            weights_edge = self.dropout_edge_attn(weights_edge)

            # compute values for each edge, then fold back onto nodes
            values_edge = self.pairwise_repr_to_values_edge(pairwise)

            # fold edges back onto each node, this corresponds to sum_j{w_ij^h * v_ij^h};
            # output shape b h n d, multiplication with values enforces mask;
            edges_to_node = einsum(
                weights_edge, values_edge, "b h i j, b h i j d -> b h i d"
            )

            # expand and reorder this node sum on a per-sequence basis (on index i)
            # for adding to decoder single representation

            # cannot use get_at due to incompatibility with checkpointing
            # edges_to_node_reordered_old = einx.get_at(
            #     "b h [i] d, b s i_o -> b h s i_o d", edges_to_node, seq_order
            # )

            edges_to_node_repeated = repeat(
                edges_to_node, "b h i d -> b h s i d", s=s
            )

            seq_order_idx_i_edge = repeat(
                seq_order, "b s io -> b h s io d",
                h=edges_to_node_repeated.shape[-4],
                d=edges_to_node_repeated.shape[-1]
            )

            edges_to_node_reordered = edges_to_node_repeated.gather(
                -2, seq_order_idx_i_edge
            )

            # note assertion needs weight init disabled to give meaningful check
            # assert (edges_to_node_reordered == edges_to_node_reordered_old).all()

            # add padding at end to accommodate for BOS token
            edges_to_node_reordered = pad(edges_to_node_reordered, (0, 0, 0, 1), value=0.)

            # compute gate from decoder sequences, output shape must match edges_to_node_reordered
            # (gate is on positions i!); apply pre-LN to decoder single representation here as this
            # is the only place it is used in edge attention
            gates = self.single_repr_decoder_to_gate(
                self.norm_edge(single_repr_decoder)
            ).sigmoid()

            # apply gate, this corresponds to g_si^h * sum_j{w_ij^h * v_ij^h}
            edges_to_node_reordered = edges_to_node_reordered * gates

            # project heads down to single decoder representation dimension and add to single representation,
            # dropout is applied at this stage
            edges_out = self.edge_attn_to_out(edges_to_node_reordered)
            single_repr_decoder = edges_out + single_repr_decoder

        return single_repr_decoder


class PairBiasOnlyAttention(Module):
    @typecheck
    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int,
        dim_pairwise: int,
        dropout: float,
        autoregressive: bool,
        context_type: Literal["single_decoder", "single_encoder", "pairwise_encoder"] = "single_decoder",
        dim_context: int | None = None,
    ):
        """
       # TODO: document parameters
       """
        super().__init__()

        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.dim_pairwise = dim_pairwise
        self.dropout = dropout

        # automatically infer context dim if not specified
        if dim_context is None:
            self.dim_context = self.dim
        else:
            self.dim_context = dim_context

        self.dim_inner = dim_head * heads

        # store whether we have to use causal attention or not
        self.autoregressive = autoregressive
        self.context_type = context_type

        # Pre-LN of main sequence (handle inside this function for compatibility with older code)
        self.norm = LayerNorm(self.dim)

        # projection of pairwise representation to attention weights; assume this already has
        # pre-LN applied globally once in outer Decoder module
        self.pairwise_repr_to_attn = Sequential(
            LinearNoBias(self.dim_pairwise, self.heads),
            Rearrange('b i j h -> b h i j')
        )

        # projection of context sequence to values; note we could in principle
        # all context sequence to have a different dim than main seq
        if self.context_type == "single_decoder":
            rearr_cmd = "b s n (h d) -> b h s n d"
        elif self.context_type == "single_encoder":
            rearr_cmd = "b n (h d) -> b h n d"
        elif self.context_type == "pairwise_encoder":
            rearr_cmd = "b i j (h d) -> b h i j d"
        else:
            raise ValueError("Invalid option for context_type")

        self.to_values = Sequential(
            LinearNoBias(self.dim_context, self.dim_inner),
            Rearrange(rearr_cmd, h=self.heads)
        )

        self.to_gates = Sequential(
            LinearNoBias(self.dim, self.dim_inner),
            Rearrange("b s n (h d) -> b h s n d", h=self.heads)
        )

        # projection from values to output
        self.to_out = Sequential(
            Rearrange("b h s n d -> b s n (h d)"),
            LinearNoBias(self.dim_inner, self.dim)
        )

        # dropout (rowwise, will be permuted by seq order)
        self.dropout = Dropout(self.dropout)

    def forward(
        self,
        *,
        seq: Float["b s n d"],
        pairwise: Float["b n n dp"],
        seq_order: Int["b s n"],
        pos_mask: Bool["b n"] | None,
        context: Float["b s n d"] | Float["b n n dp"] | None = None,
    ):
        n = pairwise.shape[-2]
        s = seq.shape[-3]

        # apply pre-LN to sequence
        seq = self.norm(seq)

        # if no context seq given, default to self-attention;
        # if context is explicitly passed in, assume it already has pre-LN applied
        if context is None:
            context = seq

        # for autoregression, symmetrize the pairwise representation first so no information is lost;
        # symmetrize by simple addition like in alphafold3_pytorch instead of stacking as in original decoder
        if self.autoregressive:
            pairwise = pairwise + rearrange(pairwise, "b i j ... -> b j i ...")

        #  project to bias terms per head and residue pair (shape b h n n)
        b = self.pairwise_repr_to_attn(pairwise)

        # then apply pos mask (along j dimension); note we can do this before reordering below as
        # seq_order must respect pos_mask
        b = einx.where(
            "b j, b h i j,", pos_mask, b, max_neg_value(b)
        )

        # repeat bias on a per-sequence basis so it can be reordered for decoding order
        b_rep = repeat(
            b, "b h i j -> b s h i j", s=s
        )

        # will reorder in first sequence dimension in any case (query seq)
        seq_order_idx_i = repeat(
            seq_order, "b s io -> b s h io n", h=self.heads, n=n,
        )

        if self.autoregressive:
            # in case of self-attention, also reorder context seq
            seq_order_idx_j = repeat(
                seq_order, "b s io -> b s h n io", h=self.heads, n=n,
            )

            # reorder along both dimensions based on random decoding order
            b_rep_ord = b_rep.gather(
                -1, seq_order_idx_j
            ).gather(
               -2, seq_order_idx_i
            )

            # then apply causal mask for autoregression
            causal_mask = torch.tril(torch.ones(n, n, device=b_rep.device)).bool()
            b_rep_ord = einx.where(
                "i j, b s h i j, ", causal_mask, b_rep_ord, max_neg_value(b_rep_ord)
            )

            # pad bias to accommodate for additional BOS token used by decoder along dimensions i and j
            b_rep_ord = pad(b_rep_ord, (0, 1, 0, 1), value=max_neg_value(b_rep_ord))
            # add valid values on dimension i to avoid nan issues by copying values from previous (actual last) i,
            # (this part of tensor corresponding to last pos/EOS will be ignored in loss anyways)
            b_rep_ord[:, :, :, -1, :] = b_rep_ord[:, :, :, -2, :]
        else:
            # in case of cross-attention, only reorder query dimension i
            b_rep_ord = b_rep.gather(
                -2, seq_order_idx_i
            )

            # pad bias to accommodate for additional BOS token used by decoder along dimensions i
            # (not j, can keep context at encoder length without BOS token in our own implementation used here)
            b_rep_ord = pad(b_rep_ord, (0, 0, 0, 1), value=max_neg_value(b_rep_ord))
            # again, copy last real attention bias row to last tensor row (EOS) to provide valid values; but will
            # be ignored in output/loss
            b_rep_ord[:, :, :, -1, :] = b_rep_ord[:, :, :, -2, :]

        # turn attn bias into weights
        w = b_rep_ord.softmax(dim=-1)

        # project values
        values = self.to_values(context)

        if self.context_type == "single_decoder":
            # rearr_cmd = "b s n (h d) -> b h s n d"
            einsum_eq = "b s h i j, b h s j d -> b h s i d"
        elif self.context_type == "single_encoder":
            # rearr_cmd = "b n (h d) -> b h n d"
            einsum_eq = "b s h i j, b h j d -> b h s i d"
        elif self.context_type == "pairwise_encoder":
            # rearr_cmd = "b i j (h d) -> b h i j d"
            einsum_eq = "b s h i j, b h i j d -> b h s i d"
            # pad values to match size in i dimension (token values won't be used)
            values = pad(values, (0, 0, 0, 0, 0, 1), value=0.)

        # compute updates based on attention weights, then project down
        weighted_sum = einsum(w, values, einsum_eq)

        # compute gates and push through activation
        gates = self.to_gates(seq).sigmoid()

        # apply gates to output values and combine heads
        assert weighted_sum.shape == gates.shape
        out = self.to_out(weighted_sum * gates)

        # apply row-wise dropout with added permutation by seq_order so dropout
        # mask is actually shared over the same residues

        # compute "standard" row-wise dropout first
        batch, _, col, dim = out.shape
        d = self.dropout(
            out.new_ones((batch, 1, col, dim))
        )

        # repeat and make one copy per sequence (rather than using broadcast as in default)
        d_rep = repeat(d, "b 1 n d -> b s n d", s=s)

        # extend seq_order with "EOS" index at the end of each sequence (last dimension),
        # as seq_order is one shorter than BOS-extended sequence to decode
        seq_order_eos = pad(seq_order, (0, 1), value=n)

        # reorder dropout mask based on seq_order
        seq_order_idx_i_dropout = repeat(
            seq_order_eos, "b s io -> b s io d", d=dim
        )

        d_rep_ord = d_rep.gather(-2, seq_order_idx_i_dropout)

        # apply dropout and return
        return out * d_rep_ord


class DecoderLayerPairBiasOnly(Module):
    @typecheck
    def __init__(
        self,
        dim_single=DIM_SINGLE,
        dim_pairwise=DIM_PAIRWISE,
        dim_single_decoder=DIM_SINGLE_DECODER,
        edge_heads=DECODER_EDGE_HEADS,
        node_heads=DECODER_NODE_HEADS,
        self_heads=DECODER_SELF_HEADS,
        dim_edge_head=DECODER_EDGE_HEAD_DIM,
        dim_node_head=DECODER_NODE_HEAD_DIM,
        dim_self_head=DECODER_SELF_HEAD_DIM,
        self_attention_dropout=DECODER_SELF_DROPOUT,
        node_attention_dropout=DECODER_NODE_DROPOUT,
        edge_attention_dropout=DECODER_EDGE_DROPOUT,
        attend_to_pairwise=True,
    ):
        """
        # TODO: document parameters
        """
        super().__init__()

        self.dim_single = dim_single
        self.dim_pairwise = dim_pairwise
        self.dim_single_decoder = dim_single_decoder

        # number of heads for different attention types
        # cross attention to encoder representations (single = node, pairwise = edge)
        self.edge_heads = edge_heads
        self.node_heads = node_heads
        self.self_heads = self_heads
        self.dim_edge_head = dim_edge_head
        self.dim_node_head = dim_node_head
        self.dim_self_head = dim_self_head

        self.self_attention_dropout = self_attention_dropout
        self.node_attention_dropout = node_attention_dropout
        self.edge_attention_dropout = edge_attention_dropout

        self.attend_to_pairwise = attend_to_pairwise

        # causal self-attention to decoded sequence
        self.attn_self = PairBiasOnlyAttention(
            dim=self.dim_single_decoder,
            dim_head=self.dim_self_head,
            heads=self.self_heads,
            dim_pairwise=self.dim_pairwise,
            dropout=self.self_attention_dropout,
            autoregressive=True,
            context_type="single_decoder"
        )

        # full cross-attention to single representation ("nodes")
        self.attn_node = PairBiasOnlyAttention(
            dim=self.dim_single_decoder,
            dim_context=self.dim_single,
            dim_head=self.dim_node_head,
            heads=self.node_heads,
            dim_pairwise=self.dim_pairwise,
            dropout=self.node_attention_dropout,
            autoregressive=False,
            context_type="single_encoder",
        )

        # full cross-attention to pair representation ("edges")
        self.attn_edge = PairBiasOnlyAttention(
            dim=self.dim_single_decoder,
            dim_context=self.dim_pairwise,
            dim_head=self.dim_edge_head,
            heads=self.edge_heads,
            dim_pairwise=self.dim_pairwise,
            dropout=self.edge_attention_dropout,
            autoregressive=False,
            context_type="pairwise_encoder",
        )

    @typecheck
    def forward(
        self,
        *,
        single_repr_decoder: Float["b s n dsd"],
        seq_order: Int["b s n"],
        single: Float["b n ds"],
        pairwise: Float["b n n dp"],
        pos_mask: Bool["b n"] | None,
    ):
        # autoregressive self-attention; pre-LN is applied inside function
        single_repr_decoder = single_repr_decoder + self.attn_self(
            seq=single_repr_decoder,
            pairwise=pairwise,
            pos_mask=pos_mask,
            seq_order=seq_order,
        )

        # cross-attention to encoder single repr
        single_repr_decoder = single_repr_decoder + self.attn_node(
            seq=single_repr_decoder,
            context=single,
            pairwise=pairwise,
            pos_mask=pos_mask,
            seq_order=seq_order,
        )

        # cross-attention to encoder pairwise repr
        single_repr_decoder = single_repr_decoder + self.attn_edge(
            seq=single_repr_decoder,
            context=pairwise,
            pairwise=pairwise,
            pos_mask=pos_mask,
            seq_order=seq_order,
        )

        return single_repr_decoder


class Decoder(Module):
    """
    Monomer protein sequence decoder, following standard encoder-decoder transformer model
    """
    @typecheck
    def __init__(
        self,
        *,
        dim_single=DIM_SINGLE,
        dim_pairwise=DIM_PAIRWISE,
        dim_input_sequence=utils.NUM_CLASSES,
        dim_single_decoder=DIM_SINGLE_DECODER,
        depth=DECODER_LAYER_DEPTH,
        decoder_layer_params: dict | None = None,
        checkpoint: bool = True,
        checkpoint_segments: int = 1,
        pairbias_only_decoder: bool = False,
    ):
        """
        # TODO: document parameters
        """
        super().__init__()

        # type of decoder
        self.pairbias_only_decoder = pairbias_only_decoder

        # dimensions of single and pairwise representations from encoder
        self.dim_single = dim_single
        self.dim_pairwise = dim_pairwise

        # dimensions of one-hot encoded input sequences and internal
        # single representation in decoder
        self.dim_input_sequence = dim_input_sequence
        self.dim_single_decoder = dim_single_decoder

        # number of decoder layers/blocks
        self.depth = depth

        # LayerNorm of single and pairwise representations
        self.single_norm = LayerNorm(self.dim_single)
        self.pairwise_norm = LayerNorm(self.dim_pairwise)

        # initial embedding projection
        self.embed = LinearNoBias(self.dim_input_sequence, self.dim_single_decoder)

        # transformer blocks/layers
        layers = ModuleList([])

        for _ in range(self.depth):
            if pairbias_only_decoder:
                # pair bias attention only (cf. AF3 MSAPairWeightedAveraging)
                dec_layer = DecoderLayerPairBiasOnly(
                    dim_single=dim_single,
                    dim_pairwise=dim_pairwise,
                    dim_single_decoder=dim_single_decoder,
                    **(decoder_layer_params if decoder_layer_params is not None else {})
                )
            else:
                # full attention with pair bias (cf. AF3 AttentionPairBias)
                dec_layer = DecoderLayer(
                    dim_single=dim_single,
                    dim_pairwise=dim_pairwise,
                    dim_single_decoder=dim_single_decoder,
                    **(decoder_layer_params if decoder_layer_params is not None else {})
                )
            single_pre_ln = partial(PreLayerNorm, dim=dim_single_decoder)
            single_transition = Transition(dim=dim_single_decoder)

            layers.append(ModuleList([
                dec_layer,
                single_pre_ln(single_transition),
            ]))

        self.layers = layers

        self.checkpoint = checkpoint
        self.checkpoint_segments = checkpoint_segments

        # final unembedding projection W_u to compute logits/probabilities; pass through global
        # (rather than just residuals) pre-LayerNorm like in Algorithm 10 of DeepMind transformer tutorial
        self.unembed = Sequential(
            LayerNorm(self.dim_single_decoder),
            LinearNoBias(self.dim_single_decoder, self.dim_input_sequence)
        )

    @typecheck
    def to_layers(
        self,
        *,
        single_repr_decoder: Float["b s n dsd"],
        seq_order: Int["b s n"],
        single: Float["b n ds"],
        pairwise: Float["b n n dp"],
        pos_mask: Bool["b n"],
    ) -> Float["b s n dsd"]:
        for (decoder_block, transition) in self.layers:
            # apply current decoder attention layer (cross- and self-attention)
            single_repr_decoder = decoder_block(
                single_repr_decoder=single_repr_decoder,
                seq_order=seq_order,
                single=single,
                pairwise=pairwise,
                pos_mask=pos_mask,
                # seq_mask=seq_mask
            )

            # apply transition
            single_repr_decoder = transition(single_repr_decoder) + single_repr_decoder

        return single_repr_decoder

    @typecheck
    def to_checkpointed_layers(
            self,
            *,
            single_repr_decoder: Float["b s n dsd"],
            seq_order: Int["b s n"],
            single: Float["b n ds"],
            pairwise: Float["b n n dp"],
            pos_mask: Bool["b n"],
    ) -> Float["b s n dsd"]:
        """
        # TODO: document parameters
        # based off on lucidrains' checkpointing code in encoder
        """
        logger.info("running checkpointed decoder")

        inputs = (single_repr_decoder, seq_order, single, pairwise, pos_mask)
        wrapped_layers = []

        def decoder_block_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                single_repr_decoder, seq_order, single, pairwise, pos_mask = inputs
                # note that we pass through single_repr_decoder and add internally
                single_repr_decoder = fn(
                    single_repr_decoder=single_repr_decoder,
                    seq_order=seq_order,
                    single=single,
                    pairwise=pairwise,
                    pos_mask=pos_mask,
                )
                return single_repr_decoder, seq_order, single, pairwise, pos_mask

            return inner

        def transition_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                single_repr_decoder, seq_order, single, pairwise, pos_mask = inputs
                # note that we add transition to representation
                single_repr_decoder = fn(single_repr_decoder) + single_repr_decoder
                return single_repr_decoder, seq_order, single, pairwise, pos_mask

            return inner

        for (decoder_block, transition) in self.layers:
            wrapped_layers.append(decoder_block_wrapper(decoder_block))
            wrapped_layers.append(transition_wrapper(transition))

        single_repr_decoder, *_ = checkpoint_sequential(
            wrapped_layers, self.checkpoint_segments, inputs, use_reentrant=False
        )

        return single_repr_decoder


    @typecheck
    def forward(
        self,
        *,
        single: Float["b n ds"],
        pairwise: Float["b n n dp"],
        seqs: Int["b s n d"],
        pos_mask: Bool["b n"],
        seq_order: Int["b s n"] | None = None,
        keep_logits_decoding_order: bool = False,
        return_embeddings: bool = False,
    ) -> Float["b s n d"]:
        """
        # TODO: document parameters
        """
        # prepare sequence order for decoding, either by random reordering (if no positions of interest given,
        # mostly relevant for training); or by using an externally supplied seq_order that already moved positions
        # of interest to the end (mutant scoring/generation).
        # seq_order is of shape "b s n"
        if seq_order is None:
            seq_order = features.reorder_sequences(seqs, pos_mask)

        # torch.vmap underlying get_at unfortunately not working with checkpointing
        # seqs_reordered_old = einx.get_at(
        #     "b s [i] d, b s iord -> b s iord d", seqs, seq_order
        # )
        seq_order_idx = repeat(seq_order, "b s iord -> b s iord d", d=seqs.shape[-1])
        seqs_reordered = seqs.gather(2, seq_order_idx)
        # assert (seqs_reordered == seqs_reordered_old).all()

        # quick and dirty way to get inverse position mapping, even if not most efficient
        inverse_order = torch.argsort(seq_order)

        # create one-hot encoded BOS token to prepend to sequences
        # (this needs to be in first position, so only do this after applying seq_order)
        bos_tokens = one_hot(
            torch.ones(
                (*seqs_reordered.shape[:2], 1), device=seqs_reordered.device
            ).to(torch.int64) * utils.AA_TO_INDEX[utils.BOS],
            num_classes=utils.NUM_CLASSES,
        )

        seqs_reordered_with_bos = einx.rearrange(
            "b s 1 d, b s n d -> b s (1 + n) d", bos_tokens, seqs_reordered
        )

        # normalize single and pairwise representations (once at beginning, as these do not change in decoder)
        single = self.single_norm(single)
        pairwise = self.pairwise_norm(pairwise)

        # initialize decoder single representation by embedding sequence
        # from *reordered* one hot encoding
        single_repr_decoder = self.embed(seqs_reordered_with_bos.float())

        if should_checkpoint(self, (single_repr_decoder, single, pairwise), "checkpoint"):
            # logger.info("checkpointing: yes")
            to_layers_fn = self.to_checkpointed_layers
        else:
            # logger.info("checkpointing: no")
            to_layers_fn = self.to_layers

        single_repr_decoder = to_layers_fn(
            single_repr_decoder=single_repr_decoder,
            seq_order=seq_order,
            single=single,
            pairwise=pairwise,
            pos_mask=pos_mask,
        )

        # version before introducing checkpointing
        # for (decoder_block, transition) in self.layers:
        #     # apply current decoder attention layer (cross- and self-attention)
        #     single_repr_decoder = decoder_block(
        #         single_repr_decoder=single_repr_decoder,
        #         seq_order=seq_order,
        #         single=single,
        #         pairwise=pairwise,
        #         pos_mask=pos_mask,
        #         # seq_mask=seq_mask
        #     )
        #
        #     # apply transition
        #     single_repr_decoder = transition(single_repr_decoder) + single_repr_decoder

        # finally, unembed sequence to logits (this includes global pre-LN),
        # and restore original sequence order; discard last token which implicitly corresponds to EOS,
        # but we predict fixed length only for now
        logits_reordered = self.unembed(single_repr_decoder[:, :, :-1, :])

        # replaced due to vmap incompatibility with gradient checkpointing
        # logits_old = einx.get_at(
        #     "b s [i] d, b s iord -> b s iord d", logits_reordered, inverse_order
        #
        if keep_logits_decoding_order:
            logits = logits_reordered
        else:
            inverse_order_idx = repeat(
                inverse_order, "b s iord -> b s iord d", d=logits_reordered.shape[-1]
            )
            logits = logits_reordered.gather(2, inverse_order_idx)
        # assert (logits == logits_old).all()

        if return_embeddings:
            embeddings_reordered = self.unembed[0](single_repr_decoder[:, :, :-1, :])
            if keep_logits_decoding_order:
                embeddings = embeddings_reordered
            else:
                inverse_order_idx_emb = repeat(
                    inverse_order, "b s iord -> b s iord d", d=embeddings_reordered.shape[-1]
                )
                embeddings = embeddings_reordered.gather(2, inverse_order_idx_emb)

            # apply pre-unembed layernorm to embeddings
            return logits, seq_order, embeddings
        else:
            return logits, seq_order

    # def score(
    #     self,
    #     *,
    #     seqs: Int["b s n d"],
    #     pos_to_score: Bool["b s n"],
    #     single: Float["b n ds"],
    #     pairwise: Float["b n n dp"],
    #     pos_mask: Bool["b n"],
    # ):
    #     """
    #     Utility for forward method that allows to score probabilities for given positions
    #     in fixed sequences by putting all other positions in prefix; order in each of these
    #     blocks is random and will be returned by this method.
    #
    #     # TODO: document parameters
    #     # TODO: this method is kind of useless, remove
    #     """
    #     # determine sequence order, moving positions to score towards end of sequence
    #     seq_order = features.reorder_sequences(seqs, pos_mask, pos_to_score)
    #
    #     return self.forward(
    #         seqs=seqs, single=single, pairwise=pairwise, pos_mask=pos_mask, seq_order=seq_order
    #     )

    @classmethod
    def _verify_inputs_and_map_seqs(
        cls,
        seqs: Sequence[str],
        single: Float["b n ds"],
        pairwise: Float["b n n dp"],
        pos_mask: Bool["b n"],
    ):
        assert (
                pos_mask.shape[0] == 1
        ), "Method only handles single batch (but single/pair reps for same positions can be stacked)"

        assert (
                single.shape[0] == pairwise.shape[0]
        ), "single and pairwise batch dimensions do not agree"

        assert len(seqs) >= 1, "Must pass at least one sequence"

        # check and map wildtype sequence
        n_pos = pos_mask[0].sum()
        assert len(seqs[0]) == n_pos, "Length of sequence and pos_mask do not agree"
        _, n_padded = pos_mask.shape

        assert len(set(len(seq) for seq in seqs)) == 1, "All sequences must have same length"

        # try to map all sequences
        try:
            seqs_mapped_list = [
                utils.map_sequence(seq) for seq in seqs
            ]
        except KeyError as e:
            raise ValueError("Invalid symbol in input sequence") from e

        # create tensor from mapped WT sequence, and pad to match single/pair representation
        seqs_mapped = torch.tensor(
            seqs_mapped_list, device=single.device
        )

        num_encodings = single.shape[0]

        return seqs_mapped, n_pos, n_padded, num_encodings

    @torch.inference_mode()
    def score_full_probability(
        self,
        seqs: Sequence[str],
        *,
        single: Float["b n ds"],
        pairwise: Float["b n n dp"],
        pos_mask: Bool["b n"],
        num_samples: int = 4,
        batch_size: int = 64,
        first_index: int = 1,
        share_decoding_order_across_encodings: bool = True,
        return_embeddings: bool = False,
        fixed_seq_order: dict | None = None,
    ):
        """
        Compute full autoregressive probability P(x_1) * P(x_2 | x_1) * ... * P(x_n | x_1, ..., x_n-1)
        for a list of sequences, using the same decoding order for each sample across all sequences
        """
        seqs_mapped, n_pos, n_padded, num_encodings = self._verify_inputs_and_map_seqs(
            seqs, single=single, pairwise=pairwise, pos_mask=pos_mask
        )

        # pad and reshape sequences right away to keep code below simpler
        seqs_mapped = rearrange(
            pad(
                seqs_mapped,
                (0, n_padded - n_pos)
            ), "s n -> 1 s n"
        )

        # extract first sequence for computing repeated sequence order
        # (exact sequence does not matter and mask tokens won't be an issue in reorder_sequences())
        first_seq = seqs_mapped[:, [0], :]

        # variables for recording results
        all_scores = []
        all_sequences = []
        all_encoder_indices = []
        all_decoder_indices = []

        all_embeddings = []

        # sequence orders from current call, if fixed order given, pass it through (will not be modified below)
        current_seq_order = {}
        if fixed_seq_order is not None:
            current_seq_order = fixed_seq_order

        # iterate through decoder samples on outermost layer so we can share the decoding order
        # across all samples
        for dec_idx in range(num_samples):
            # determine sequence order for reuse
            if share_decoding_order_across_encodings:
                seq_order_shared = features.reorder_sequences(
                    first_seq, pos_mask, masked_positions=None,
                )
            else:
                seq_order_shared = None

            for enc_idx in range(num_encodings):
                if seq_order_shared is None:
                    seq_order = features.reorder_sequences(
                        first_seq, pos_mask, masked_positions=None,
                    )
                else:
                    seq_order = seq_order_shared

                # overwrite with pre-specified seq order (do not modify if clauses above for simplicit as simple calls)
                if fixed_seq_order is not None:
                    # try to extract from dict or will fail with KeyError otherwise; do not try to compute
                    # missing keys as this indicates user is doing something wrong
                    seq_order = torch.from_numpy(fixed_seq_order[(dec_idx, enc_idx)]).to(single.device)
                else:
                    # store new seq order; this will only happen in case we instantiated a new dict above
                    assert seq_order is not None
                    current_seq_order[(dec_idx, enc_idx)] = seq_order.cpu().numpy()

                # repeat seq_order to match batch size
                seq_order = repeat(
                    seq_order, "b 1 n -> b s n", s=batch_size
                )
                assert (seq_order[:, :, :n_pos] < n_pos).all(), "Invalid scoring order"

                for i in range(0, len(seqs), batch_size):
                    cur_range = list(
                        range(i, min(i + batch_size, len(seqs)))
                    )

                    cur_seqs_one_hot = features.msa_to_onehot(
                        seqs_mapped[:, i:i + batch_size, :],
                        pos_mask=pos_mask, seq_mask=None, num_classes=utils.NUM_CLASSES
                    )

                    cur_seq_order = seq_order[:, :len(cur_range), :]

                    logits, _seq_order, embeddings = self.forward(
                        seqs=cur_seqs_one_hot,
                        single=single[[enc_idx]],
                        pairwise=pairwise[[enc_idx]],
                        pos_mask=pos_mask,
                        seq_order=cur_seq_order,
                        keep_logits_decoding_order=False,
                        return_embeddings=True
                    )

                    if return_embeddings:
                        all_embeddings.append(embeddings[0, :, :n_pos, :].cpu().numpy())

                    assert (cur_seq_order == _seq_order).all(), "seq_order should match before and after but does not"

                    # reduce to per sequence score
                    seq_scores = (
                        logits.log_softmax(dim=-1) * cur_seqs_one_hot
                    ).sum(dim=-1).sum(dim=-1).cpu().numpy()[0]

                    all_sequences += cur_range
                    all_scores.append(seq_scores)
                    all_encoder_indices += [enc_idx] * len(cur_range)
                    all_decoder_indices += [dec_idx] * len(cur_range)

        # create final dataframe representation and return
        df = pd.DataFrame(
            data=np.concatenate(
                all_scores
            ),
            columns=["score"],
        ).assign(
            seq_idx=all_sequences,
            encoder_sample=all_encoder_indices,
            decoder_sample=all_decoder_indices,
        ).set_index(
            ["seq_idx", "encoder_sample", "decoder_sample"]
        ).sort_index()

        if return_embeddings:
            embeddings = np.concatenate(
                all_embeddings, axis=0
            ).reshape(
                (len(seqs), num_encodings, num_samples, n_pos, -1), order="F"
            )

            return df, current_seq_order, embeddings
        else:
            return df, current_seq_order

    @torch.inference_mode()
    def score_single_mutants(
        self,
        seq: str | Sequence[str],
        *,
        single: Float["b n ds"],
        pairwise: Float["b n n dp"],
        pos_mask: Bool["b n"],
        num_samples: int = 4,
        batch_size: int = 64,
        position_subset: Sequence[int] | None = None,
        first_index: int = 1,
    ):
        """
        Inference helper method for scoring all single mutants of a sequence

        TODO: allow to specify multiple single/pairwise representations as argument

        TODO: could reexpress this function as wrapper around score_conditionals, creating
          duplicated WT sequences paired with each position, and subtracting WT logit afterwards
        """
        assert (
            single.shape[0] == pairwise.shape[0] == pos_mask.shape[0] == 1
        ), "Method only handles single batch"

        # check and map wildtype sequence
        assert len(seq) == pos_mask[0].sum(), "Length of sequence and pos_mask do not agree"
        _, n_padded = pos_mask.shape

        try:
            seq_mapped_list = utils.map_sequence(seq)
        except KeyError as e:
            raise ValueError("Invalid symbol in input sequence") from e

        # create tensor from mapped WT sequence
        seq_mapped = torch.tensor(
            seq_mapped_list, device=single.device
        )

        # map list of positions to compute single mutants for
        if position_subset is not None:
            position_subset = set(position_subset)

            # check if position subset is valid
            invalid_pos = {
                pos_ext for pos_ext in position_subset
                if pos_ext - first_index < 0 or pos_ext - first_index >= len(seq)
            }
            if len(invalid_pos) > 0:
                raise ValueError(f"Invalid positions in selection: {invalid_pos}")

        # get relevant positions in internal 0-based indexing
        positions = [
            pos for pos, _ in enumerate(seq)
            if position_subset is None or (pos + first_index) in position_subset
        ]

        # repeat positions, keeping same position grouped together for multiple samples
        scoring_mask_indices = [
            (pos, sample_num) for pos in positions for sample_num in range(num_samples)
        ]

        # create one-hot encoded WT sequence and repeat it batch_size times
        wt_seq_rep = repeat(
            pad(seq_mapped, (0, n_padded - len(seq))),
            "n -> b s n", b=1, s=batch_size
        )

        wt_seq_rep_one_hot = features.msa_to_onehot(
           wt_seq_rep, pos_mask=pos_mask, seq_mask=None, num_classes=utils.NUM_CLASSES
        )

        # store mutant scores per position here
        mutant_scores = np.zeros(
            (len(scoring_mask_indices), utils.NUM_CLASSES)
        )

        # iterate through positions/mutants to predict, chunking by batch size
        for i in range(0, len(scoring_mask_indices), batch_size):
            # extract current batch
            cur_batch = scoring_mask_indices[i:i + batch_size]

            # construct sequence matrix and mask for which positions to score
            # TODO: is there a better solution for the following two lines?
            # TODO: yes, can replace with one_hot solution from score_conditional
            pos_to_score = torch.zeros(
                (len(cur_batch), n_padded), device=single.device
            ).to(torch.bool)

            for j, (pos, sample_num) in enumerate(cur_batch):
                pos_to_score[j, pos] = True

            # put through model; pos_to_score will move position of interest to end of sequence;
            # randomizing all other positions
            cur_seqs = wt_seq_rep_one_hot[:len(cur_batch)]
            cur_seq_order = features.reorder_sequences(
                cur_seqs, pos_mask, pos_to_score[None, ...]
            )

            logits, seq_order = self.forward(
                seqs=cur_seqs,
                single=single,
                pairwise=pairwise,
                pos_mask=pos_mask,
                seq_order=cur_seq_order,
            )

            # the selected positions should be at the very end of the sequence order
            assert (seq_order[0, :, len(seq) - 1].cpu() == torch.tensor(
                [pos for pos, sample_num in cur_batch])).all(), "scoring position mismatch"
            assert (seq_order[0, :, :len(seq)].cpu() < len(seq)).all(), "invalid scoring order"

            # extract probabilities (note that before restoring original order in forward(), each of the extracted
            # vector corresponds to P(x_n | x_<n) and is at the very end of the ordering
            for j, (pos, sample_num) in enumerate(cur_batch):
                mutant_scores[i + j, :] = torch.nn.functional.log_softmax(
                    logits[0, j, pos, :], dim=-1
                ).cpu().numpy()

        # turn into dataframe and return
        df = pd.DataFrame(
            mutant_scores,
            columns=[utils.INDEX_TO_AA[idx] for idx in range(utils.NUM_CLASSES)],
        ).assign(
            pos=[pos + first_index for (pos, sample_num) in scoring_mask_indices],
            # reverse map sequence to make sure all positions have canonical AA
            wt_aa=[
                utils.INDEX_TO_AA[seq_mapped_list[pos]] for (pos, sample_num) in scoring_mask_indices
            ],
            # extract WT probability based on mapped AA as well
            wt=[
                mutant_scores[i, seq_mapped_list[pos]] for i, (pos, sample_num) in enumerate(scoring_mask_indices)
            ],
            sample_num=[sample_num for (pos, sample_num) in scoring_mask_indices],
        ).set_index(
            ["pos", "wt_aa", "sample_num"]
        )

        return df

    @torch.inference_mode()
    def score_conditional(
        self,
        seqs: Sequence[str],
        positions: Sequence[int],
        *,
        single: Float["b n ds"],
        pairwise: Float["b n n dp"],
        pos_mask: Bool["b n"],
        num_samples: int = 4,
        batch_size: int = 64,
        first_index: int = 1,
        share_decoding_order_across_encodings: bool = False,
    ):
        """
        Inference helper method for scoring conditional probabilities
        P(x_i | x_\i) batched across a set of sequences (position i can be individually
        defined for each sequence), e.g. for use in Gibbs sampling
        """
        seqs_mapped, n_pos, n_padded, num_encodings = self._verify_inputs_and_map_seqs(
            seqs, single=single, pairwise=pairwise, pos_mask=pos_mask
        )

        # perform additional mapping and verification specific to this method
        assert len(seqs) == len(positions), "Length of seqs and positions must agree (one position per seq)"

        # map positions to internal indexing and verify
        positions = [
            pos - first_index for pos in positions
        ]
        assert all([0 <= pos < n_pos for pos in positions]), "Position index out of bounds"

        # variables for recording results
        all_positions = []
        all_scores = []
        all_sequences = []
        all_encoder_indices = []
        all_decoder_indices = []

        # iterate sequence/position combinations in chunks, could combine this with decoder samples for
        # some extra efficiency but this complicates implementation so leave for now
        for i in range(0, len(seqs), batch_size):
            # extract current sequence chunk, pad, add batch dimension and and apply one-hot encoding;
            # note: if going for batched implementation would need to repeat b times here
            cur_seqs = rearrange(
                pad(
                    seqs_mapped[i:i + batch_size],
                    (0, n_padded - n_pos)
                ), "s n -> 1 s n"
            )

            # extract corresponding slice of positions to score in the above sequences
            cur_pos_list = positions[i:i + batch_size]
            cur_pos = torch.tensor(
                cur_pos_list, device=single.device
            )

            cur_seqs_one_hot = features.msa_to_onehot(
                cur_seqs, pos_mask=pos_mask, seq_mask=None, num_classes=utils.NUM_CLASSES
            )

            # map positions for scoring and determine sequence order
            pos_to_score = one_hot(
                cur_pos,
                num_classes=n_padded
            ).to(torch.bool)

            # iterate decoder samples as outer loop, this allows to share decoding order across encodings
            for dec_idx in range(num_samples):
                # determine sequence order for reuse
                if share_decoding_order_across_encodings:
                    seq_order_shared = features.reorder_sequences(
                        cur_seqs_one_hot, pos_mask, pos_to_score
                    )
                else:
                    seq_order_shared = None

                # iterate encoder samples (one or more); could also push this through decoder in batched fashion
                # but opt for simpler implementation here with finer-grained control over used memory for batch size
                for enc_idx in range(num_encodings):
                    if seq_order_shared is None:
                        seq_order = features.reorder_sequences(
                            cur_seqs_one_hot, pos_mask, pos_to_score
                        )
                    else:
                        seq_order = seq_order_shared

                    # pass through model
                    logits, _seq_order = self.forward(
                        seqs=cur_seqs_one_hot,
                        single=single[[enc_idx]],
                        pairwise=pairwise[[enc_idx]],
                        pos_mask=pos_mask,
                        seq_order=seq_order,
                        keep_logits_decoding_order=True
                    )

                    assert (seq_order == _seq_order).all(), "seq_order should match before and after but does not"
                    assert (seq_order[:, :, n_pos - 1] == cur_pos).all(), "Scored position not in right place"
                    assert (seq_order[:, :, :n_pos] < n_pos).all(), "Invalid scoring order"

                    # extract last position (we keep reordered sequences above), and normalize
                    pos_logits = torch.nn.functional.log_softmax(
                        logits[:, :, n_pos - 1, :], dim=-1
                    ).cpu().numpy()

                    # could directly index into a numpy array but use the following as it is more flexible
                    # to change if above code changes
                    all_scores.append(pos_logits[0])
                    # map positions back to external numbering
                    all_positions += [
                        pos + first_index for pos in cur_pos_list
                    ]
                    # add sequence indices - use length of pos list to handle last chunk length properly
                    cur_range = list(range(i, i + len(cur_pos_list)))
                    all_sequences += cur_range

                    # add encoder and decoder indices
                    all_encoder_indices += [enc_idx] * len(cur_range)
                    all_decoder_indices += [dec_idx] * len(cur_range)

        # create final dataframe representation and return
        df = pd.DataFrame(
            data=np.concatenate(
                all_scores
            ),
            columns=[utils.INDEX_TO_AA[idx] for idx in range(utils.NUM_CLASSES)],
        ).assign(
            seq_idx=all_sequences,
            pos=all_positions,
            encoder_sample=all_encoder_indices,
            decoder_sample=all_decoder_indices,
        ).set_index(
            ["seq_idx", "pos", "encoder_sample", "decoder_sample"]
        ).sort_index()

        return df

    @torch.inference_mode()
    def score_mutants(
        self,
        seq: str | Sequence[str],
        mutants: Sequence[Sequence[Tuple]],
        *,
        single: Float["b n ds"],
        pairwise: Float["b n n dp"],
        pos_mask: Bool["b n"],
        num_samples: int = 4,
        batch_size: int = 64,
        first_index: int = 1,
        mutant_sep: str = "_",
    ):
        """
        Inference helper method for scoring arbitrary mutants of a WT sequence,
        will compute ordering-tied WT sequence probability for each mutant

        # TODO: allow to specify multiple single/pairwise representations as argument
        """
        # - following segment shared with score_single_mutants, clean up if methods ---
        # do not diverge in their logic
        assert (
            single.shape[0] == pairwise.shape[0] == pos_mask.shape[0] == 1
        ), "Method only handles single batch"

        # check and map wildtype sequence
        assert len(seq) == pos_mask[0].sum(), "Length of sequence and pos_mask do not agree"
        _, n_padded = pos_mask.shape

        try:
            seq_mapped_list = utils.map_sequence(seq)
        except KeyError as e:
            raise ValueError("Invalid symbol in input sequence") from e

        # create tensor from mapped WT sequence
        seq_mapped = torch.tensor(
            seq_mapped_list, device=single.device
        )
        # - end of shared segment ----

        if batch_size % 2 != 0:
            raise ValueError("batch_size must be an even number")

        # verify and map mutations; nested for loop for easier raising of meaningful exceptions
        mutants_mapped = []
        # iterate each higher-order mutant
        for mutant in mutants:
            mutant_mapped = []
            # iterate individual substitutions in higher-order mutant
            for (pos, aa_from, aa_to) in mutant:
                mut_str = f"{aa_from}{pos}{aa_to}"
                # map to internal 0-based indexing
                pos_mapped = pos - first_index
                if pos_mapped < 0 or pos_mapped >= len(seq):
                    raise ValueError(
                        f"Mutant position out of bounds for mutant {mut_str}"
                    )
                try:
                    aa_from_mapped = utils.AA_TO_INDEX[aa_from]
                    aa_to_mapped = utils.AA_TO_INDEX[aa_to]
                except KeyError as e:
                    raise ValueError(
                        f"Invalid substitution or WT residue for mutant {mut_str}"
                    ) from e

                if aa_from_mapped != seq_mapped[pos_mapped]:
                    raise ValueError(
                        f"WT residue {aa_from} for mutant {mut_str} does not match WT sequence {seq[pos_mapped]} "
                        "(when comparing mapped characters)"
                    )

                mutant_mapped.append(
                    (pos_mapped, aa_from_mapped, aa_to_mapped)
                )

            # store mapped higher-order mutation
            mutants_mapped.append(mutant_mapped)

        # repeat mapped mutations num_samples times
        mutants_repeated = [
            (mutant, sample_num) for mutant in mutants_mapped for sample_num in range(num_samples)
        ]

        # create one-hot encoded WT sequence and repeat it batch_size times
        wt_seq_rep = repeat(
            pad(seq_mapped, (0, n_padded - len(seq))),
            "n -> b s n", b=1, s=batch_size
        )

        # accumulator variable for mutant scores
        mutant_scores = np.zeros(
             (len(mutants_repeated), )
        )

        # iterate through mutants with half of batch size, other half in current batch
        # will be used to score WT sequence with same decoding order
        half_batch_size = batch_size // 2

        for i in range(0, len(mutants_repeated), half_batch_size):
            # extract current batch
            cur_batch_mutants = mutants_repeated[i:i + half_batch_size]
            cur_batch_half_size = len(cur_batch_mutants)

            # copy WT sequence for mutating in first half of batch (retain other half at WT sequence);
            # need to account for last slice that may not fully fill the batch
            batch_seqs = wt_seq_rep.clone().detach()[:, :(2 * cur_batch_half_size), :]

            # mask for which positions to score (i.e. at end of decoding order), for half batch size
            # (then duplicated afterwards)
            pos_to_score = torch.zeros(
                (len(cur_batch_mutants), n_padded), device=single.device
            ).to(torch.bool)

            # iterate through higher-order mutants
            for j, (mutant, sample_num) in enumerate(cur_batch_mutants):
                # iterate through individual substitutions in mutant
                for (pos, aa_from, aa_to) in mutant:
                    # flag mutant position for scoring
                    pos_to_score[j, pos] = True

                    # mutate to target residue
                    assert aa_from == batch_seqs[0, j, pos]
                    batch_seqs[0, j, pos] = aa_to

            # determine sequence order based on mutants (slice to first half for this).
            # moving mutated towards end of sequence for scoring
            seq_order = features.reorder_sequences(
                batch_seqs[0, :cur_batch_half_size], pos_mask, pos_to_score
            )

            assert (seq_order[0, :, :len(seq)] < len(seq)).all(), "invalid scoring order"

            # duplicate seq_order along sequence dimension to apply order to WT sequences in second half of batch_seqs
            seq_order_dup = einx.rearrange(
                "b s n, b s n -> b (s + s) n", seq_order, seq_order
            )

            # expand sequences (mutants in first half, WT in second half) to 1-hot encoding
            batch_seqs_one_hot = features.msa_to_onehot(
                batch_seqs, pos_mask=pos_mask, seq_mask=None, num_classes=utils.NUM_CLASSES
            )

            # compute mutant scores and normalize
            raw_logits, _seq_order_out = self.forward(
                seqs=batch_seqs_one_hot, single=single, pairwise=pairwise, pos_mask=pos_mask, seq_order=seq_order_dup
            )

            logits = torch.nn.functional.log_softmax(
                 raw_logits, dim=-1
            ).cpu()

            # make sure we used the sequence ordering supplied as input
            assert (_seq_order_out == seq_order_dup).all()

            # extract probability ratios (note that forward call above reorders probabilities)
            for j, (mutant, sample_num) in enumerate(cur_batch_mutants):
                # iterate through individual substitutions in mutant
                for (pos, aa_from, aa_to) in mutant:
                    # mutant sequence is at index j; corresponding WT sequence is at position j + cur_batch_half_size
                    log_prob_ratio = logits[0, j, pos, aa_to] - logits[0, j + cur_batch_half_size, pos, aa_from]

                    # accumulate product over all changed positions (all positions that were not mutated
                    # cancel out, so can be ignored here)
                    mutant_scores[i + j] += log_prob_ratio

        # turn into dataframe and return
        df = pd.DataFrame(
            mutant_scores, columns=["score"]
        ).assign(
            sample_num=[sample_num for (mutant, sample_num) in mutants_repeated],
            mutant=[
                mutant_sep.join([
                    "{}{}{}".format(utils.INDEX_TO_AA[aa_from], pos + first_index, utils.INDEX_TO_AA[aa_to])
                    for (pos, aa_from, aa_to) in mutant
                ])
                for (mutant, sample_num) in mutants_repeated
            ],
        ).set_index(
            ["mutant", "sample_num"]
        )

        return df

    @torch.inference_mode()
    def sample_inefficient(
        self,
        *,
        single: Float["b n ds"],
        pairwise: Float["b n n dp"],
        pos_mask: Bool["b n"],
        seq: str | Sequence[str] | None = None,
        num_samples: int = 128,
        batch_size: int = 64,
        temperature: float = 1.0,
        min_p: float | None = None,
        sample_gaps: bool = False,
    ):
        """
        Generate new sequences by autoregressive sampling from the model. Specify
        fixed positions using the seq argument, while masking any positions that
        need to be designed with "*" (these will be used as prefix during generation
        to provide maximum context).

        NOTE: Since KV caching is not yet implemented for decoder, this implementation will
        repeatedly evaluate the full generated sequence state leading to cubic rather than
        quadratic time complexity in the sequence length.

        Useful links for guiding implementation:
        https://github.com/huggingface/transformers/blob/ce1d328e3b73cf6d1d993fc0d487b7dc8a14d7ee/benchmark/llama.py#L141C9-L149C25
        https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
        https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        # verify inputs
        assert (
            pos_mask.shape[0] == 1
        ), "Method only handles single batch (but single/pair reps for same positions can be stacked)"

        assert (
            single.shape[0] == pairwise.shape[0]
        ), "single and pairwise batch dimensions do not agree"

        n_padded = single.shape[1]
        assert (
            n_padded == pairwise.shape[1] == pairwise.shape[2] == pos_mask.shape[1]
        ), "Sequence length dimensions do not match"

        # number of actually covered positions, check it also maps to WT sequence
        n_pos = pos_mask[0].sum()
        if seq is not None:
            assert len(seq) == n_pos, "Length of sequence and pos_mask do not agree"

        assert num_samples % batch_size == 0, "num_samples must be multiple of batch_size"

        # tokens to exclude from sampling
        if sample_gaps:
            # avoid special tokens except gaps
            avoided_tokens = utils.AVOIDED_TOKENS
        else:
            # also avoid gaps
            avoided_tokens = utils.AVOIDED_TOKENS_WITH_GAP

        # map avoided tokens to internal numbering
        avoided_tokens = utils.map_sequence(avoided_tokens)

        assert min_p is None or 0 <= min_p <= 1, "min_p must be in range from 0 to 1 (or None)"

        # handle pre-specified/fixed portions of designed sequence
        if seq is not None:
            try:
                seq_mapped_list = utils.map_sequence(seq)
            except KeyError as e:
                raise ValueError("Invalid symbol in input sequence") from e

            # create tensor from mapped WT sequence
            start_seq = torch.tensor(
                seq_mapped_list, device=single.device
            )

            # variable positions to design marked by mask token / "*" for now
            pos_to_design = (start_seq == utils.AA_TO_INDEX[utils.MASK])
        else:
            # design of full sequence
            start_seq = torch.zeros(n_pos, device=single.device, dtype=torch.int64)
            pos_to_design = pos_mask[0]

        # determine number of positions that will need to be designed
        n_pos_to_sample = pos_to_design.sum()
        assert n_pos_to_sample > 0, "Need to sample at least one position"

        # repeat starting sequence and positions to design
        start_seq_rep = repeat(
            pad(start_seq, (0, n_padded - n_pos)),
            "n -> b s n", b=1, s=batch_size
        )

        pos_to_design_rep = repeat(
            pad(pos_to_design, (0, n_padded - n_pos)),
            "n -> b s n", b=1, s=batch_size
        )

        # sequential index into all sequences in batch for fancy indexing
        idx_into_seqs = torch.arange(0, batch_size, device=single.device)

        # accumulate all detailed results for returning across batches
        all_results = []

        # iterate through target number of designs in batches
        for i in range(0, num_samples, batch_size):
            # determine sequence decoding order for current batch,
            # moving positions that can be sampled towards end of sequence
            seq_order = features.reorder_sequences(
                start_seq_rep, pos_mask, pos_to_design_rep
            )

            # accumulate log probabilities of sampled position
            sample_logits = torch.zeros((batch_size,), device=single.device)

            # initialize sequences for current batch in one-hot encoded state
            # (note this tensor will be mutated in loop below as positions are designed).
            batch_seqs_one_hot = features.msa_to_onehot(
                start_seq_rep, pos_mask=pos_mask, seq_mask=None, num_classes=utils.NUM_CLASSES
            )

            # sample variable positions for current batch one by one
            for j in range(n_pos_to_sample):
                # compute index of current position that will be generated
                # (variable positions will be at the end of the decoding order, prefixed by fixed positions)
                cur_logits_idx = n_pos - n_pos_to_sample + j

                # corresponding indices in generated sequences
                cur_seq_pos_idx = seq_order[0, :, cur_logits_idx]

                # iterate through one or multiple sets of single/pair representation stacked on first dimension;
                n_encodings = single.shape[0]

                # initialize to empty logits (will be used to accumulate)
                logits = None

                # iterate through all supplied encodings to compute amino acid logits, then average at the end
                for idx_enc in range(n_encodings):
                    # keep decoding order of output tensor, so we can simply pick logits;
                    # note this step is currently highly inefficient as we recompute logits over all tokens so far
                    # rather than just computing the logits for the next token (i.e. cubic instead of quadratic
                    # complexity)
                    raw_logits, _seq_order_out = self.forward(
                        seqs=batch_seqs_one_hot,
                        single=single[[idx_enc]],
                        pairwise=pairwise[[idx_enc]],
                        pos_mask=pos_mask,
                        seq_order=seq_order,
                        keep_logits_decoding_order=True
                    )

                    assert (_seq_order_out == seq_order).all()

                    # slice logits for current position that will be sampled and normalize in log space
                    norm_logits = torch.nn.functional.log_softmax(
                        raw_logits[0, :, cur_logits_idx, :], dim=-1
                    )

                    # accumulate logits over different encodings
                    if logits is None:
                        logits = norm_logits
                    else:
                        logits += norm_logits

                # average logits if we have more than one encoding
                if n_encodings > 1:
                    logits /= n_encodings

                # set tokens we do not want to sample altogether to -infinity for zero probability
                logits[:, avoided_tokens] = -torch.inf

                # scale with temperature
                logits_temp_scaled = logits / temperature

                # apply filtering strategy if specified; for now, only min-p supported
                # apply *after* temperature as recommended in Section 7 of https://arxiv.org/html/2407.01082v1
                if min_p is not None:
                    logits_temp_scaled = utils.min_p_filter(
                        logits_temp_scaled, min_p=min_p
                    )

                # finally, turn into probability distribution and sample
                token_probs = logits_temp_scaled.softmax(dim=-1)
                sampled_token = torch.multinomial(token_probs, 1).flatten()

                # update current sequence state: first blank out entire last dimension for one-hot encoding,
                # then set one hot for sampled AA
                batch_seqs_one_hot[0, idx_into_seqs, cur_seq_pos_idx, :] = 0
                batch_seqs_one_hot[0, idx_into_seqs, cur_seq_pos_idx, sampled_token] = 1

                # update sequence log probability for sampled positions (use logits before adjusting for sampling
                # with temperature etc.)
                sample_logits += logits[idx_into_seqs, sampled_token]

            # assemble final results for current batch:
            # turn back into dense sequence representation from one-hot encoding
            batch_seqs_final = batch_seqs_one_hot.argmax(dim=-1)

            all_results.append((
                batch_seqs_final.cpu().numpy(), sample_logits.cpu().numpy(), seq_order.cpu().numpy()
            ))

        # create summary dataframe from results
        result_summary = pd.DataFrame([
            {
                "seq": "".join(utils.INDEX_TO_AA[aa_idx] for aa_idx in seq),
                "sum_sampled_logits": score,
            }
            for (seqs, logits, order) in all_results
            for seq, score in zip(
                seqs[0, :, :],
                logits,
            )
        ])

        return result_summary, all_results


def loss_function(
    logits: Float["b s n dsd"],
    target: Float["b s n dsd"] | Int["b s n dsd"],
    pos_mask: Bool["b n"],
    seq_mask: Bool["b s"],
) -> Float:
    """
    Calculate cross-entropy for predicted sequences

    Note there is *no* need to shift logits / target to account for added BOS token as all of this is
    handled internally by decoder.

    Parameters
    ----------
    logits
        Predicted logits from model
    target
        True sequences (one-hot encoded)
    pos_mask
        Valid position mask
    seq_mask
        Valid sequence mask
    Returns
    -------
    cross_ent
        Mean cross entropy between logits and target
    """
    if logits.shape != target.shape:
        raise ValueError(
            f"Shape of logits and target must agree: {logits.shape} vs {target.shape}"
        )

    # turn one-hot encoded target sequence back into class indices
    target_class = torch.argmax(target, dim=-1)

    # apply masking by setting ignore_index value of cross_entropy function
    ignore_index = -100

    # first apply position mask
    target_class = einx.where(
        "b n, b s n, -> b s n", pos_mask, target_class, ignore_index
    )

    # then apply sequence mask
    target_class = einx.where(
        "b s, b s n, -> b s n", seq_mask, target_class, ignore_index
    )

    # rearrange may or may not return a view so use torch.view() here
    target_class_flat = target_class.view(-1)
    # target_class_flat_alt = einx.rearrange("b s n -> (b s n)", target_class)
    # assert (target_class_flat == target_class_flat_alt).all()

    # adjust shape of logits accordingly
    logits_flat = logits.view(-1, logits.size(-1))
    # logits_flat_alt = einx.rearrange("b s n d -> (b s n) d", logits)
    # assert (logits_flat_alt == logits_flat).all()

    cross_ent = cross_entropy(
        logits_flat, target_class_flat, ignore_index=ignore_index
    )

    return cross_ent


def cuda_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    else:
        return None


class Model(L.LightningModule):

    def __init__(
        self,
        encoder_params=None,
        decoder_params=None,
        # training_loss_msa_sample_size: int = 64,
        training_max_recycles: int = 4,
        optimizer_settings=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(
            **(encoder_params if encoder_params is not None else ENCODER_DEFAULT_SETTINGS)
        )

        self.decoder = Decoder(
            **(decoder_params if decoder_params is not None else DECODER_DEFAULT_SETTINGS)
        )

        # self.loss_msa_sample_size = training_loss_msa_sample_size
        self.training_max_recycles = training_max_recycles
        self.optimizer_settings = optimizer_settings if optimizer_settings is not None else OPTIMIZER_DEFAULT_SETTINGS

    def forward(self, x, num_recycling_steps, return_loss=True):
        """
        Encode input features and reconstruct set of sequences by running through decoder
        """
        if x.seqs_to_decode is None or x.seqs_to_decode_mask is None:
            raise ValueError(
                "Need to specify seqs_to_decode and seqs_to_decode_mask on input features"
            )

        # encode to single and pair representations
        s, p = self.encoder(
            x, num_recycling_steps=num_recycling_steps
        )

        # extract random sequence sample for decoding
        # seqs_sampled, seq_mask_sampled, _ = features.sample_sequences_dense(
        #     self.loss_msa_sample_size,
        #     msa=x.msa,
        #     msa_mask=x.msa_mask,
        #     pos_mask=x.pos_mask,
        #     num_classes=utils.NUM_CLASSES
        # )

        # convert sampled sequences to one-hot encoded version
        seqs_to_decode_onehot = features.msa_to_onehot(
            x.seqs_to_decode, pos_mask=x.pos_mask, seq_mask=x.seqs_to_decode_mask, num_classes=utils.NUM_CLASSES
        )

        # decode sampled sequences
        logits, seq_order = self.decoder(
            single=s,
            pairwise=p,
            seqs=seqs_to_decode_onehot,
            pos_mask=x.pos_mask
        )

        if return_loss:
            loss = loss_function(
                logits=logits,
                target=seqs_to_decode_onehot,
                pos_mask=x.pos_mask,
                seq_mask=x.seqs_to_decode_mask
            )
        else:
            loss = None

        return logits, seq_order, loss

    def training_step(self, batch, batch_idx):
        # retrieve current sequence chunk
        #  (note we do not have labels, but we subsample MSA to reconstruct in unsupervised setting)
        x = batch

        # sample number of recycling steps for entire batch (cf. AF2 supplement Algorithm 31)
        sampled_recycle_number = torch.randint(
            1, self.training_max_recycles + 1, (1,)
        )[0]
        logger.info(f"Sampled number of recycles: {sampled_recycle_number}")

        _, _, loss = self.forward(
            x, num_recycling_steps=sampled_recycle_number, return_loss=True
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.pos_mask.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        # compute with maximum number of recycling steps
        _, _, loss = self.forward(
            x, num_recycling_steps=self.training_max_recycles, return_loss=True
        )

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=x.pos_mask.shape[0],
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            **self.optimizer_settings
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=default_lambda_lr_fn, verbose=True
        )

        # cf. https://github.com/amorehead/alphafold3-pytorch-lightning-hydra/blob/3cc8b88e1ee28091273bd3c686d836d3812c5322/alphafold3_pytorch/models/alphafold3_module.py#L270C9-L279C10
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "step",
                "frequency": 1,
                "name": "lambda_lr",
            },
        }

    # https://lightning.ai/docs/pytorch/stable/debug/debugging_intermediate.html#look-out-for-exploding-gradients
    # https://github.com/Lightning-AI/pytorch-lightning/issues/19987
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)

        # self.log_dict(norms)
        self.log(
            "grad_2.0_norm_total", norms["grad_2.0_norm_total"]
        )

