"""
Feature generation and transformation

Some code included below where stated from OpenFold according to following license:

# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
from typing import Tuple, NamedTuple, Sequence, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import one_hot, pad
from einops import rearrange, repeat
import einx
from evmutation2.alphafold3_pytorch.tensor_typing import Float, Int, Bool
from evmutation2.alphafold3_pytorch.alphafold3 import max_neg_value, pack_one

from evmutation2.parsers import Msa, TemplateHit
from evmutation2.utils import AA_TO_INDEX, NUM_CLASSES, MASK, GAP
from evmutation2.structures import Chain
from evmutation2.rigid_utils import Rigid


def extract_msa_feature_data(msa: Msa) -> Tuple[np.ndarray, np.ndarray]:
    """
    Featurize sequence alignment, in preparation for full expansion to
    PyTorch input features

    Parameters
    ----------
    msa: 
        Parsed multiple sequence alignment

    Returns
    -------
    (msa_matrix, deletion_matrix)
    """
    # verify we have at least one sequence
    if len(msa) == 0:
        raise ValueError("Alignment must contain at least one sequence") 
    
    # number of sequences (rows) and positions in alignment (columns)
    num_seq = len(msa)
    num_pos = len(msa.sequences[0])

    msa_matrix = np.empty(
        (num_seq, num_pos), dtype=np.int64
    )
    
    deletion_matrix = np.empty(
        (num_seq, num_pos), dtype=np.float32
        # dtype=np.int32
    )

    # iterate through sequences and assign values to matrix
    for idx, (seq, deletions) in enumerate(
        zip(msa.sequences, msa.deletion_matrix)
    ):
        msa_matrix[idx, :] = [
            AA_TO_INDEX[aa] for aa in seq
        ]
        
        deletion_matrix[idx, :] = deletions

    # derive target sequence features
    # token_index = np.array(range(num_pos))
    # restype = msa_matrix[0, :]
    
    return msa_matrix, deletion_matrix


class TemplateInputFeatures(NamedTuple):
    template_restype: torch.Tensor
    template_pseudo_beta_mask: torch.Tensor
    template_backbone_frame_mask: torch.Tensor
    template_distogram: torch.Tensor
    template_unit_vector: torch.Tensor


class InputFeatures(NamedTuple):
    # sequence-level features
    token_index: torch.Tensor
    restype: torch.Tensor

    # Encoder MSA features
    msa: torch.Tensor
    profile: torch.Tensor
    has_deletion: torch.Tensor
    deletion_value: torch.Tensor
    deletion_mean: torch.Tensor

    # Decoder features
    seqs_to_decode: torch.Tensor | None

    # 3D structure template features
    # (note that we need to nest these into list as we can have multiple templates per target)
    templates: Sequence[TemplateInputFeatures] | None


def prepare_msa_features(
        msa_matrix: np.ndarray,
        deletion_matrix: np.ndarray
) -> InputFeatures:
    """
    Create full set of pytorch features from core feature data

    Generated target sequence features in simplified implementation (AF3 nomenclature)
    1) token_index = residue_index [N_token]
    2) restype [N_token, N_alphabet] (note: may be excluded from encoder-decoder architecture)

    Generated MSA features in simplified implementation (AF3 nomenclature):
    3) msa [N_msa, N_token, N_alphabet]
    4) has_deletion [N_msa, N_token]
    5) deletion_value [N_msa, N_token]
    6) profile [N_token, N_alphabet]
    7) deletion_mean [N_token]

    Parameters
    ----------
    msa_matrix: 
        Alignment matrix, with first sequence corresponding to target 
        (ungapped and complete)
    deletion_matrix: 
        Deletion count matrix

    Returns
    -------
    InputFeatures
        Named tuple with all relevant features as PyTorch tensors
    """
    if msa_matrix.shape != deletion_matrix.shape:
        raise ValueError("msa_matrix and deletion_matrix shapes must agree")
    
    num_seq, num_pos = msa_matrix.shape

    # turn deletion matrix into tensor for further processing
    deletion_tensor = torch.from_numpy(deletion_matrix)
    
    # prepare sequence-level features
    
    # one-hot encode target sequence
    restype = one_hot(
        torch.from_numpy(msa_matrix[0, :]),
        num_classes=NUM_CLASSES,
    )
    
    token_index = torch.from_numpy(
        np.array(range(num_pos))
    )

    # prepare alignment-level features
    msa = torch.from_numpy(msa_matrix)

    # one-hot encode all alignment sequences
    msa_one_hot = one_hot(
        msa, num_classes=NUM_CLASSES,
    )

    # Compute the sequence profile (currently without reweighting sequences)
    profile = msa_one_hot.to(torch.float32).mean(dim=0)

    # remove one-hot encoded version again to save memory and use dense representation before subsampling
    del msa_one_hot

    # binary feature if there is a deletion present
    has_deletion = deletion_tensor.clip(0.0, 1.0)

    # normalized deletion count, value in range (0, 1)
    deletion_value = torch.atan(deletion_tensor / 3.0) * (
        2.0 / torch.pi
    )

    # average number of deletions per position
    # follow https://github.com/lucidrains/alphafold3-pytorch/blob/f3e51735862fa97d3177970a6a05313ab06a1e9f/alphafold3_pytorch/inputs.py#L2837C25-L2839C10
    # https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/data/data_transforms_multimer.py#L203
    # https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/data/data_transforms.py#L571
    deletion_mean = torch.atan(deletion_tensor.mean(dim=0) / 3.0) * (
        2.0 / torch.pi
    )
    
    return InputFeatures(
        # sequence-level features
        token_index=token_index,
        restype=restype,

        # MSA-level features
        msa=msa,
        profile=profile,
        has_deletion=has_deletion,
        deletion_value=deletion_value,
        deletion_mean=deletion_mean,

        # decoder features,
        seqs_to_decode=None,

        # initialize template features to None
        templates=None,
    )


class InputFeatureBatch(NamedTuple):
    # target sequence features
    pos_mask: Bool["b n"]
    token_index: Int["b n"]  # Int64
    restype: Int["b n d"]  # Int64

    # encoder/alignment features
    msa: Int["b s n"]  # Int64 dense encoding (expanded to one-hot for subsampled MSA on the fly)
    msa_mask: Bool["b s"]
    has_deletion: Bool["b s n"]
    deletion_value: Float["b s n"]
    deletion_mean: Float["b n"]
    profile: Float["b n d"]

    # decoder features
    seqs_to_decode: Int["b s n"] | None
    seqs_to_decode_mask: Bool["b s"] | None

    # template features (already merged from individual features)
    templates: Float['b t n n dt'] | None
    template_mask: Bool['b t'] | None


def batch_features(
    feature_collections: Sequence[InputFeatures],
    device: str = "cpu",
    fixed_seq_len: Optional[int] = None,
    fixed_msa_len: Optional[int] = None,
    fixed_decoder_msa_len: Optional[int] = None,
    fixed_template_num: Optional[int] = None,
    template_feature_dim: int = 92,
    # pad_value: Optional[int] = None,
) -> InputFeatureBatch:
    """
    Assemble input features for one or more samples into one tensor per feature
    for entire batch, adding padding along dimensions as needed

    Parameters
    ----------
    feature_collections:
        Sequence of input features to be assembled into batch tensor
    device:
        PyTorch device to be used for final tensors (default: cpu)
    fixed_seq_len:
        Pad to this size in sequence dimension 
        (optional, otherwise largest sequence determines size)
    fixed_msa_len:
        Pad to this size in MSA dimension
        (optional, otherwise largest alignment determines size)
    fixed_decoder_msa_len:
        Pad to this size in decoder sequences (sub-MSA) dimension
        (optional, otherwise largest sequence set/sub-MSA determines size)
    fixed_template_num:
        Pad to this size in template dimension
        (optional, otherwise largest template list determines size)
    template_feature_dim:
        Dimensionality of template input feature vector
    # pad_value:
    #    Padding value forwarded to torch.nn.functional.pad
    #    (default: None, leading to 0 padding)

    Returns
    -------
    batch_feat:
        Assembled features padded and stacked together as batch
    """
    # do not expose padding value, use zeroes by default
    pad_value = None

    # determine sequence lengths across all samples
    seq_lens = [
        len(f.token_index) for f in feature_collections
    ]

    # determine MSA number of sequences across all samples
    msa_lens = [
        f.msa.shape[0] for f in feature_collections
    ]

    # determine if decoder seqs are present (must be in either none, or all input samples),
    # and determine their respective sub-MSA lengths
    decoder_msa_lens = [
        (f.seqs_to_decode.shape[0] if f.seqs_to_decode is not None else None) for f in feature_collections
    ]
    has_decoder_seqs = all(dml is not None for dml in decoder_msa_lens)

    assert (
            all(dml is None for dml in decoder_msa_lens) or has_decoder_seqs
    ), "Decoder sequences must be specified for all input samples, or for none"

    # determine number of templates across all samples (if present)
    template_lens = [
        (len(f.templates) if f.templates is not None else 0) for f in feature_collections
    ]

    if fixed_template_num is not None and fixed_template_num < 1:
        raise ValueError("fixed_template_num must be either None or >= 1")

    def _map_lens(lengths, length_spec):
        # fixed sequence length requested for batch
        if length_spec is not None:
            # verify we do not exceed requested dimension size
            if max(lengths) > length_spec:
                raise ValueError(
                    f"Max sequence length {max(lengths)} exceeds fixed_length"
                )

            batch_length = length_spec
        else:
            # use maximum length across sequences otherwise
            batch_length = max(lengths)
            
        pad_length = [batch_length - cl for cl in lengths]
        return pad_length

    # determine required padding
    seq_padding = _map_lens(seq_lens, fixed_seq_len)
    msa_padding = _map_lens(msa_lens, fixed_msa_len)
    template_padding = _map_lens(template_lens, fixed_template_num)

    # first, target sequence features:
    # token_index
    token_index = torch.stack([
        pad(f.token_index, (0, pad_seq), value=pad_value)
        for f, pad_seq in zip(feature_collections, seq_padding)
    ])

    # restype
    restype = torch.stack([
        pad(f.restype, (0, 0, 0, pad_seq), value=pad_value)
        for f, pad_seq in zip(feature_collections, seq_padding)
    ])
    
    # position mask per sequence; dimension: seq
    pos_mask = torch.stack([
        pad(torch.ones_like(f.token_index), (0, pad_seq), value=pad_value)
        for f, pad_seq in zip(feature_collections, seq_padding)
    ]).bool()

    # second, MSA features:
    # dimensions: seq, pos (reverse for pad function)
    msa_padded = torch.stack([
        pad(f.msa, (0, pad_seq, 0, pad_msa), value=pad_value)
        for f, pad_seq, pad_msa in zip(feature_collections, seq_padding, msa_padding)
    ])

    # MSA mask
    msa_mask = torch.stack([
        pad(
            torch.ones(msa_len), (0, pad_msa), value=pad_value
        ) for msa_len, pad_msa in zip(msa_lens, msa_padding)
    ]).bool()

    # has_deletion
    has_deletion = torch.stack([
        pad(f.has_deletion, (0, pad_seq, 0, pad_msa), value=pad_value)
        for f, pad_seq, pad_msa in zip(feature_collections, seq_padding, msa_padding)
    ])

    # deletion_value
    deletion_value = torch.stack([
        pad(f.deletion_value, (0, pad_seq, 0, pad_msa), value=pad_value)
        for f, pad_seq, pad_msa in zip(feature_collections, seq_padding, msa_padding)
    ])

    # deletion_mean
    deletion_mean = torch.stack([
        pad(f.deletion_mean, (0, pad_seq), value=pad_value)
        for f, pad_seq in zip(feature_collections, seq_padding)
    ])                      

    # profile
    profile = torch.stack([
        pad(f.profile, (0, 0, 0, pad_seq), value=pad_value)
        for f, pad_seq in zip(feature_collections, seq_padding)
    ])

    # sequences for decoding (during training)
    if has_decoder_seqs:
        decoder_msa_padding = _map_lens(decoder_msa_lens, fixed_decoder_msa_len)

        # sub-MSA decoder features
        seqs_to_decode_padded = torch.stack([
            pad(f.seqs_to_decode, (0, pad_seq, 0, pad_dec_msa), value=pad_value)
            for f, pad_seq, pad_dec_msa in zip(feature_collections, seq_padding, decoder_msa_padding)
        ])

        # sub-MSA mask
        seqs_to_decode_mask = torch.stack([
            pad(
                torch.ones(dec_msa_len), (0, pad_dec_msa), value=pad_value
            ) for dec_msa_len, pad_dec_msa in zip(decoder_msa_lens, decoder_msa_padding)
        ]).bool()
    else:
        seqs_to_decode_padded = None
        seqs_to_decode_mask = None

    # templates (if available)
    templates = None
    template_mask = None

    # determine if any template features are present
    total_templates = sum(template_lens)

    # only add templates if we have at least one actual template in batch
    if total_templates > 0:
        # concatenate template features per sample first, then stack
        # together into one tensor per batch (these all must have the same seq len by definition);
        # if no templates present for some sample, create one dummy entry with all zeroes for later padding
        templates_per_sample = [
            (torch.stack([
                torch.cat((
                    t.template_restype,
                    t.template_pseudo_beta_mask[..., None],
                    t.template_distogram,
                    t.template_backbone_frame_mask[..., None],
                    t.template_unit_vector,
                ), dim=-1)
                for t in f.templates
            ]) if f.templates is not None else torch.zeros(
                (1, seq_len, seq_len, template_feature_dim)
            ))
            for f, seq_len in zip(feature_collections, seq_lens)
        ]

        # add padding to match seq_len - this needs to be consistent by definition, and
        # stack together across samples in batch;
        # account for adding 1-sample template for None entries above by subtracting from padding
        templates = torch.stack([
            pad(
                t,
                (
                    0, 0,  # dt
                    0, pad_seq,  # n
                    0, pad_seq,  # n
                    0, pad_template if f.templates is not None else pad_template - 1,  # t,
                ),
                value=pad_value
            )
            for f, t, pad_template, pad_seq in zip(
                feature_collections, templates_per_sample, template_padding, seq_padding
            )
        ])

        # create template mask
        template_mask = torch.stack([
            pad(
                torch.ones(template_len), (0, pad_template), value=pad_value
            ) for template_len, pad_template in zip(template_lens, template_padding)
        ]).bool()

    # assemble features and return
    batch_feat = InputFeatureBatch(
        # target sequence features
        pos_mask=pos_mask.to(device),
        token_index=token_index.to(device),
        restype=restype.to(device),
    
        # alignment features
        msa=msa_padded.to(device),
        msa_mask=msa_mask.to(device),
        has_deletion=has_deletion.to(device),
        deletion_value=deletion_value.to(device),
        deletion_mean=deletion_mean.to(device),
        profile=profile.to(device),

        # seqs for decoder features
        seqs_to_decode=seqs_to_decode_padded,
        seqs_to_decode_mask=seqs_to_decode_mask,

        # template features
        templates=templates.to(device) if templates is not None else None,
        template_mask=template_mask.to(device) if template_mask is not None else None,
    )

    return batch_feat


def reorder_sequences(
        seqs: Float["b s n d"],   # TODO: last dimension d should be removed?
        pos_mask: Bool["b n"],
        masked_positions: Bool["b s n"] | None = None
) -> Int["b s n"]:
    """
    Move mask tokens to the end of each sequence, randomizing the decoding order of all other tokens
    in the prefix before mask tokens

    Parameters
    ----------
    seqs
        Batched sequence tensor (can be MSA or MSA subset)
    pos_mask
        Position mask for sequences
    masked_positions
        Positions that will be moved towards end/right of each sequence
        to maximize context prefix for prediction

    Returns
    -------
    order
        Indices corresponding to token order for each sequence
    """
    # determine which positions per sequence are mask token (to move these to the end of masked sequence),
    # these will receive a value of 1
    # is_mask_token = seqs[..., mask_token_index] > 0

    # if positions of interest are not given, create equivalent tensor with all entries false == position for prefix
    if masked_positions is None:
        # TODO: last dimension corresponding to d should only be unpacked if 4 and not 3 dimensions/dense seqs?
        b, s, n = seqs.shape[:3]
        masked_positions = torch.zeros(
            (b, s, n), device=seqs.device
        ).to(torch.bool)

    # move positions not covered by mask to very end of sorting with value 2;
    # will need broadcasting from b n to b s n
    pos_mask_sorting = rearrange((~pos_mask) * 2, "b n -> b 1 n")

    # get random numbers to determine random ordering within 0/1/2 groups, drawing from [0, 1)
    # should retain grouping already but divide by 10 to make easier to spot groups
    to_sort = masked_positions + pos_mask_sorting + (torch.rand(masked_positions.shape, device=seqs.device) / 10)

    # determine indices of ordering by argsort
    order = torch.argsort(to_sort, dim=-1)

    return order


def sample_sequences(
        msa: Int["b s n d"],
        msa_mask: Bool["b s"] | None,
        max_num_samples: int
) -> Tuple[Int["b s n d"], Bool["b s"] | None]:
    """
    Sample sequences without replacement

    Based on sequence sampling code in MSAModule:
    https://github.com/lucidrains/alphafold3-pytorch/blob/a89dc1f54264697562f380718c8c6543667e6767/alphafold3_pytorch/alphafold3.py#L1013

    Parameters
    ----------
    msa
        One-hot encoded multiple sequence alignment
    msa_mask:
        Mask indicating actual sequence entries in msa
    max_num_samples
        Number of sequences to sample (maximum, if fewer sequences
        are present, a smaller number will be returned)

    Returns
    -------
    msa_sampled
        Sampled alignment, with at most max_num_samples sequences per batch entry
    msa_mask
        Valid entries in msa_sampled
    """
    batch, num_msa, device = *msa.shape[:2], msa.device

    if max_num_samples > num_msa:
        raise ValueError("Requested more samples than present in MSA")

    rand = torch.randn((batch, num_msa), device=device)

    if msa_mask is not None:
        rand.masked_fill_(~msa_mask, max_neg_value(msa))

    indices = rand.topk(max_num_samples, dim=-1).indices

    # deactivate get_at due to checkpointing issues
    # msa_sampled = einx.get_at('b [s] n dm, b sampled -> b sampled n dm', msa, indices)
    msa, unpack_one = pack_one(msa, "b s *")
    msa_indices = repeat(indices, "b sampled -> b sampled d", d=msa.shape[-1])
    msa_sampled = msa.gather(1, msa_indices)
    msa_sampled = unpack_one(msa_sampled)

    if msa_mask is not None:
        # msa_mask_sampled = einx.get_at('b [s], b sampled -> b sampled', msa_mask, indices)
        msa_mask_sampled = msa_mask.gather(1, indices)
    else:
        msa_mask_sampled = None

    return msa_sampled, msa_mask_sampled


def msa_to_onehot(
    msa: Int["b s n"],
    pos_mask: Bool["b n"] | None = None,
    seq_mask: Bool["b s"] | None = None,
    num_classes=NUM_CLASSES,
):
    # create one hot representation
    msa_one_hot = one_hot(msa, num_classes=num_classes)

    # mask with position mask (as dense MSA is padded with zeros)
    if pos_mask is not None:
        msa_one_hot = einx.where(
            "b n, b s n d,", pos_mask, msa_one_hot, 0
        )

    # also mask with sequence mask
    if seq_mask is not None:
        msa_one_hot = einx.where(
            "b s, b s n d,", seq_mask, msa_one_hot, 0
        )

    return msa_one_hot


def sample_sequences_dense(
        max_num_msa: int,
        msa: Int["b s n"],
        msa_mask: Bool["b s"],
        pos_mask: Bool["b n"],
        additional_msa_feats: Float["b s n d"] | None = None,
        num_classes=NUM_CLASSES,
) -> Tuple[Int["b s n d"], Bool["b s"] | None, Float["b s n d"]]:
    """
    Sample sequences without replacement from *dense* input MSA, return one-hot encoded output MSA

    Based on sequence sampling code in MSAModule and modified for dense representation
    https://github.com/lucidrains/alphafold3-pytorch/blob/a89dc1f54264697562f380718c8c6543667e6767/alphafold3_pytorch/alphafold3.py#L1013

    Parameters
    ----------
    max_num_msa
        Number of sequences to sample (maximum, if fewer sequences
        are present, a smaller number will be returned)
    msa
        Densely encoded multiple sequence alignment
    msa_mask:
        Mask indicating actual sequence entries in msa
    pos_mask:
        Mask indicating actual sequences positions
    additional_msa_feats:
        Additional alignment features to add to one-hot encoded feature
        (has_deletion, deletion_value features)
    num_classes:
        Number of states in msa to be used for one-hot encoding

    Returns
    -------
    msa_sampled
        One-hot encoded sampled alignment, with at most max_num_samples sequences per batch entry
    msa_mask
        Valid entries in msa_sampled
    additional_msa_feats
        Corresponding sampled additional features
    """
    batch, num_msa, device = *msa.shape[:2], msa.device

    # sample without replacement

    if num_msa > max_num_msa:
        rand = torch.randn((batch, num_msa), device=device)

        if msa_mask is not None:
            rand.masked_fill_(~msa_mask, max_neg_value(pos_mask.float()))

        indices = rand.topk(max_num_msa, dim=-1).indices

        # msa = einx.get_at('b [s] n dm, b sampled -> b sampled n dm', msa, indices)

        msa, unpack_one = pack_one(msa, "b s *")
        msa_indices = repeat(indices, "b sampled -> b sampled d", d=msa.shape[-1])
        msa = msa.gather(1, msa_indices)
        msa = unpack_one(msa)

        if msa_mask is not None:
            # msa_mask = einx.get_at('b [s], b sampled -> b sampled', msa_mask, indices)
            msa_mask = msa_mask.gather(1, indices)

        if additional_msa_feats is not None:
            # additional_msa_feats = einx.get_at('b s 2, b sampled -> b sampled 2', additional_msa_feats, indices)
    
            additional_msa_feats, unpack_one = pack_one(additional_msa_feats, "b s *")
            additional_msa_indices = repeat(
                indices, "b sampled -> b sampled d", d=additional_msa_feats.shape[-1]
            )
            additional_msa_feats = additional_msa_feats.gather(1, additional_msa_indices)
            additional_msa_feats = unpack_one(additional_msa_feats)

    # create one hot representation
    msa_one_hot = msa_to_onehot(
        msa, pos_mask=pos_mask, seq_mask=msa_mask, num_classes=num_classes
    )

    return msa_one_hot, msa_mask, additional_msa_feats


# from github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/utils/feats.py#L93
def dgram_from_positions(
    pos: torch.Tensor,
    min_bin: float = 3.25,
    max_bin: float = 50.75,
    no_bins: float = 39,
    inf: float = 1e8,
):
    dgram = torch.sum(
        (pos[..., None, :] - pos[..., None, :, :]) ** 2, dim=-1, keepdim=True
    )
    lower = torch.linspace(min_bin, max_bin, no_bins, device=pos.device) ** 2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram >= lower) * (dgram < upper)).type(dgram.dtype)

    return dgram


def prepare_template_features(
    chain: Chain,
    target_to_chain_map: pd.DataFrame,
    raise_seq_mismatches: bool = True,
    min_bin: float = 3.25,
    max_bin: float = 50.75,
    no_bins: float = 39,
    eps: float = 1e-20,
    inf: float = 1e8
):
    """
    # TODO: document parameters
    note: pdb_pos needs to be string for easier nan handling
    """
    # extract relevant AA residues, move index to column for later merging to atoms
    res = chain.residues.dropna(
        subset="seqres_id"
    ).reset_index().rename(columns={
        "index": "residue_index"
    })

    # expand residue list with mapping to target, and prepare for reindexing atom coord pivot table
    map_with_res = target_to_chain_map.merge(
        res, how="left", left_on="pdb_pos", right_on="seqres_id"
    )
    idx_into_coords = map_with_res.residue_index.replace(np.nan, -1).astype(int).values

    # verify sequence agreement, but allow target to be specified incompletely
    if raise_seq_mismatches:
        mismatch = map_with_res.query(
            "(pdb_aa.notnull() and one_letter_code.notnull() and pdb_aa != one_letter_code) or " +
            "(three_letter_code == 'MSE' and pdb_aa != 'M')"
        )

        if len(mismatch) > 0:
            raise ValueError("Sequences of target and structure do not match" + str(mismatch))

    # extract relevant atom coords (not yet reindexed for target);
    # take highest-occupancy atom per residue if altloc is present
    coords = pd.pivot_table(
        chain.coords.query(
            "atom_name in ('N', 'CA', 'C', 'CB')"
        ).sort_values(
            by=["residue_index", "atom_name", "alt_loc"],
            ascending=[True, True, False]
        ).drop_duplicates(
            subset=["residue_index", "atom_name", "alt_loc"],
        ),
        index="residue_index",
        columns="atom_name",
        values=["x", "y", "z"]
    ).swaplevel(
        axis=1
    ).sort_index(
        axis=1
    ).reindex(
        # make sure all residue indices are still available (may disappear if none of N/CA/C/CB available in structure;
        # e.g. for 6f3a:U
        res.residue_index.astype(int).values,
        axis=0
    )

    # create pseudo-C beta (CA if glycine, CB otherwise),
    # make copy of coords so we don't modify original dataframe
    gly_pos = res.query("three_letter_code == 'GLY'")
    try:
        pseudo_beta = coords.loc[:, "CB"].copy()
    except KeyError:
        # allow to skip structure on outer calling functions by reraising as ValueError
        raise ValueError("No CB atoms in structure")

    try:
        pseudo_beta.loc[gly_pos.residue_index, :] = coords.loc[gly_pos.residue_index, "CA"]
    except KeyError:
        raise ValueError("No CB atoms in structure or missing glycine residue index")

    # map into target sequence space, this will fill all missing positions
    # in protein chain with placeholder value
    coords_mapped = coords.reindex(idx_into_coords, axis=0)
    pseudo_beta_mapped = pseudo_beta.reindex(idx_into_coords, axis=0)

    # compute actual features
    valid_for_dgram = torch.from_numpy(
        pseudo_beta_mapped.notnull().all(axis=1).values
    )

    # template_pseudo_beta_mask (here: already expanded to 2D)
    #   Mask indicating if the Cβ (Cα for glycine) has coordinates for the template at this residue.
    template_pseudo_beta_mask = valid_for_dgram[:, None] * valid_for_dgram[None, :]

    # template_distogram; nan for missing positions leads to all bins == 0
    dgram = dgram_from_positions(
        torch.from_numpy(pseudo_beta_mapped.values),
        min_bin, max_bin, no_bins, inf
    ).to(torch.float16)

    # template_restype [Ntempl, Ntoken]:
    #   One-hot encoding of the template sequence, see restype.
    # note: for now use sequence from alignment to avoid MSE issues etc. - could
    # be an issue
    aa_mapped = torch.from_numpy(
        map_with_res.pdb_aa.replace(np.nan, GAP).map(AA_TO_INDEX).values
    )

    # encode as one hot and mask out invalid positions
    template_restype_1d = one_hot(
        aa_mapped, num_classes=NUM_CLASSES,
    ) * valid_for_dgram[:, None]

    # tile and stack for 2D seq representation
    template_restype_rep = repeat(
        template_restype_1d.to(torch.float16),
        "n d -> n n2 d", n2=template_restype_1d.shape[-2]
    )

    # transpose and stack
    template_restype_2d = einx.rearrange(
        "n n2 d, n2 n d -> n n2 (d + d)", template_restype_rep, template_restype_rep
    )

    # template_backbone_frame_mask
    valid_for_unit_vector = torch.from_numpy(
        coords_mapped[["N", "CA", "C"]].notnull().all(axis=1).values
    )
    template_backbone_frame_mask = valid_for_unit_vector[:, None] * valid_for_unit_vector[None, :]

    # template_unit_vector; following section copied/modified from
    # https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/utils/feats.py#L93;
    # note: set all missing values to 0 rather than nan to easily get rid of vectors
    # by multiplying with mask later on
    input = coords_mapped.replace(np.nan, 0)

    rigids = Rigid.make_transform_from_reference(
        n_xyz=torch.from_numpy(input["N"].values),
        ca_xyz=torch.from_numpy(input["CA"].values),
        c_xyz=torch.from_numpy(input["C"].values),
        eps=eps
    )

    points = rigids.get_trans()[..., None, :, :]
    rigid_vec = rigids[..., None].invert_apply(points)
    inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec ** 2, dim=-1))

    inv_distance_scalar = inv_distance_scalar * template_backbone_frame_mask
    unit_vector = rigid_vec * inv_distance_scalar[..., None]

    return TemplateInputFeatures(
        template_restype=template_restype_2d,
        template_pseudo_beta_mask=template_pseudo_beta_mask,
        template_backbone_frame_mask=template_backbone_frame_mask,
        template_distogram=dgram,
        template_unit_vector=unit_vector,
    )


def add_seqs_to_decode(input_features, seqs_to_decode: Int["s n"]) -> InputFeatures:
    """
    Extend input features with structural templates

    Parameters
    ----------
    input_features
        Input features without templates added (MSA only)
    seqs_to_decode
        Sequences to reconstruct with decoder

    Returns
    -------
    Updated input features with structural templates added

    """
    return input_features._replace(seqs_to_decode=seqs_to_decode)


def add_template_features(input_features, templates: Sequence[TemplateInputFeatures]) -> InputFeatures:
    """
    Extend input features with structural templates

    Parameters
    ----------
    input_features
        Input features without templates added (MSA only)
    templates
        One or more templates matched to same target sequence

    Returns
    -------
    Updated input features with structural templates added

    """
    return input_features._replace(templates=templates)


def template_hit_to_posmap(
    hit: TemplateHit,
    target_seq: str,
    target_first_index: int = 0,
    raise_seq_mismatches: bool = True
) -> pd.DataFrame:
    """

    Note: target sequence indices are expected to be 0-based

    Parameters
    ----------
    hit
        Template hit from HHsearch
    target_seq
        Full target sequence (as HHsearch object only contains hit region)
    target_first_index
        First index of target sequence, 0-based indexing (default: 0)
    raise_seq_mismatches
        Verify agreement of target_seq with target sequence in hit object
        (for subset of aligned residues)

    Returns
    -------
    Residue mapping table for extract_template_feature_data function
    """
    # note: sequence indices from hhr parser are 0-based (query/target and hit)
    hit_df = pd.DataFrame({
        "target_pos": hit.indices_query,
        "target_aa_hit": list(hit.query),
        "pdb_pos": hit.indices_hit,
        "pdb_aa": list(hit.hit_sequence),
    }).query(
        "target_pos != -1 and pdb_pos != -1"
    ).assign(
        # adjust PDB seqres indices to expected 1-based indexing and string
        target_pos=lambda df: df.target_pos + target_first_index,
        pdb_pos=lambda df: (df.pdb_pos + 1).astype(str)
    )

    # create pos map for full target sequence (or crop thereof, which may be different to
    # region covered by template)
    target_df = pd.DataFrame({
        "target_pos": range(
            target_first_index, target_first_index + len(target_seq)
        ),
        "target_aa": list(target_seq),
    })

    target_hit_df = target_df.merge(
        hit_df, how="left", on="target_pos"
    )

    assert len(target_hit_df) == len(target_seq)

    if raise_seq_mismatches:
        mismatches = target_hit_df.query("target_aa_hit.notnull() and target_aa_hit != target_aa")
        if len(mismatches) > 0:
            raise ValueError(
                "Reference target sequence does not match template hit target sequence: " + str(mismatches)
            )

    return target_hit_df


def filter_template_hits(
    template_hits: Sequence[TemplateHit],
    target_seq: str,
    crop_start: int,
    crop_end: int,
    target_first_index: int = 0,
    max_evalue: float | None = 1e-05,
    min_crop_coverage: int | None = 64,
    max_hits: int | None = 20,
    valid_pdb_ids: Sequence | None = None,
    raise_seq_mismatches: bool = False
) -> Sequence[Tuple[pd.DataFrame, TemplateHit, int, int]]:
    """
    Filter full template hit to subset that overlaps the currently
    sampled crop

    Note: target sequence indices are expected to be 0-based

    Parameters
    ----------
    template_hits
        All template hits from HHsearch
    target_seq
        Full target sequence
    crop_start
        First index (inclusive) of region from template to consider
    crop_end
        Last index (exclusive) of region from template to consider
    max_evalue
        Highest acceptable E-value for hit to be retained
    min_crop_coverage
        Minimum PDB coverage of cropped target sequence to be retained as hit
        (number of positions)
    max_hits:
        Maximum number of hits to retain after applying all other filters
    valid_pdb_ids:
        Check each hit against valid identifier list
    target_first_index
        First index of target sequence, 0-based indexing (default: 0)
    raise_seq_mismatches
        Verify agreement of target_seq with target sequence in hit object
        (for subset of aligned residues); if True, any mismatches will be
        raised as a ValueError, if False, will skip and filter any hits
        that do not match

    Returns
    -------
    Filtered tuples (hit, PDB identifier, PDB chain, index in original list, coverage of crop)
    """
    filtered_hits = []
    for i, hit in enumerate(template_hits):
        try:
            # only keep templates passing E-value requirement (if specified)
            if max_evalue is not None and hit.e_value > max_evalue:
                continue

            # expand hit into position mapping to target sequence and filter to crop region
            posmap = template_hit_to_posmap(
                hit, target_seq, target_first_index, raise_seq_mismatches
            ).query(
                "@crop_start <= target_pos < @crop_end"
            )

            # compute actual number of target positions covered by template
            pos_crop_covered = posmap.pdb_pos.notnull().sum()

            # filter by requested minimum coverage of crop
            if min_crop_coverage is not None and pos_crop_covered < min_crop_coverage:
                continue

            # filter by valid PDB ids
            pdb_id, pdb_chain = hit.name.split()[0].split("_")
            pdb_id = pdb_id.lower()

            if valid_pdb_ids is not None and pdb_id not in valid_pdb_ids:
                continue

            filtered_hits.append(
                (posmap, pdb_id, pdb_chain, i, pos_crop_covered)
            )

        except ValueError as e:
            if raise_seq_mismatches:
                raise

        # limit total number of hits
        if max_hits is not None and len(filtered_hits) >= max_hits:
            break

    return filtered_hits
