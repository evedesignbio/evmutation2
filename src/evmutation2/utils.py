from typing import Sequence
import torch

# https://github.com/aqlaboratory/openfold/blob/f6c875b3c8e3e873a932cbe3b31f94ae011f6fd4/openfold/np/residue_constants.py#L975

MASK = "*"
BOS = ">"
GAP = "-"

AA_TO_INDEX = {
    "A": 0,
    "B": 2,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "J": 20,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "O": 20,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "U": 1,
    "V": 17,
    "W": 18,
    "X": 20,
    "Y": 19,
    "Z": 3,
    GAP: 21,
    MASK: 22,
    BOS: 23,
}

# NUM_CLASSES = len(set(AA_TO_INDEX.values()))
NUM_CLASSES = max(AA_TO_INDEX.values()) + 1

# inverse mapping, replacing ambiguous symbols with canonical AA
INDEX_TO_AA = {
    idx: symbol for symbol, idx in AA_TO_INDEX.items() if symbol not in {"U", "B", "Z", "J", "O"}
}

# tokens not to sample during generation
AVOIDED_TOKENS = [MASK, BOS, "X"]
AVOIDED_TOKENS_WITH_GAP = [GAP] + AVOIDED_TOKENS


def map_sequence(seq: str | Sequence[str]):
    return [
        AA_TO_INDEX[c] for c in seq
    ]


def min_p_filter(
    scores: torch.Tensor,
    min_p: float = 0.05,
    min_tokens_to_keep: int = 1,
    filter_value: float = -torch.inf
) -> torch.Tensor:
    """
    Min-P filtering as presented in https://arxiv.org/html/2407.01082v1
    According to Section 7, apply *after* temperature

    Implementation from https://github.com/gante/transformers/blob/54739a320e38bc86cd250303a35e68d5d3f14a83/src/transformers/generation/logits_process.py#L537
    """
    # Convert logits to probabilities
    probs = torch.softmax(scores, dim=-1)
    # Get the probability of the top token for each sequence in the batch
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    # Calculate the actual min_p threshold by scaling min_p with the top token's probability
    scaled_min_p = min_p * top_probs
    # Create a mask for tokens that have a probability less than the scaled min_p
    tokens_to_remove = probs < scaled_min_p

    sorted_indices = torch.argsort(scores, descending=True, dim=-1)
    sorted_indices_to_remove = torch.gather(tokens_to_remove, dim=-1, index=sorted_indices)
    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., : min_tokens_to_keep] = False

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    scores_processed = scores.masked_fill(indices_to_remove, filter_value)
    return scores_processed
