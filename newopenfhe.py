#!/usr/bin/env python3
"""Merged pure-Python CKKS bootstrap helper.

This script combines two steps into one entry:
1) Generate pEnc/pDec from EvalBootstrapSetup-like high-level inputs.
2) Generate FindBootstrapRotationIndices-equivalent rotation indices.

No pybind/OpenFHE runtime is required for these calculations.
"""

import argparse
import json
import math
from enum import IntEnum
from typing import Iterable, List, Optional, Sequence, Tuple


class CKKSBootParams(IntEnum):
    LEVEL_BUDGET = 0
    LAYERS_COLL = 1
    LAYERS_REM = 2
    NUM_ROTATIONS = 3
    BABY_STEP = 4
    GIANT_STEP = 5
    NUM_ROTATIONS_REM = 6
    BABY_STEP_REM = 7
    GIANT_STEP_REM = 8


def reduce_rotation(idx: int, size: int) -> int:
    if size <= 0:
        raise ValueError("size must be positive")
    # Mirrors OpenFHE behavior for power-of-two modulus with fallback for generic case.
    return idx & (size - 1) if (size & (size - 1)) == 0 else idx % size


def select_layers(log_slots: int, budget: int) -> Tuple[int, int, int]:
    """Python mirror of SelectLayers(logSlots, budget) in ckksrns-utils.cpp."""
    if log_slots <= 0:
        raise ValueError("log_slots must be positive")
    if budget <= 0:
        raise ValueError("budget must be positive")

    layers = int(math.ceil(float(log_slots) / budget))
    rows = log_slots // layers
    rem = log_slots % layers
    dim = rows + (1 if rem != 0 else 0)

    # The above choice ensures dim <= budget.
    if dim < budget:
        layers -= 1
        rows = log_slots // layers
        rem = log_slots - rows * layers
        dim = rows + (1 if rem != 0 else 0)

        # The above choice ensures dim >= budget.
        if dim > budget:
            while dim != budget:
                rows -= 1
                rem = log_slots - rows * layers
                dim = rows + (1 if rem != 0 else 0)

    return layers, rows, rem


def get_collapsed_fft_params(slots: int, level_budget: int, dim1: int) -> List[int]:
    """Python mirror of GetCollapsedFFTParams(slots, levelBudget, dim1)."""
    if slots == 0:
        raise ValueError("slots can not be 0")
    if level_budget == 0:
        raise ValueError("levelBudget can not be 0")

    log_slots = 1 if slots < 3 else int(math.log2(slots))
    layers_collapse, _rows, rem_collapse = select_layers(log_slots, level_budget)

    num_rotations = (1 << (layers_collapse + 1)) - 1
    num_rotations_rem = (1 << (rem_collapse + 1)) - 1

    if dim1 == 0 or dim1 > num_rotations:
        g = 1 << (layers_collapse // 2 + 1 + (1 if num_rotations > 7 else 0))
    else:
        g = dim1
    b = (num_rotations + 1) // g

    if rem_collapse != 0:
        g_rem = 1 << (rem_collapse // 2 + 1 + (1 if num_rotations_rem > 7 else 0))
        b_rem = (num_rotations_rem + 1) // g_rem
    else:
        g_rem = 0
        b_rem = 0

    return [
        int(level_budget),
        int(layers_collapse),
        int(rem_collapse),
        int(num_rotations),
        int(b),
        int(g),
        int(num_rotations_rem),
        int(b_rem),
        int(g_rem),
    ]


def normalize_setup_inputs(
    M: int,
    num_slots: int,
    level_budget: Sequence[int],
    dim1: Sequence[int],
) -> Tuple[int, List[int], List[int], int]:
    """Apply the same basic setup guards used by EvalBootstrapSetup."""
    if M <= 0:
        raise ValueError("M must be positive")
    if len(level_budget) != 2:
        raise ValueError("level_budget must have 2 integers: [encBudget, decBudget]")
    if len(dim1) != 2:
        raise ValueError("dim1 must have 2 integers: [dim1Enc, dim1Dec]")

    slots = (M // 4) if num_slots == 0 else num_slots
    if slots <= 0:
        raise ValueError("slots must be positive")

    log_slots = 1 if slots < 3 else int(math.log2(slots))
    enc_budget = max(1, min(int(level_budget[0]), log_slots))
    dec_budget = max(1, min(int(level_budget[1]), log_slots))

    return slots, [enc_budget, dec_budget], [int(dim1[0]), int(dim1[1])], log_slots


def gen_penc_pdec_from_setup_inputs(
    M: int,
    num_slots: int,
    level_budget: Sequence[int],
    dim1: Sequence[int],
) -> Tuple[List[int], List[int], int]:
    slots, budgets, dims, _log_slots = normalize_setup_inputs(M, num_slots, level_budget, dim1)
    p_enc = get_collapsed_fft_params(slots, budgets[0], dims[0])
    p_dec = get_collapsed_fft_params(slots, budgets[1], dims[1])
    return p_enc, p_dec, slots


def find_linear_transform_rotation_indices(slots: int, M: int, dim1_enc: int) -> List[int]:
    g = int(math.ceil(math.sqrt(slots))) if dim1_enc == 0 else dim1_enc
    h = int(math.ceil(float(slots) / g))

    out: List[int] = []
    for i in range(1, g + 1):
        out.append(i)
    for i in range(2, h):
        out.append(g * i)

    m = slots * 4
    if m != M:
        j = 1
        while j < (M // m):
            out.append(j * slots)
            j <<= 1

    return out


def find_coeffs_to_slots_rotation_indices(slots: int, M: int, p_enc: Sequence[int]) -> List[int]:
    level_budget = int(p_enc[CKKSBootParams.LEVEL_BUDGET])
    layers_collapse = int(p_enc[CKKSBootParams.LAYERS_COLL])
    rem_collapse = int(p_enc[CKKSBootParams.LAYERS_REM])
    num_rotations = int(p_enc[CKKSBootParams.NUM_ROTATIONS])
    b = int(p_enc[CKKSBootParams.BABY_STEP])
    g = int(p_enc[CKKSBootParams.GIANT_STEP])
    num_rotations_rem = int(p_enc[CKKSBootParams.NUM_ROTATIONS_REM])
    b_rem = int(p_enc[CKKSBootParams.BABY_STEP_REM])
    g_rem = int(p_enc[CKKSBootParams.GIANT_STEP_REM])

    flag_rem = 0 if rem_collapse == 0 else 1
    out: List[int] = []

    for s in range(level_budget - 1, flag_rem - 1, -1):
        scaling_factor = 1 << ((s - flag_rem) * layers_collapse + rem_collapse)
        half_rots = 1 - ((num_rotations + 1) // 2)

        for j in range(half_rots, g + half_rots):
            out.append(reduce_rotation(j * scaling_factor, slots))
        for i in range(b):
            out.append(reduce_rotation((g * i) * scaling_factor, M // 4))

    if flag_rem:
        half_rots = 1 - ((num_rotations_rem + 1) // 2)
        for j in range(half_rots, g_rem + half_rots):
            out.append(reduce_rotation(j, slots))
        for i in range(b_rem):
            out.append(reduce_rotation(g_rem * i, M // 4))

    m = slots * 4
    if m != M:
        j = 1
        while j < (M // m):
            out.append(j * slots)
            j <<= 1

    return out


def find_slots_to_coeffs_rotation_indices(slots: int, M: int, p_dec: Sequence[int]) -> List[int]:
    level_budget = int(p_dec[CKKSBootParams.LEVEL_BUDGET])
    layers_collapse = int(p_dec[CKKSBootParams.LAYERS_COLL])
    rem_collapse = int(p_dec[CKKSBootParams.LAYERS_REM])
    num_rotations = int(p_dec[CKKSBootParams.NUM_ROTATIONS])
    b = int(p_dec[CKKSBootParams.BABY_STEP])
    g = int(p_dec[CKKSBootParams.GIANT_STEP])
    num_rotations_rem = int(p_dec[CKKSBootParams.NUM_ROTATIONS_REM])
    b_rem = int(p_dec[CKKSBootParams.BABY_STEP_REM])
    g_rem = int(p_dec[CKKSBootParams.GIANT_STEP_REM])

    flag_rem = 0 if rem_collapse == 0 else 1
    out: List[int] = []

    for s in range(0, level_budget - flag_rem):
        scaling_factor = 1 << (s * layers_collapse)
        half_rots = 1 - ((num_rotations + 1) // 2)

        for j in range(half_rots, g + half_rots):
            out.append(reduce_rotation(j * scaling_factor, M // 4))
        for i in range(b):
            out.append(reduce_rotation((g * i) * scaling_factor, M // 4))

    if flag_rem:
        s = level_budget - flag_rem
        scaling_factor = 1 << (s * layers_collapse)
        half_rots = 1 - ((num_rotations_rem + 1) // 2)

        for j in range(half_rots, g_rem + half_rots):
            out.append(reduce_rotation(j * scaling_factor, M // 4))
        for i in range(b_rem):
            out.append(reduce_rotation((g_rem * i) * scaling_factor, M // 4))

    m = slots * 4
    if m != M:
        j = 1
        while j < (M // m):
            out.append(j * slots)
            j <<= 1

    return out


def find_bootstrap_rotation_indices(M: int, slots: int, p_enc: Sequence[int], p_dec: Sequence[int], dim1_enc: int) -> List[int]:
    is_lt_bootstrap = (
        int(p_enc[CKKSBootParams.LEVEL_BUDGET]) == 1
        and int(p_dec[CKKSBootParams.LEVEL_BUDGET]) == 1
    )

    if is_lt_bootstrap:
        full_index_list = find_linear_transform_rotation_indices(slots, M, dim1_enc)
    else:
        full_index_list = find_coeffs_to_slots_rotation_indices(slots, M, p_enc)
        full_index_list.extend(find_slots_to_coeffs_rotation_indices(slots, M, p_dec))

    # Remove duplicates, then remove automorphisms 0 and M/4.
    uniq = sorted(set(int(v) for v in full_index_list))
    filtered = [v for v in uniq if v not in (0, M // 4)]
    return filtered


def parse_int_list(text: Optional[str]) -> List[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def compare_with_autoidx(indices: Sequence[int], autoidx_to_rotidx_values: Iterable[int]) -> dict:
    idx_set = set(int(x) for x in indices)
    map_set = set(int(x) for x in autoidx_to_rotidx_values)
    return {
        "missing_in_autoidx_to_rotidx": sorted(idx_set - map_set),
        "extra_in_autoidx_to_rotidx": sorted(map_set - idx_set),
        "intersection_size": len(idx_set & map_set),
        "indices_size": len(idx_set),
        "map_values_size": len(map_set),
    }


def to_signed_indices(indices: Sequence[int], size: int) -> List[int]:
    if size <= 0:
        raise ValueError("size must be positive")

    half = size // 2
    signed: List[int] = []
    for v in indices:
        vv = reduce_rotation(int(v), size)
        if vv > half:
            vv -= size
        signed.append(vv)

    return signed

def main() -> None:
    # parser = argparse.ArgumentParser(
    #     description="Generate pEnc/pDec and bootstrap rotation indices from EvalBootstrapSetup-like inputs."
    # )
    # parser.add_argument("--M", type=int, required=True, help="Cyclotomic order M")
    # parser.add_argument("--num-slots", type=int, default=0, help="numSlots passed to setup, 0 means M/4")
    # parser.add_argument("--level-budget", type=str, required=True, help="enc,dec (example: 4,4)")
    # parser.add_argument("--dim1", type=str, required=True, help="enc,dec (example: 0,0)")
    # parser.add_argument(
    #     "--autoidx-to-rotidx-values",
    #     type=str,
    #     default="",
    #     help="Optional comma-separated values from AUTOIDX_TO_ROTIDX.values() for diff check",
    # )
    # parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")

    # args = parser.parse_args()

    # level_budget = parse_int_list(args.level_budget)
    # dim1 = parse_int_list(args.dim1)


    logN = 16
    logbsslot = 15
    level_budget = [5,5]
    dim1 = [0,0]
    
    
    N = 2**logN
    M = 2*N
    num_slots = 2**logbsslot
    
    p_enc, p_dec, num_slots = gen_penc_pdec_from_setup_inputs(
        M=M,
        num_slots=num_slots,
        level_budget=level_budget,
        dim1=dim1,
    )

    indices = find_bootstrap_rotation_indices(
        M=M,
        slots=num_slots,
        p_enc=p_enc,
        p_dec=p_dec,
        dim1_enc=dim1[0],
    )

    indices_signed = to_signed_indices(indices, num_slots)

    out = {
        "inputs": {
            "M": M,
            "num_slots": num_slots,
            "level_budget": level_budget,
            "dim1": dim1,
            "effective_slots": num_slots,
        },
        "pEnc": p_enc,
        "pDec": p_dec,
        "indices": indices_signed,
        "indices_unsigned": indices,
    }

    # values = parse_int_list(args.autoidx_to_rotidx_values)
    # if values:
    #     out["compare"] = compare_with_autoidx(indices, values)

    # if args.pretty:
    #     print(json.dumps(out, indent=2, sort_keys=False))
    # else:
    print(json.dumps(out, separators=(",", ":"), sort_keys=False))


if __name__ == "__main__":
    main()
