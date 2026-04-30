#!/usr/bin/env python3
"""Joint codesign search for bootstrap keys and Orion multistep decomposition."""

import argparse
import importlib.util
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
ORION_PATH = ROOT / "codesignOrion&Bs.py"
BOOT_PATH = ROOT / "newopenfhe.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORION = _load_module("codesign_orion", ORION_PATH)
BOOT = _load_module("bootstrap_helper", BOOT_PATH)


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_range_list(text: str) -> List[int]:
    values: List[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            step = 1 if start <= end else -1
            values.extend(list(range(start, end + step, step)))
        else:
            values.append(int(chunk))
    return values


def load_orion_groups(path: Path) -> List[List[int]]:
    text = path.read_text()
    match = re.search(r'all_a_diags_raw = """(.*)"""', text, re.S)
    if not match:
        raise ValueError("Failed to locate all_a_diags_raw in Orion script")
    return ORION._parse_list_of_lists(match.group(1))


def merged_orion_diags(groups: Sequence[Sequence[int]], slots: int) -> List[int]:
    return sorted({int(x) & (slots - 1) for group in groups for x in group})


def canonical_set(values: Iterable[int], slots: int) -> set:
    return ORION._collapse_with_conjugation(values, slots)


def final_union_count(bs_key: Sequence[int], orion_needed: Iterable[int], slots: int, use_conjugation: bool) -> int:
    if use_conjugation:
        bs_norm = canonical_set(bs_key, slots)
        need = canonical_set(orion_needed, slots)
    else:
        bs_norm = {ORION.norm_rot_index(int(x) & (slots - 1), slots * 2) for x in bs_key}
        need = set(int(x) for x in orion_needed)
    return len(bs_norm | need)


def bootstrap_rotation_cost(p: Sequence[int]) -> int:
    return int(p[BOOT.CKKSBootParams.LEVEL_BUDGET]) * (
        int(p[BOOT.CKKSBootParams.BABY_STEP]) + int(p[BOOT.CKKSBootParams.GIANT_STEP])
    )


def bootstrap_metadata(indices: Sequence[int], slots: int) -> Dict[str, object]:
    unsigned = sorted(int(x) for x in indices)
    signed = BOOT.to_signed_indices(unsigned, slots)
    canonical = sorted(canonical_set(unsigned, slots))
    return {
        "bs_key_unsigned": unsigned,
        "bs_key_signed": signed,
        "bs_key_canonical": canonical,
    }


def evaluate_joint_candidate(
    merged_a_diags: List[int],
    M: int,
    slots: int,
    level_budget: Sequence[int],
    dim1: Sequence[int],
    levels: int,
    use_conjugation: bool,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> Dict[str, object]:
    p_enc, p_dec, effective_slots = BOOT.gen_penc_pdec_from_setup_inputs(
        M=M,
        num_slots=slots,
        level_budget=level_budget,
        dim1=dim1,
    )
    bs_key = BOOT.find_bootstrap_rotation_indices(
        M=M,
        slots=effective_slots,
        p_enc=p_enc,
        p_dec=p_dec,
        dim1_enc=int(dim1[0]),
    )

    orion_stat = ORION.find_best_multistep_strategy(
        merged_a_diags,
        n=M // 2,
        bs_key=list(bs_key),
        slots=effective_slots,
        levels=levels,
        mode="weighted",
        alpha=1.0,
        beta=alpha,
        gamma=beta,
        use_conjugation=use_conjugation,
    )

    bs_meta = bootstrap_metadata(bs_key, effective_slots)
    final_union = final_union_count(bs_key, orion_stat["needed_all"], effective_slots, use_conjugation)
    bootstrap_keys = len(bs_meta["bs_key_canonical"] if use_conjugation else bs_meta["bs_key_unsigned"])
    bootstrap_cost = bootstrap_rotation_cost(p_enc) + bootstrap_rotation_cost(p_dec)
    joint_score = (
        alpha * float(len(orion_stat["new_only"]))
        + beta * float(final_union)
        + gamma * float(orion_stat["online_rotations_avg"])
        + delta * float(bootstrap_keys)
    )

    return {
        "level_budget": list(level_budget),
        "dim1": list(dim1),
        "levels": levels,
        "use_conjugation": use_conjugation,
        "p_enc": p_enc,
        "p_dec": p_dec,
        "bootstrap_key_count": bootstrap_keys,
        "bootstrap_rotation_cost": bootstrap_cost,
        "joint_final_union_count": final_union,
        "joint_score": joint_score,
        "orion": orion_stat,
        **bs_meta,
    }


def markdown_table(rows: Sequence[Dict[str, object]]) -> str:
    headers = [
        "rank",
        "budget",
        "dim1",
        "orion_levels",
        "orion_steps",
        "conj",
        "bs_keys",
        "orion_new",
        "orion_final",
        "joint_final",
        "avg_rot",
        "joint_score",
    ]
    out = [
        "| " + " | ".join(headers) + " |",
        "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows, start=1):
        orion = row["orion"]
        out.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    f"`{row['level_budget']}`",
                    f"`{row['dim1']}`",
                    str(row["levels"]),
                    f"`{' x '.join(str(x) for x in orion['steps'])}`",
                    "yes" if row["use_conjugation"] else "no",
                    str(row["bootstrap_key_count"]),
                    str(len(orion["new_only"])),
                    str(len(orion["final_keyindex"])),
                    str(row["joint_final_union_count"]),
                    f"{float(orion['online_rotations_avg']):.4f}",
                    f"{float(row['joint_score']):.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(out)


def detail_block(best: Dict[str, object]) -> str:
    orion = best["orion"]
    lines = [
        "## Best Candidate Details",
        "",
        f"- `level_budget = {best['level_budget']}`",
        f"- `dim1 = {best['dim1']}`",
        f"- `orion_steps = {orion['steps']}`",
        f"- `use_conjugation = {best['use_conjugation']}`",
        f"- `joint_score = {best['joint_score']:.4f}`",
        f"- `bootstrap_key_count = {best['bootstrap_key_count']}`",
        f"- `bootstrap_rotation_cost = {best['bootstrap_rotation_cost']}`",
        f"- `orion_new_only = {len(orion['new_only'])}`",
        f"- `orion_final_keyindex = {len(orion['final_keyindex'])}`",
        f"- `joint_final_union = {best['joint_final_union_count']}`",
        f"- `orion_avg_rot = {orion['online_rotations_avg']:.4f}`",
        "",
        "### Bootstrap Keys",
        "",
        f"- unsigned: `{best['bs_key_unsigned']}`",
        f"- signed: `{best['bs_key_signed']}`",
        f"- canonical: `{best['bs_key_canonical']}`",
        "",
        "### Orion",
        "",
        f"- needed_all: `{sorted(orion['needed_all'])}`",
        f"- new_only: `{sorted(orion['new_only'])}`",
        f"- reused: `{sorted(orion['reused'])}`",
        f"- final_keyindex: `{sorted(orion['final_keyindex'])}`",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint codesign search across bootstrap params and Orion multistep decomposition.")
    parser.add_argument("--logN", type=int, default=16)
    parser.add_argument("--log-slots", type=int, default=15)
    parser.add_argument("--budgets", type=str, default="4-6", help="budget search range, e.g. 4-6 or 4,5,6")
    parser.add_argument("--dim1", type=str, default="0,8,16,32,64,128,256", help="dim1 candidate list for enc/dec")
    parser.add_argument("--levels", type=str, default="2,3,4", help="Orion multistep levels to search")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0, help="weight for Orion new keys")
    parser.add_argument("--beta", type=float, default=0.7, help="weight for final joint union size")
    parser.add_argument("--gamma", type=float, default=6.0, help="weight for Orion avg rotations")
    parser.add_argument("--delta", type=float, default=0.2, help="weight for bootstrap key count")
    args = parser.parse_args()

    N = 1 << args.logN
    M = 2 * N
    slots = 1 << args.log_slots

    groups = load_orion_groups(ORION_PATH)
    merged_a_diags = merged_orion_diags(groups, slots)

    budgets = parse_range_list(args.budgets)
    dim1_values = parse_int_list(args.dim1)
    levels_values = parse_int_list(args.levels)

    results: List[Dict[str, object]] = []
    for enc_budget in budgets:
        for dec_budget in budgets:
            for dim1_enc in dim1_values:
                for dim1_dec in dim1_values:
                    for levels in levels_values:
                        for use_conjugation in (False, True):
                            results.append(
                                evaluate_joint_candidate(
                                    merged_a_diags=merged_a_diags,
                                    M=M,
                                    slots=slots,
                                    level_budget=[enc_budget, dec_budget],
                                    dim1=[dim1_enc, dim1_dec],
                                    levels=levels,
                                    use_conjugation=use_conjugation,
                                    alpha=args.alpha,
                                    beta=args.beta,
                                    gamma=args.gamma,
                                    delta=args.delta,
                                )
                            )

    results.sort(
        key=lambda row: (
            float(row["joint_score"]),
            int(row["joint_final_union_count"]),
            len(row["orion"]["new_only"]),
            float(row["orion"]["online_rotations_avg"]),
            row["level_budget"],
            row["dim1"],
            row["levels"],
            row["use_conjugation"],
        )
    )

    top_rows = results[: max(1, args.topk)]
    print("# Joint Codesign Search")
    print("")
    print(f"- `M = {M}`")
    print(f"- `slots = {slots}`")
    print(f"- `budgets = {budgets}`")
    print(f"- `dim1 candidates = {dim1_values}`")
    print(f"- `Orion levels = {levels_values}`")
    print("")
    print(markdown_table(top_rows))
    print("")
    print(detail_block(top_rows[0]))


if __name__ == "__main__":
    main()
