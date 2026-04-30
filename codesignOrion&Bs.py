import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


RESULTS_CSV_PATH = Path(__file__).resolve().parent / "results" / "codesign_results.csv"
BETA_SWEEP_VALUES = [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def norm_rot_index(i: int, n: int) -> int:
    if i < 0:
        i = n // 2 + i
    return i


def _bsgs_index(non_zero_diags: Iterable[int], slots: int, n1: int) -> Tuple[int, int]:
    rot_n1_map: Set[int] = set()
    rot_n2_map: Set[int] = set()

    for rot in non_zero_diags:
        rot &= (slots - 1)
        idx_n1 = ((rot // n1) * n1) & (slots - 1)
        idx_n2 = rot & (n1 - 1)
        rot_n1_map.add(idx_n1)
        rot_n2_map.add(idx_n2)

    # Exclude rotation-0 in count.
    return len(rot_n1_map) - 1, len(rot_n2_map) - 1


def _find_best_bsgs_ratio(non_zero_diags: Iterable[int], max_n: int, log_max_ratio: int) -> int:
    max_ratio = float(1 << log_max_ratio)
    n1 = 1
    while n1 < max_n:
        num_rot_n1, num_rot_n2 = _bsgs_index(non_zero_diags, max_n, n1)
        if num_rot_n1 == 0:
            return n1
        current_ratio = float(num_rot_n2) / float(num_rot_n1)
        if current_ratio == max_ratio:
            return n1
        if current_ratio > max_ratio:
            return n1 // 2 if n1 > 1 else 1
        n1 <<= 1
    return 1


def _collect_needed_rotations(a_diags: Iterable[int], n: int, b_step: int, slots: int) -> Set[int]:
    keylist = {x & (slots - 1) for x in a_diags}
    needed: Set[int] = set()
    g_step = (slots + b_step - 1) // b_step

    for i in range(1, b_step):
        if i in keylist:
            needed.add(norm_rot_index(i, n))

    for j in range(1, g_step):
        changed = False
        if b_step * j in keylist:
            changed = True
        for i in range(1, b_step):
            if b_step * j + i in keylist:
                needed.add(norm_rot_index(i, n))
                changed = True
        if changed:
            needed.add(norm_rot_index(b_step * j, n))

    return needed


def _collect_rotation_breakdown(a_diags: Iterable[int], n: int, b_step: int, slots: int) -> Tuple[Set[int], Set[int], Set[int]]:
    keylist = {x & (slots - 1) for x in a_diags}
    baby: Set[int] = set()
    giant: Set[int] = set()
    g_step = (slots + b_step - 1) // b_step

    for i in range(1, b_step):
        if i in keylist:
            baby.add(norm_rot_index(i, n))

    for j in range(1, g_step):
        changed = False
        if b_step * j in keylist:
            changed = True
        for i in range(1, b_step):
            if b_step * j + i in keylist:
                baby.add(norm_rot_index(i, n))
                changed = True
        if changed:
            giant.add(norm_rot_index(b_step * j, n))

    return baby | giant, baby, giant


def _collect_rotation_breakdown_structural_conj(a_diags: Iterable[int], n: int, b_step: int, slots: int) -> Tuple[Set[int], Set[int], Set[int]]:
    keylist = {x & (slots - 1) for x in a_diags}
    baby: Set[int] = set()
    giant: Set[int] = set()
    g_step = (slots + b_step - 1) // b_step

    for i in range(1, b_step):
        if i in keylist:
            baby.add(_canonical_conj_index(norm_rot_index(i, n), slots))

    for j in range(1, g_step):
        changed = False
        if b_step * j in keylist:
            changed = True
        for i in range(1, b_step):
            if b_step * j + i in keylist:
                baby.add(_canonical_conj_index(norm_rot_index(i, n), slots))
                changed = True
        if changed:
            giant.add(_canonical_conj_index(norm_rot_index(b_step * j, n), slots))

    return baby | giant, baby, giant


def evaluate_strategy(a_diags: List[int], n: int, bs_key: List[int], b_step: int, slots: int = 2**15, name: str = "") -> Dict[str, object]:
    needed, baby, giant = _collect_rotation_breakdown(a_diags, n, b_step, slots)
    bs_norm = {norm_rot_index(x & (slots - 1), n) for x in bs_key}
    new_only = needed - bs_norm
    return {
        "strategy": name or f"n1={b_step}",
        "b_step": b_step,
        "g_step": (slots + b_step - 1) // b_step,
        "needed_all": needed,
        "new_only": new_only,
        "reused": needed & bs_norm,
        "bs_norm": bs_norm,
        "needed_baby": baby,
        "needed_giant": giant,
    }


def evaluate_strategy_structural_conj(a_diags: List[int], n: int, bs_key: List[int], b_step: int, slots: int = 2**15, name: str = "") -> Dict[str, object]:
    needed, baby, giant = _collect_rotation_breakdown_structural_conj(a_diags, n, b_step, slots)
    bs_norm = _collapse_with_conjugation(bs_key, slots)
    new_only = needed - bs_norm
    return {
        "strategy": name or f"n1={b_step}",
        "b_step": b_step,
        "g_step": (slots + b_step - 1) // b_step,
        "needed_all": needed,
        "new_only": new_only,
        "reused": needed & bs_norm,
        "bs_norm": bs_norm,
        "needed_baby": baby,
        "needed_giant": giant,
        "fold_stage": "decomposition",
    }


def bsgs_evaluate_linear_transform(a_diags: List[int], n: int, bs_key: List[int], slots: int = 2**15) -> Dict[str, object]:
    # Ratio baseline in co-design style: choose bStep with A+bsKey, then evaluate on A.
    b_step = _find_best_bsgs_ratio(a_diags + bs_key, slots, log_max_ratio=0)
    return evaluate_strategy(a_diags, n, bs_key, b_step, slots, name="codesign_ratio(A+bsKey)")


def _candidate_bsteps(slots: int) -> List[int]:
    candidates: List[int] = []
    n1 = 1
    while n1 < slots:
        candidates.append(n1)
        n1 <<= 1
    return candidates


def _factorizations_of_power_of_two(total_log_slots: int, levels: int, min_log_step: int = 1) -> List[Tuple[int, ...]]:
    if levels <= 0:
        return []
    results: List[Tuple[int, ...]] = []

    def rec(remaining: int, depth: int, prefix: List[int]) -> None:
        if depth == levels:
            if remaining == 0:
                results.append(tuple(1 << x for x in prefix))
            return

        min_required = (levels - depth - 1) * min_log_step
        max_here = remaining - min_required
        for current in range(min_log_step, max_here + 1):
            prefix.append(current)
            rec(remaining - current, depth + 1, prefix)
            prefix.pop()

    rec(total_log_slots, 0, [])
    return results


def _decompose_rotation_steps(rot: int, steps: Sequence[int], slots: int) -> List[int]:
    remaining = rot & (slots - 1)
    stride = 1
    pieces: List[int] = []

    for step in steps[:-1]:
        digit = remaining % step
        pieces.append(digit * stride)
        remaining //= step
        stride *= step

    pieces.append(remaining * stride)
    return pieces


def _collect_rotation_breakdown_multistep(
    a_diags: Iterable[int],
    n: int,
    steps: Sequence[int],
    slots: int,
    use_conjugation: bool = False,
) -> Tuple[Set[int], List[Set[int]], int, float]:
    if not steps:
        raise ValueError("steps must not be empty")

    product = 1
    for step in steps:
        if step <= 0:
            raise ValueError("all steps must be positive")
        product *= step
    if product != slots:
        raise ValueError(f"product(steps) must equal slots, got {product} vs {slots}")

    stage_sets: List[Set[int]] = [set() for _ in steps]
    total_online_rotations = 0
    diag_count = 0

    for raw_rot in a_diags:
        diag_count += 1
        pieces = _decompose_rotation_steps(raw_rot, steps, slots)
        per_diag_non_zero = 0
        for idx, piece in enumerate(pieces):
            if piece == 0:
                continue
            rot_idx = norm_rot_index(piece, n)
            if use_conjugation:
                rot_idx = _canonical_conj_index(rot_idx, slots)
            stage_sets[idx].add(rot_idx)
            per_diag_non_zero += 1
        total_online_rotations += per_diag_non_zero

    needed_all: Set[int] = set()
    for stage_set in stage_sets:
        needed_all |= stage_set

    avg_online_rotations = float(total_online_rotations) / float(diag_count) if diag_count > 0 else 0.0
    return needed_all, stage_sets, total_online_rotations, avg_online_rotations


def _score_strategy(
    stat: Dict[str, object],
    mode: str,
    alpha: float,
    beta: float,
) -> Tuple[float, ...]:
    new_cnt = len(stat["new_only"])
    needed_cnt = len(stat["needed_all"])
    online_total = int(stat.get("online_rotations_total", needed_cnt))
    online_avg = float(stat.get("online_rotations_avg", float(needed_cnt)))

    if mode == "strict-min-new":
        return (new_cnt, needed_cnt, online_total, online_avg)
    if mode == "weighted":
        score = alpha * float(new_cnt) + beta * float(online_total)
        stat["score"] = score
        return (score, new_cnt, needed_cnt, online_total, online_avg)
    raise ValueError(f"Unsupported mode: {mode}")


def find_best_bstep_dual_mode(
    a_diags: List[int],
    n: int,
    bs_key: List[int],
    slots: int = 2**15,
    mode: str = "strict-min-new",
    alpha: float = 1.0,
    beta: float = 0.15,
) -> Dict[str, object]:
    """
    mode = "strict-min-new":
        optimize only new keys, tie-break by fewer needed_all then smaller b_step.
    mode = "weighted":
        optimize alpha*|new_only| + beta*|needed_all|.
        beta controls how much we care about online rotation overhead.
    """
    best = None
    for c in _candidate_bsteps(slots):
        stat = evaluate_strategy(a_diags, n, bs_key, c, slots, name=f"scan_n1={c}")
        new_cnt = len(stat["new_only"])
        needed_cnt = len(stat["needed_all"])

        if mode == "strict-min-new":
            metric = (new_cnt, needed_cnt, c)
        elif mode == "weighted":
            score = alpha * float(new_cnt) + beta * float(needed_cnt)
            stat["score"] = score
            metric = (score, new_cnt, needed_cnt, c)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if best is None or metric < best[0]:
            best = (metric, stat)

    chosen = best[1]
    chosen["mode"] = mode
    chosen["alpha"] = alpha
    chosen["beta"] = beta
    return chosen


def find_best_bstep_structural_conj(
    a_diags: List[int],
    n: int,
    bs_key: List[int],
    slots: int = 2**15,
    mode: str = "strict-min-new",
    alpha: float = 1.0,
    beta: float = 0.15,
) -> Dict[str, object]:
    best = None
    for c in _candidate_bsteps(slots):
        stat = evaluate_strategy_structural_conj(a_diags, n, bs_key, c, slots, name=f"scan_struct_n1={c}")
        new_cnt = len(stat["new_only"])
        needed_cnt = len(stat["needed_all"])

        if mode == "strict-min-new":
            metric = (new_cnt, needed_cnt, c)
        elif mode == "weighted":
            score = alpha * float(new_cnt) + beta * float(needed_cnt)
            stat["score"] = score
            metric = (score, new_cnt, needed_cnt, c)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if best is None or metric < best[0]:
            best = (metric, stat)

    chosen = best[1]
    chosen["mode"] = mode
    chosen["alpha"] = alpha
    chosen["beta"] = beta
    chosen["optimized_for"] = "structural-conj-fold"
    return chosen


def evaluate_strategy_multistep(
    a_diags: List[int],
    n: int,
    bs_key: List[int],
    steps: Sequence[int],
    slots: int = 2**15,
    name: str = "",
    use_conjugation: bool = False,
) -> Dict[str, object]:
    needed_all, stage_sets, online_total, online_avg = _collect_rotation_breakdown_multistep(
        a_diags,
        n,
        steps,
        slots,
        use_conjugation=use_conjugation,
    )
    bs_norm = _collapse_with_conjugation(bs_key, slots) if use_conjugation else {norm_rot_index(x & (slots - 1), n) for x in bs_key}
    new_only = needed_all - bs_norm
    final_keyindex = new_only | bs_norm

    stat: Dict[str, object] = {
        "strategy": name or f"multistep={tuple(steps)}",
        "steps": tuple(steps),
        "levels": len(steps),
        "needed_all": needed_all,
        "new_only": new_only,
        "reused": needed_all & bs_norm,
        "bs_norm": bs_norm,
        "final_keyindex": final_keyindex,
        "needed_stages": stage_sets,
        "online_rotations_total": online_total,
        "online_rotations_avg": online_avg,
        "use_conjugation": use_conjugation,
    }

    for idx, stage_set in enumerate(stage_sets, start=1):
        stat[f"needed_stage_{idx}"] = stage_set

    if len(steps) == 2:
        stat["b_step"] = steps[0]
        stat["g_step"] = steps[1]
        stat["needed_baby"] = stage_sets[0]
        stat["needed_giant"] = stage_sets[1]
    else:
        stat["needed_baby"] = stage_sets[0]
        stat["needed_giant"] = stage_sets[-1]

    return stat


def find_best_multistep_strategy(
    a_diags: List[int],
    n: int,
    bs_key: List[int],
    slots: int = 2**15,
    levels: int = 3,
    mode: str = "weighted",
    alpha: float = 1.0,
    beta: float = 0.15,
    use_conjugation: bool = False,
    min_log_step: int = 1,
) -> Dict[str, object]:
    total_log_slots = slots.bit_length() - 1
    if (1 << total_log_slots) != slots:
        raise ValueError("find_best_multistep_strategy currently expects slots to be a power of two")

    best = None
    for steps in _factorizations_of_power_of_two(total_log_slots, levels, min_log_step=min_log_step):
        stat = evaluate_strategy_multistep(
            a_diags,
            n,
            bs_key,
            steps,
            slots=slots,
            name=f"scan_multistep(levels={levels},steps={steps})",
            use_conjugation=use_conjugation,
        )
        metric = _score_strategy(stat, mode, alpha, beta)
        metric = metric + (steps,)

        if best is None or metric < best[0]:
            best = (metric, stat)

    chosen = best[1]
    chosen["mode"] = mode
    chosen["alpha"] = alpha
    chosen["beta"] = beta
    chosen["optimized_for"] = "multistep-conj" if use_conjugation else "multistep"
    return chosen


def print_stats(title: str, stat: Dict[str, object]) -> None:
    print(f"[{title}] strategy={stat['strategy']}")
    if "steps" in stat:
        print(f"steps={stat['steps']}, levels={stat['levels']}")
    else:
        print(f"bStep={stat['b_step']}, gStep={stat['g_step']}")
    print(f"needed_all={len(stat['needed_all'])}, new_only={len(stat['new_only'])}, reused={len(stat['reused'])}")
    print(f"needed_baby={len(stat['needed_baby'])}, needed_giant={len(stat['needed_giant'])}")
    if "online_rotations_total" in stat:
        print(f"online_rotations_total={stat['online_rotations_total']}, online_rotations_avg={stat['online_rotations_avg']:.4f}")
    if "score" in stat:
        print(f"score(alpha*new + beta*online_total)={stat['score']:.4f} (alpha={stat['alpha']}, beta={stat['beta']})")


def _format_steps_for_table(stat: Dict[str, object]) -> str:
    if "steps" in stat:
        return " x ".join(str(x) for x in stat["steps"])
    if "b_step" in stat and "g_step" in stat:
        return f"{stat['b_step']} x {stat['g_step']}"
    return "-"


def _collect_stage_sizes(stat: Dict[str, object]) -> str:
    stage_sets = stat.get("needed_stages")
    if stage_sets is not None:
        return "/".join(str(len(x)) for x in stage_sets)
    parts: List[str] = []
    if "needed_baby" in stat:
        parts.append(str(len(stat["needed_baby"])))
    if "needed_giant" in stat:
        parts.append(str(len(stat["needed_giant"])))
    return "/".join(parts) if parts else "-"


def print_strategy_table(title: str, rows: Sequence[Tuple[str, Dict[str, object]]]) -> None:
    headers = [
        "label",
        "steps",
        "levels",
        "stage_sizes",
        "avg_rot",
        "needed",
        "new",
        "reused",
        "final",
        "score",
    ]
    table_rows: List[List[str]] = []
    for label, stat in rows:
        score = stat.get("score")
        table_rows.append([
            label,
            _format_steps_for_table(stat),
            str(stat.get("levels", 2)),
            _collect_stage_sizes(stat),
            f"{float(stat.get('online_rotations_avg', 2.0)):.4f}",
            str(len(stat["needed_all"])),
            str(len(stat["new_only"])),
            str(len(stat["reused"])),
            str(len(stat.get("final_keyindex", stat["new_only"] | stat["bs_norm"]))),
            f"{float(score):.4f}" if score is not None else "-",
        ])

    widths = [len(header) for header in headers]
    for row in table_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt(row: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))

    print(f"\n[{title}]")
    print(fmt(headers))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(fmt(row))


def _format_int_values(values: Iterable[int]) -> str:
    return " ".join(str(x) for x in sorted(values))


def _strategy_csv_row(experiment: str, label: str, stat: Dict[str, object]) -> Dict[str, object]:
    final_keyindex = stat.get("final_keyindex", stat["new_only"] | stat["bs_norm"])
    score = stat.get("score")
    return {
        "experiment": experiment,
        "label": label,
        "strategy": stat["strategy"],
        "mode": stat.get("mode", ""),
        "optimized_for": stat.get("optimized_for", ""),
        "use_conjugation": stat.get("use_conjugation", ""),
        "levels": stat.get("levels", 2),
        "steps": _format_steps_for_table(stat),
        "stage_sizes": _collect_stage_sizes(stat),
        "alpha": stat.get("alpha", ""),
        "beta": stat.get("beta", ""),
        "score": f"{float(score):.8f}" if score is not None else "",
        "needed_count": len(stat["needed_all"]),
        "new_count": len(stat["new_only"]),
        "reused_count": len(stat["reused"]),
        "final_keyindex_count": len(final_keyindex),
        "online_rotations_total": stat.get("online_rotations_total", ""),
        "online_rotations_avg": f"{float(stat['online_rotations_avg']):.8f}" if "online_rotations_avg" in stat else "",
        "needed_baby_count": len(stat["needed_baby"]) if "needed_baby" in stat else "",
        "needed_giant_count": len(stat["needed_giant"]) if "needed_giant" in stat else "",
        "needed_all": _format_int_values(stat["needed_all"]),
        "new_only": _format_int_values(stat["new_only"]),
        "reused": _format_int_values(stat["reused"]),
        "final_keyindex": _format_int_values(final_keyindex),
    }


def write_results_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _best_scored_row(rows: Sequence[Tuple[str, Dict[str, object]]]) -> Tuple[str, Dict[str, object]]:
    return min(
        rows,
        key=lambda item: (
            float(item[1].get("score", 0.0)),
            len(item[1]["new_only"]),
            int(item[1].get("online_rotations_total", 0)),
            len(item[1]["needed_all"]),
        ),
    )


def _parse_list_of_lists(raw: str) -> List[List[int]]:
    groups: List[List[int]] = []
    current: List[int] = []
    token = ""
    depth = 0

    for ch in raw:
        if ch == "[":
            depth += 1
            if depth == 1:
                current = []
                token = ""
        elif ch == "]":
            if depth == 1:
                if token.strip():
                    current.append(int(token.strip()))
                    token = ""
                groups.append(current)
            depth = max(0, depth - 1)
        elif depth == 1:
            if ch == ",":
                if token.strip():
                    current.append(int(token.strip()))
                    token = ""
            elif ch.isdigit() or ch == "-":
                token += ch
            elif ch in "\n\r\t ":
                # Ignore whitespace to tolerate pasted line wraps inside numbers.
                continue
            else:
                # Ignore any non-numeric separators in pasted text.
                if token.strip():
                    current.append(int(token.strip()))
                    token = ""

    return groups


def _sorted_int_set(values: Set[int]) -> List[int]:
    return sorted(values)


def _canonical_conj_index(x: int, slots: int) -> int:
    x &= (slots - 1)
    if x == 0:
        return 0
    neg = (-x) & (slots - 1)
    return x if x < neg else neg


def _collapse_with_conjugation(values: Iterable[int], slots: int) -> Set[int]:
    return {_canonical_conj_index(v, slots) for v in values if (v & (slots - 1)) != 0}


def evaluate_with_conjugation(base_stat: Dict[str, object], bs_key: List[int], slots: int) -> Dict[str, object]:
    needed_baby = _collapse_with_conjugation(base_stat["needed_baby"], slots)
    needed_giant = _collapse_with_conjugation(base_stat["needed_giant"], slots)
    needed_all = needed_baby | needed_giant
    bs_norm = _collapse_with_conjugation(bs_key, slots)
    new_only = needed_all - bs_norm
    reused = needed_all & bs_norm
    final_keyindex = new_only | bs_norm
    return {
        "strategy": f"{base_stat['strategy']}+new_conjugation_3step",
        "base_strategy": base_stat["strategy"],
        "b_step": base_stat["b_step"],
        "g_step": base_stat["g_step"],
        "needed_all": needed_all,
        "needed_baby": needed_baby,
        "needed_giant": needed_giant,
        "new_only": new_only,
        "reused": reused,
        "bs_norm": bs_norm,
        "final_keyindex": final_keyindex,
        "extra_step": "conjugation",
    }


if __name__ == "__main__":
    bsKey = [8, 64, 512, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32320, 32384, 32448, 32512, 32576, 32640, 32704, 32712, 32720, 32728, 32736, 32744, 32752, 32760, 32761, 32762, 32763, 32764, 32765, 32766, 32767]
    n = 2**16
    all_a_diags_raw = """
[0, 1, 31, 32, 33, 991, 992, 993, 1023, 1024, 1025, 1055, 1056, 1057, 2015, 2016, 2017, 2047, 2048, 2049, 2079, 2080, 2081, 3039, 3040, 3041, 3071, 3072, 3073, 3103, 3104, 3105, 4063, 4064, 4065, 4095, 4096, 4097, 4127, 4128, 4129, 5087, 5088, 5089, 5119, 5120, 5121, 5151, 5152, 5153, 6111, 6112, 6113, 6143, 6144, 6145, 6175, 6176, 6177, 7135, 7136, 7137, 7167, 7168, 7169, 7199, 7200, 7201, 8159, 8160, 8161, 8191, 8192, 8193, 8223, 8224, 8225, 9183, 9184, 9185, 9215, 9216, 9217, 9247, 9248, 9249, 10207, 10208, 10209, 10239, 10240, 10241, 10271, 10272, 10273, 11231, 11232, 11233, 11263, 11264, 11265, 11295, 11296, 11297, 12255, 12256, 12257, 12287, 12288, 12289, 12319, 12320, 12321, 13279, 13280, 13281, 13311, 13312, 13313, 13343, 13344, 13345, 14303, 14304, 14305, 14335, 14336, 14337, 14367, 14368, 14369, 15327, 15328, 15329, 15359, 15360, 15361, 15391, 15392, 15393, 16351, 16352, 16353, 16383]
[0, 1, 31, 32, 33, 991, 992, 993, 1023, 1024, 1025, 1055, 1056, 1057, 2015, 2016, 2017, 2047, 2048, 2049, 2079, 2080, 2081, 3039, 3040, 3041, 3071, 3072, 3073, 3103, 3104, 3105, 4063, 4064, 4065, 4095, 4096, 4097, 4127, 4128, 4129, 5087, 5088, 5089, 5119, 5120, 5121, 5151, 5152, 5153, 6111, 6112, 6113, 6143, 6144, 6145, 6175, 6176, 6177, 7135, 7136, 7137, 7167, 7168, 7169, 7199, 7200, 7201, 8159, 8160, 8161, 8191, 8192, 8193, 8223, 8224, 8225, 9183, 9184, 9185, 9215, 9216, 9217, 9247, 9248, 9249, 10207, 10208, 10209, 10239, 10240, 10241, 10271, 10272, 10273, 11231, 11232, 11233, 11263, 11264, 11265, 11295, 11296, 11297, 12255, 12256, 12257, 12287, 12288, 12289, 12319, 12320, 12321, 13279, 13280, 13281, 13311, 13312, 13313, 13343, 13344, 13345, 14303, 14304, 14305, 14335, 14336, 14337, 14367, 14368, 14369, 15327, 15328, 15329, 15359, 15360, 15361, 15391, 15392, 15393, 16351, 16352, 16353, 16383]
[0, 1, 31, 32, 33, 991, 992, 993, 1023, 1024, 1025, 1055, 1056, 1057, 2015, 2016, 2017, 2047, 2048, 2049, 2079, 2080, 2081, 3039, 3040, 3041, 3071, 3072, 3073, 3103, 3104, 3105, 4063, 4064, 4065, 4095, 4096, 4097, 4127, 4128, 4129, 5087, 5088, 5089, 5119, 5120, 5121, 5151, 5152, 5153, 6111, 6112, 6113, 6143, 6144, 6145, 6175, 6176, 6177, 7135, 7136, 7137, 7167, 7168, 7169, 7199, 7200, 7201, 8159, 8160, 8161, 8191, 8192, 8193, 8223, 8224, 8225, 9183, 9184, 9185, 9215, 9216, 9217, 9247, 9248, 9249, 10207, 10208, 10209, 10239, 10240, 10241, 10271, 10272, 10273, 11231, 11232, 11233, 11263, 11264, 11265, 11295, 11296, 11297, 12255, 12256, 12257, 12287, 12288, 12289, 12319, 12320, 12321, 13279, 13280, 13281, 13311, 13312, 13313, 13343, 13344, 13345, 14303, 14304, 14305, 14335, 14336, 14337, 14367, 14368, 14369, 15327, 15328, 15329, 15359, 15360, 15361, 15391, 15392, 15393, 16351, 16352, 16353, 16383]
[0, 1, 31, 32, 33, 991, 992, 993, 1023, 1024, 1025, 1055, 1056, 1057, 2015, 2016, 2017, 2047, 2048, 2049, 2079, 2080, 2081, 3039, 3040, 3041, 3071, 3072, 3073, 3103, 3104, 3105, 4063, 4064, 4065, 4095, 4096, 4097, 4127, 4128, 4129, 5087, 5088, 5089, 5119, 5120, 5121, 5151, 5152, 5153, 6111, 6112, 6113, 6143, 6144, 6145, 6175, 6176, 6177, 7135, 7136, 7137, 7167, 7168, 7169, 7199, 7200, 7201, 8159, 8160, 8161, 8191, 8192, 8193, 8223, 8224, 8225, 9183, 9184, 9185, 9215, 9216, 9217, 9247, 9248, 9249, 10207, 10208, 10209, 10239, 10240, 10241, 10271, 10272, 10273, 11231, 11232, 11233, 11263, 11264, 11265, 11295, 11296, 11297, 12255, 12256, 12257, 12287, 12288, 12289, 12319, 12320, 12321, 13279, 13280, 13281, 13311, 13312, 13313, 13343, 13344, 13345, 14303, 14304, 14305, 14335, 14336, 14337, 14367, 14368, 14369, 15327, 15328, 15329, 15359, 15360, 15361, 15391, 15392, 15393, 16351, 16352, 16353, 16383]
[0, 1, 31, 32, 33, 991, 992, 993, 1023, 1024, 1025, 1055, 1056, 1057, 2015, 2016, 2017, 2047, 2048, 2049, 2079, 2080, 2081, 3039, 3040, 3041, 3071, 3072, 3073, 3103, 3104, 3105, 4063, 4064, 4065, 4095, 4096, 4097, 4127, 4128, 4129, 5087, 
5088, 5089, 5119, 5120, 5121, 5151, 5152, 5153, 6111, 6112, 6113, 6143, 6144, 6145, 6175, 6176, 6177, 7135, 7136, 7137, 7167, 7168, 7169, 7199, 7200, 7201, 8159, 8160, 8161, 8191, 8192, 8193, 8223, 8224, 8225, 9183, 9184, 9185, 9215, 9216
, 9217, 9247, 9248, 9249, 10207, 10208, 10209, 10239, 10240, 10241, 10271, 10272, 10273, 11231, 11232, 11233, 11263, 11264, 11265, 11295, 11296, 11297, 12255, 12256, 12257, 12287, 12288, 12289, 12319, 12320, 12321, 13279, 13280, 13281, 13
311, 13312, 13313, 13343, 13344, 13345, 14303, 14304, 14305, 14335, 14336, 14337, 14367, 14368, 14369, 15327, 15328, 15329, 15359, 15360, 15361, 15391, 15392, 15393, 16351, 16352, 16353, 16383]
[0, 1, 31, 32, 33, 991, 992, 993, 1023, 1024, 1025, 1055, 1056, 1057, 2015, 2016, 2017, 2047, 2048, 2049, 2079, 2080, 2081, 3039, 3040, 3041, 3071, 3072, 3073, 3103, 3104, 3105, 4063, 4064, 4065, 4095, 4096, 4097, 4127, 4128, 4129, 5087, 
5088, 5089, 5119, 5120, 5121, 5151, 5152, 5153, 6111, 6112, 6113, 6143, 6144, 6145, 6175, 6176, 6177, 7135, 7136, 7137, 7167, 7168, 7169, 7199, 7200, 7201, 8159, 8160, 8161, 8191, 8192, 8193, 8223, 8224, 8225, 9183, 9184, 9185, 9215, 9216
, 9217, 9247, 9248, 9249, 10207, 10208, 10209, 10239, 10240, 10241, 10271, 10272, 10273, 11231, 11232, 11233, 11263, 11264, 11265, 11295, 11296, 11297, 12255, 12256, 12257, 12287, 12288, 12289, 12319, 12320, 12321, 13279, 13280, 13281, 13
311, 13312, 13313, 13343, 13344, 13345, 14303, 14304, 14305, 14335, 14336, 14337, 14367, 14368, 14369, 15327, 15328, 15329, 15359, 15360, 15361, 15391, 15392, 15393, 16351, 16352, 16353, 16383]
[0, 1, 31, 32, 33, 991, 992, 993, 1023, 1024, 1025, 1055, 1056, 1057, 2015, 2016, 2017, 2047, 2048, 2049, 2079, 2080, 2081, 3039, 3040, 3041, 3071, 3072, 3073, 3103, 3104, 3105, 4063, 4064, 4065, 4095, 4096, 4097, 4127, 4128, 4129, 5087, 
5088, 5089, 5119, 5120, 5121, 5151, 5152, 5153, 6111, 6112, 6113, 6143, 6144, 6145, 6175, 6176, 6177, 7135, 7136, 7137, 7167, 7168, 7169, 7199, 7200, 7201, 8159, 8160, 8161, 8191, 8192, 8193, 8223, 8224, 8225, 9183, 9184, 9185, 9215, 9216
, 9217, 9247, 9248, 9249, 10207, 10208, 10209, 10239, 10240, 10241, 10271, 10272, 10273, 11231, 11232, 11233, 11263, 11264, 11265, 11295, 11296, 11297, 12255, 12256, 12257, 12287, 12288, 12289, 12319, 12320, 12321, 13279, 13280, 13281, 13
311, 13312, 13313, 13343, 13344, 13345, 14303, 14304, 14305, 14335, 14336, 14337, 14367, 14368, 14369, 15327, 15328, 15329, 15359, 15360, 15361, 15391, 15392, 15393, 16351, 16352, 16353, 16383]
[0, 1, 30, 31, 32, 33, 958, 959, 960, 961, 990, 991, 992, 993, 1022, 1023, 1024, 1025, 1054, 1055, 1056, 1057, 1982, 1983, 1984, 1985, 2014, 2015, 2016, 2017, 2046, 2047, 2048, 2049, 2078, 2079, 2080, 2081, 3006, 3007, 3008, 3009, 3038, 3
039, 3040, 3041, 3070, 3071, 3072, 3073, 3102, 3103, 3104, 3105, 4030, 4031, 4032, 4033, 4062, 4063, 4064, 4065, 4094, 4095, 4096, 4097, 4126, 4127, 4128, 4129, 5054, 5055, 5056, 5057, 5086, 5087, 5088, 5089, 5118, 5119, 5120, 5121, 5150,
 5151, 5152, 5153, 6078, 6079, 6080, 6081, 6110, 6111, 6112, 6113, 6142, 6143, 6144, 6145, 6174, 6175, 6176, 6177, 7102, 7103, 7104, 7105, 7134, 7135, 7136, 7137, 7166, 7167, 7168, 7169, 7198, 7199, 7200, 7201, 8126, 8127, 8128, 8129, 815
8, 8159, 8160, 8161, 8190, 8191]
[0, 1, 2, 3, 29, 30, 31, 32, 33, 34, 35, 61, 62, 63, 64, 65, 66, 67, 93, 94, 95, 96, 97, 98, 99, 925, 926, 927, 928, 929, 930, 931, 957, 958, 959, 960, 961, 962, 963, 989, 990, 991, 992, 993, 994, 995, 1021, 1022, 1023, 1024, 1025, 1026, 
1027, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 2013, 2014, 2015, 2016
, 2017, 2018, 2019, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 3005, 30
06, 3007, 3008, 3009, 3010, 3011, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3165, 3166, 3167, 3168, 3169, 3170, 
3171, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4093, 4094, 4095, 4096, 4097, 4098, 4099, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4157, 4158, 4159, 4160
, 4161, 4162, 4163, 4189, 4190, 4191, 4192, 4193, 4194, 4195, 5021, 5022, 5023, 5024, 5025, 5026, 5027, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5085, 5086, 5087, 5088, 5089, 5090, 5091, 5117, 5118, 5119, 5120, 5121, 5122, 5123, 5149, 51
50, 5151, 5152, 5153, 5154, 5155, 5181, 5182, 5183, 5184, 5185, 5186, 5187, 5213, 5214, 5215, 5216, 5217, 5218, 5219, 6045, 6046, 6047, 6048, 6049, 6050, 6051, 6077, 6078, 6079, 6080, 6081, 6082, 6083, 6109, 6110, 6111, 6112, 6113, 6114, 
6115, 6141, 6142, 6143, 6144, 6145, 6146, 6147, 6173, 6174, 6175, 6176, 6177, 6178, 6179, 6205, 6206, 6207, 6208, 6209, 6210, 6211, 6237, 6238, 6239, 6240, 6241, 6242, 6243, 7069, 7070, 7071, 7072, 7073, 7074, 7075, 7101, 7102, 7103, 7104
, 7105, 7106, 7107, 7133, 7134, 7135, 7136, 7137, 7138, 7139, 7165, 7166, 7167, 7168, 7169, 7170, 7171, 7197, 7198, 7199, 7200, 7201, 7202, 7203, 7229, 7230, 7231, 7232, 7233, 7234, 7235, 7261, 7262, 7263, 7264, 7265, 7266, 7267, 8093, 80
94, 8095, 8096, 8097, 8098, 8099, 8125, 8126, 8127, 8128, 8129, 8130, 8131, 8157, 8158, 8159, 8160, 8161, 8162, 8163, 8189, 8190, 8191]
[0, 991, 992, 1023, 1024, 2015, 2016, 2047, 2048, 3039, 3040, 3071, 3072, 4063, 4064, 4095, 4096, 5087, 5088, 5119, 5120, 6111, 6112, 6143, 6144, 7135, 7136, 7167, 7168, 8159, 8160, 8191]
[0, 1, 2, 3, 29, 30, 31, 32, 33, 34, 35, 61, 62, 63, 64, 65, 66, 67, 93, 94, 95, 96, 97, 98, 99, 925, 926, 927, 928, 929, 930, 931, 957, 958, 959, 960, 961, 962, 963, 989, 990, 991, 992, 993, 994, 995, 1021, 1022, 1023, 1024, 1025, 1026, 
1027, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 2013, 2014, 2015, 2016
, 2017, 2018, 2019, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 3005, 30
06, 3007, 3008, 3009, 3010, 3011, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3165, 3166, 3167, 3168, 3169, 3170, 
3171, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4093, 4094, 4095, 4096, 4097, 4098, 4099, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4157, 4158, 4159, 4160
, 4161, 4162, 4163, 4189, 4190, 4191, 4192, 4193, 4194, 4195, 5021, 5022, 5023, 5024, 5025, 5026, 5027, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5085, 5086, 5087, 5088, 5089, 5090, 5091, 5117, 5118, 5119, 5120, 5121, 5122, 5123, 5149, 51
50, 5151, 5152, 5153, 5154, 5155, 5181, 5182, 5183, 5184, 5185, 5186, 5187, 5213, 5214, 5215, 5216, 5217, 5218, 5219, 6045, 6046, 6047, 6048, 6049, 6050, 6051, 6077, 6078, 6079, 6080, 6081, 6082, 6083, 6109, 6110, 6111, 6112, 6113, 6114, 
6115, 6141, 6142, 6143, 6144, 6145, 6146, 6147, 6173, 6174, 6175, 6176, 6177, 6178, 6179, 6205, 6206, 6207, 6208, 6209, 6210, 6211, 6237, 6238, 6239, 6240, 6241, 6242, 6243, 7069, 7070, 7071, 7072, 7073, 7074, 7075, 7101, 7102, 7103, 7104
, 7105, 7106, 7107, 7133, 7134, 7135, 7136, 7137, 7138, 7139, 7165, 7166, 7167, 7168, 7169, 7170, 7171, 7197, 7198, 7199, 7200, 7201, 7202, 7203, 7229, 7230, 7231, 7232, 7233, 7234, 7235, 7261, 7262, 7263, 7264, 7265, 7266, 7267, 8093, 80
94, 8095, 8096, 8097, 8098, 8099, 8125, 8126, 8127, 8128, 8129, 8130, 8131, 8157, 8158, 8159, 8160, 8161, 8162, 8163, 8189, 8190, 8191]
[0, 1, 2, 3, 29, 30, 31, 32, 33, 34, 35, 61, 62, 63, 64, 65, 66, 67, 93, 94, 95, 96, 97, 98, 99, 925, 926, 927, 928, 929, 930, 931, 957, 958, 959, 960, 961, 962, 963, 989, 990, 991, 992, 993, 994, 995, 1021, 1022, 1023, 1024, 1025, 1026, 
1027, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 2013, 2014, 2015, 2016
, 2017, 2018, 2019, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 3005, 30
06, 3007, 3008, 3009, 3010, 3011, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3165, 3166, 3167, 3168, 3169, 3170, 
3171, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4093, 4094, 4095, 4096, 4097, 4098, 4099, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4157, 4158, 4159, 4160
, 4161, 4162, 4163, 4189, 4190, 4191, 4192, 4193, 4194, 4195, 5021, 5022, 5023, 5024, 5025, 5026, 5027, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5085, 5086, 5087, 5088, 5089, 5090, 5091, 5117, 5118, 5119, 5120, 5121, 5122, 5123, 5149, 51
50, 5151, 5152, 5153, 5154, 5155, 5181, 5182, 5183, 5184, 5185, 5186, 5187, 5213, 5214, 5215, 5216, 5217, 5218, 5219, 6045, 6046, 6047, 6048, 6049, 6050, 6051, 6077, 6078, 6079, 6080, 6081, 6082, 6083, 6109, 6110, 6111, 6112, 6113, 6114, 
6115, 6141, 6142, 6143, 6144, 6145, 6146, 6147, 6173, 6174, 6175, 6176, 6177, 6178, 6179, 6205, 6206, 6207, 6208, 6209, 6210, 6211, 6237, 6238, 6239, 6240, 6241, 6242, 6243, 7069, 7070, 7071, 7072, 7073, 7074, 7075, 7101, 7102, 7103, 7104
, 7105, 7106, 7107, 7133, 7134, 7135, 7136, 7137, 7138, 7139, 7165, 7166, 7167, 7168, 7169, 7170, 7171, 7197, 7198, 7199, 7200, 7201, 7202, 7203, 7229, 7230, 7231, 7232, 7233, 7234, 7235, 7261, 7262, 7263, 7264, 7265, 7266, 7267, 8093, 80
94, 8095, 8096, 8097, 8098, 8099, 8125, 8126, 8127, 8128, 8129, 8130, 8131, 8157, 8158, 8159, 8160, 8161, 8162, 8163, 8189, 8190, 8191]
[0, 1, 2, 3, 29, 30, 31, 32, 33, 34, 35, 61, 62, 63, 64, 65, 66, 67, 93, 94, 95, 96, 97, 98, 99, 925, 926, 927, 928, 929, 930, 931, 957, 958, 959, 960, 961, 962, 963, 989, 990, 991, 992, 993, 994, 995, 1021, 1022, 1023, 1024, 1025, 1026, 
1027, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 2013, 2014, 2015, 2016
, 2017, 2018, 2019, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 3005, 30
06, 3007, 3008, 3009, 3010, 3011, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3165, 3166, 3167, 3168, 3169, 3170, 
3171, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4093, 4094, 4095, 4096, 4097, 4098, 4099, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4157, 4158, 4159, 4160
, 4161, 4162, 4163, 4189, 4190, 4191, 4192, 4193, 4194, 4195, 5021, 5022, 5023, 5024, 5025, 5026, 5027, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5085, 5086, 5087, 5088, 5089, 5090, 5091, 5117, 5118, 5119, 5120, 5121, 5122, 5123, 5149, 51
50, 5151, 5152, 5153, 5154, 5155, 5181, 5182, 5183, 5184, 5185, 5186, 5187, 5213, 5214, 5215, 5216, 5217, 5218, 5219, 6045, 6046, 6047, 6048, 6049, 6050, 6051, 6077, 6078, 6079, 6080, 6081, 6082, 6083, 6109, 6110, 6111, 6112, 6113, 6114, 
6115, 6141, 6142, 6143, 6144, 6145, 6146, 6147, 6173, 6174, 6175, 6176, 6177, 6178, 6179, 6205, 6206, 6207, 6208, 6209, 6210, 6211, 6237, 6238, 6239, 6240, 6241, 6242, 6243, 7069, 7070, 7071, 7072, 7073, 7074, 7075, 7101, 7102, 7103, 7104
, 7105, 7106, 7107, 7133, 7134, 7135, 7136, 7137, 7138, 7139, 7165, 7166, 7167, 7168, 7169, 7170, 7171, 7197, 7198, 7199, 7200, 7201, 7202, 7203, 7229, 7230, 7231, 7232, 7233, 7234, 7235, 7261, 7262, 7263, 7264, 7265, 7266, 7267, 8093, 80
94, 8095, 8096, 8097, 8098, 8099, 8125, 8126, 8127, 8128, 8129, 8130, 8131, 8157, 8158, 8159, 8160, 8161, 8162, 8163, 8189, 8190, 8191]
[0, 1, 2, 3, 29, 30, 31, 32, 33, 34, 35, 61, 62, 63, 64, 65, 66, 67, 93, 94, 95, 96, 97, 98, 99, 925, 926, 927, 928, 929, 930, 931, 957, 958, 959, 960, 961, 962, 963, 989, 990, 991, 992, 993, 994, 995, 1021, 1022, 1023, 1024, 1025, 1026, 
1027, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 2013, 2014, 2015, 2016
, 2017, 2018, 2019, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 3005, 30
06, 3007, 3008, 3009, 3010, 3011, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3165, 3166, 3167, 3168, 3169, 3170, 
3171, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4093, 4094, 4095, 4096, 4097, 4098, 4099, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4157, 4158, 4159, 4160
, 4161, 4162, 4163, 4189, 4190, 4191, 4192, 4193, 4194, 4195, 5021, 5022, 5023, 5024, 5025, 5026, 5027, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5085, 5086, 5087, 5088, 5089, 5090, 5091, 5117, 5118, 5119, 5120, 5121, 5122, 5123, 5149, 51
50, 5151, 5152, 5153, 5154, 5155, 5181, 5182, 5183, 5184, 5185, 5186, 5187, 5213, 5214, 5215, 5216, 5217, 5218, 5219, 6045, 6046, 6047, 6048, 6049, 6050, 6051, 6077, 6078, 6079, 6080, 6081, 6082, 6083, 6109, 6110, 6111, 6112, 6113, 6114, 
6115, 6141, 6142, 6143, 6144, 6145, 6146, 6147, 6173, 6174, 6175, 6176, 6177, 6178, 6179, 6205, 6206, 6207, 6208, 6209, 6210, 6211, 6237, 6238, 6239, 6240, 6241, 6242, 6243, 7069, 7070, 7071, 7072, 7073, 7074, 7075, 7101, 7102, 7103, 7104
, 7105, 7106, 7107, 7133, 7134, 7135, 7136, 7137, 7138, 7139, 7165, 7166, 7167, 7168, 7169, 7170, 7171, 7197, 7198, 7199, 7200, 7201, 7202, 7203, 7229, 7230, 7231, 7232, 7233, 7234, 7235, 7261, 7262, 7263, 7264, 7265, 7266, 7267, 8093, 80
94, 8095, 8096, 8097, 8098, 8099, 8125, 8126, 8127, 8128, 8129, 8130, 8131, 8157, 8158, 8159, 8160, 8161, 8162, 8163, 8189, 8190, 8191]
[0, 1, 2, 3, 27, 28, 29, 30, 31, 32, 33, 34, 35, 59, 60, 61, 62, 63, 64, 65, 66, 67, 91, 92, 93, 94, 95, 96, 97, 98, 99, 859, 860, 861, 862, 863, 864, 865, 866, 867, 891, 892, 893, 894, 895, 896, 897, 898, 899, 923, 924, 925, 926, 927, 92
8, 929, 930, 931, 955, 956, 957, 958, 959, 960, 961, 962, 963, 987, 988, 989, 990, 991, 992, 993, 994, 995, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1083, 1084, 1085, 1086
, 1087, 1088, 1089, 1090, 1091, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 19
54, 1955, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2107, 2108, 
2109, 2110, 2111, 2112, 2113, 2114, 2115, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2907, 2908, 2909, 2910, 2911, 2912, 2913, 2914, 2915, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2971, 2972, 2973, 2974, 2975, 2976
, 2977, 2978, 2979, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 31
31, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3163, 3164, 3165, 3166, 3167, 3168, 3169, 3170, 3171, 3931, 3932, 3933, 3934, 3935, 3936, 3937, 3938, 3939, 3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 3995, 3996, 3997, 3998, 
3999, 4000, 4001, 4002, 4003, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4091, 4092, 4093, 4094, 4095]
[0, 1, 2, 3, 4, 5, 6, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 121, 122, 123, 124, 125, 126,
 127, 128, 129, 130, 131, 132, 133, 134, 135, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 217, 218, 219, 220, 221, 222, 223, 224, 22
5, 226, 227, 228, 229, 230, 231, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 
868, 869, 870, 871, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966
, 967, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 
1060, 1061, 1062, 1063, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1145, 1146, 1147, 1148, 1149, 1150
, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 12
41, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 
1859, 1860, 1861, 1862, 1863, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1945, 1946, 1947, 1948, 1949
, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 20
23, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 
2114, 2115, 2116, 2117, 2118, 2119, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2201, 2202, 2203, 2204
, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 22
78, 2279, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 
2913, 2914, 2915, 2916, 2917, 2918, 2919, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 3001, 3002, 3003
, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 30
77, 3078, 3079, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 
3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3257, 3258
, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3289, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 38
76, 3877, 3878, 3879, 3897, 3898, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3929, 3930, 3931, 3932, 3933, 3934, 3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3961, 3962, 3963, 3964, 3965, 3966, 
3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4057
, 4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4089, 4090, 4091, 4092, 4093, 4094, 4095]
[0, 1, 29, 30, 31, 32, 33, 925, 926, 927, 928, 929, 957, 958, 959, 960, 961, 989, 990, 991, 992, 993, 1021, 1022, 1023, 1024, 1025, 1053, 1054, 1055, 1056, 1057, 1949, 1950, 1951, 1952, 1953, 1981, 1982, 1983, 1984, 1985, 2013, 2014, 2015
, 2016, 2017, 2045, 2046, 2047, 2048, 2049, 2077, 2078, 2079, 2080, 2081, 2973, 2974, 2975, 2976, 2977, 3005, 3006, 3007, 3008, 3009, 3037, 3038, 3039, 3040, 3041, 3069, 3070, 3071, 3072, 3073, 3101, 3102, 3103, 3104, 3105, 3997, 3998, 39
99, 4000, 4001, 4029, 4030, 4031, 4032, 4033, 4061, 4062, 4063, 4064, 4065, 4093, 4094, 4095]
[0, 1, 2, 3, 4, 5, 6, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 121, 122, 123, 124, 125, 126,
 127, 128, 129, 130, 131, 132, 133, 134, 135, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 217, 218, 219, 220, 221, 222, 223, 224, 22
5, 226, 227, 228, 229, 230, 231, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 
868, 869, 870, 871, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966
, 967, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 
1060, 1061, 1062, 1063, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1145, 1146, 1147, 1148, 1149, 1150
, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 12
41, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 
1859, 1860, 1861, 1862, 1863, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1945, 1946, 1947, 1948, 1949
, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 20
23, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 
2114, 2115, 2116, 2117, 2118, 2119, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2201, 2202, 2203, 2204
, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 22
78, 2279, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 
2913, 2914, 2915, 2916, 2917, 2918, 2919, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 3001, 3002, 3003
, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 30
77, 3078, 3079, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 
3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3257, 3258
, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3289, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 38
76, 3877, 3878, 3879, 3897, 3898, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3929, 3930, 3931, 3932, 3933, 3934, 3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3961, 3962, 3963, 3964, 3965, 3966, 
3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4057
, 4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4089, 4090, 4091, 4092, 4093, 4094, 4095]
[0, 1, 2, 3, 4, 5, 6, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 121, 122, 123, 124, 125, 126,
 127, 128, 129, 130, 131, 132, 133, 134, 135, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 217, 218, 219, 220, 221, 222, 223, 224, 22
5, 226, 227, 228, 229, 230, 231, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 
868, 869, 870, 871, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966
, 967, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 
1060, 1061, 1062, 1063, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1145, 1146, 1147, 1148, 1149, 1150
, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 12
41, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 
1859, 1860, 1861, 1862, 1863, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1945, 1946, 1947, 1948, 1949
, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 20
23, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 
2114, 2115, 2116, 2117, 2118, 2119, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2201, 2202, 2203, 2204
, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 22
78, 2279, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 
2913, 2914, 2915, 2916, 2917, 2918, 2919, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 3001, 3002, 3003
, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 30
77, 3078, 3079, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 
3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3257, 3258
, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3289, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 38
76, 3877, 3878, 3879, 3897, 3898, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3929, 3930, 3931, 3932, 3933, 3934, 3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3961, 3962, 3963, 3964, 3965, 3966, 
3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4057
, 4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4089, 4090, 4091, 4092, 4093, 4094, 4095]
[0, 1, 2, 3, 4, 5, 6, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 121, 122, 123, 124, 125, 126,
 127, 128, 129, 130, 131, 132, 133, 134, 135, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 217, 218, 219, 220, 221, 222, 223, 224, 22
5, 226, 227, 228, 229, 230, 231, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 
868, 869, 870, 871, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966
, 967, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 
1060, 1061, 1062, 1063, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1145, 1146, 1147, 1148, 1149, 1150
, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 12
41, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 
1859, 1860, 1861, 1862, 1863, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1945, 1946, 1947, 1948, 1949
, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 20
23, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 
2114, 2115, 2116, 2117, 2118, 2119, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2201, 2202, 2203, 2204
, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 22
78, 2279, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 
2913, 2914, 2915, 2916, 2917, 2918, 2919, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 3001, 3002, 3003
, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 30
77, 3078, 3079, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 
3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3257, 3258
, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3289, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 38
76, 3877, 3878, 3879, 3897, 3898, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3929, 3930, 3931, 3932, 3933, 3934, 3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3961, 3962, 3963, 3964, 3965, 3966, 
3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4057
, 4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4089, 4090, 4091, 4092, 4093, 4094, 4095]
[0, 1, 2, 3, 4, 5, 6, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 121, 122, 123, 124, 125, 126,
 127, 128, 129, 130, 131, 132, 133, 134, 135, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 217, 218, 219, 220, 221, 222, 223, 224, 22
5, 226, 227, 228, 229, 230, 231, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 
868, 869, 870, 871, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966
, 967, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 
1060, 1061, 1062, 1063, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1145, 1146, 1147, 1148, 1149, 1150
, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 12
41, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 
1859, 1860, 1861, 1862, 1863, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1945, 1946, 1947, 1948, 1949
, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 20
23, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 
2114, 2115, 2116, 2117, 2118, 2119, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2201, 2202, 2203, 2204
, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 22
78, 2279, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 
2913, 2914, 2915, 2916, 2917, 2918, 2919, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 3001, 3002, 3003
, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 30
77, 3078, 3079, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 
3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3257, 3258
, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3289, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 38
76, 3877, 3878, 3879, 3897, 3898, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3929, 3930, 3931, 3932, 3933, 3934, 3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3961, 3962, 3963, 3964, 3965, 3966, 
3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4057
, 4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4089, 4090, 4091, 4092, 4093, 4094, 4095]
[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 
212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332, 336, 340, 344, 348, 352, 356, 360, 364, 368, 372, 376, 380, 384, 388, 392, 396, 400
, 404, 408, 412, 416, 420, 424, 428, 432, 436, 440, 444, 448, 452, 456, 460, 464, 468, 472, 476, 480, 484, 488, 492, 496, 500, 504, 508, 512, 516, 520, 524, 528, 532, 536, 540, 544, 548, 552, 556, 560, 564, 568, 572, 576, 580, 584, 588, 5
92, 596, 600, 604, 608, 612, 616, 620, 624, 628, 632, 636, 640, 644, 648, 652, 656, 660, 664, 668, 672, 676, 680, 684, 688, 692, 696, 700, 704, 708, 712, 716, 720, 724, 728, 732, 736, 740, 744, 748, 752, 756, 760, 764, 768, 772, 776, 780,
 784, 788, 792, 796, 800, 804, 808, 812, 816, 820, 824, 828, 832, 836, 840, 844, 848, 852, 856, 860, 864, 868, 872, 876, 880, 884, 888, 892, 896, 900, 904, 908, 912, 916, 920, 924, 928, 932, 936, 940, 944, 948, 952, 956, 960, 964, 968, 97
2, 976, 980, 984, 988, 992, 996, 1000, 1004, 1008, 1012, 1016, 1020]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
 62, 63, 32759, 32760, 32761, 32762, 32763, 32764, 32765, 32766, 32767]
 """

    # Paste all matrix diagonal-index lists into all_a_diags_raw, one list per line/block.
    a_diags_groups = _parse_list_of_lists(all_a_diags_raw)
    if not a_diags_groups:
        raise ValueError("No diagonal-index list parsed from all_a_diags_raw.")

    slots = 2**15

    # Mode A: run one simulation for each index list.
    print("\n===== MODE A: PER-LIST SIMULATION =====")
    per_list_new_union: Set[int] = set()
    per_list_final_union: Set[int] = set()
    per_list_needed_union: Set[int] = set()
    per_list_baby_union: Set[int] = set()
    per_list_giant_union: Set[int] = set()

    per_list_needed_union_conj: Set[int] = set()
    per_list_new_union_conj: Set[int] = set()
    per_list_final_union_conj: Set[int] = set()
    per_list_baby_union_conj: Set[int] = set()
    per_list_giant_union_conj: Set[int] = set()

    per_list_needed_union_struct: Set[int] = set()
    per_list_new_union_struct: Set[int] = set()
    per_list_final_union_struct: Set[int] = set()
    per_list_baby_union_struct: Set[int] = set()
    per_list_giant_union_struct: Set[int] = set()

    per_list_needed_union_struct_opt: Set[int] = set()
    per_list_new_union_struct_opt: Set[int] = set()
    per_list_final_union_struct_opt: Set[int] = set()
    per_list_baby_union_struct_opt: Set[int] = set()
    per_list_giant_union_struct_opt: Set[int] = set()

    for idx, diag_list in enumerate(a_diags_groups, start=1):
        b_step_each = _find_best_bsgs_ratio(diag_list, slots, log_max_ratio=0)
        stat_each = evaluate_strategy(
            diag_list,
            n,
            bsKey,
            b_step_each,
            slots,
            name=f"baseline_ratio(per_list)[group={idx}]",
        )
        final_each = stat_each["new_only"] | stat_each["bs_norm"]

        per_list_new_union |= stat_each["new_only"]
        per_list_final_union |= final_each
        per_list_needed_union |= stat_each["needed_all"]
        per_list_baby_union |= stat_each["needed_baby"]
        per_list_giant_union |= stat_each["needed_giant"]

        stat_each_conj = evaluate_with_conjugation(stat_each, bsKey, slots)
        per_list_needed_union_conj |= stat_each_conj["needed_all"]
        per_list_new_union_conj |= stat_each_conj["new_only"]
        per_list_final_union_conj |= stat_each_conj["final_keyindex"]
        per_list_baby_union_conj |= stat_each_conj["needed_baby"]
        per_list_giant_union_conj |= stat_each_conj["needed_giant"]

        stat_each_struct = evaluate_strategy_structural_conj(
            diag_list,
            n,
            bsKey,
            b_step_each,
            slots,
            name=f"baseline_ratio(per_list)[group={idx}]+structural_conj",
        )
        per_list_needed_union_struct |= stat_each_struct["needed_all"]
        per_list_new_union_struct |= stat_each_struct["new_only"]
        per_list_final_union_struct |= stat_each_struct["new_only"] | stat_each_struct["bs_norm"]
        per_list_baby_union_struct |= stat_each_struct["needed_baby"]
        per_list_giant_union_struct |= stat_each_struct["needed_giant"]

        stat_each_struct_opt = find_best_bstep_structural_conj(
            diag_list,
            n,
            bsKey,
            slots,
            mode="strict-min-new",
        )
        per_list_needed_union_struct_opt |= stat_each_struct_opt["needed_all"]
        per_list_new_union_struct_opt |= stat_each_struct_opt["new_only"]
        per_list_final_union_struct_opt |= stat_each_struct_opt["new_only"] | stat_each_struct_opt["bs_norm"]
        per_list_baby_union_struct_opt |= stat_each_struct_opt["needed_baby"]
        per_list_giant_union_struct_opt |= stat_each_struct_opt["needed_giant"]

        # print(f"\n[GROUP {idx}] size={len(diag_list)}")
        # print_stats("BASE", stat_each)

    print("\n[MODE A SUMMARY]")
    print("groups:", len(a_diags_groups))
    print("needed_all_union_count:", len(per_list_needed_union))
    print("needed_baby_union_count:", len(per_list_baby_union))
    print("needed_giant_union_count:", len(per_list_giant_union))
    print("needed_baby_union_sorted:", _sorted_int_set(per_list_baby_union))
    print("needed_giant_union_sorted:", _sorted_int_set(per_list_giant_union))
    print("needed_all_union_sorted:", _sorted_int_set(per_list_needed_union))
    print("new_only_union_count:", len(per_list_new_union))
    print("new_only_union_sorted:", _sorted_int_set(per_list_new_union))
    print("final_keyindex_union_count:", len(per_list_final_union))
    print("final_keyindex_union_sorted:", _sorted_int_set(per_list_final_union))

    print("\n[MODE A NEW: CONJUGATION-3STEP]")
    print("needed_all_union_count_conj:", len(per_list_needed_union_conj))
    print("needed_baby_union_count_conj:", len(per_list_baby_union_conj))
    print("needed_giant_union_count_conj:", len(per_list_giant_union_conj))
    print("new_only_union_count_conj:", len(per_list_new_union_conj))
    print("final_keyindex_union_count_conj:", len(per_list_final_union_conj))
    print("needed_baby_union_sorted_conj:", _sorted_int_set(per_list_baby_union_conj))
    print("needed_giant_union_sorted_conj:", _sorted_int_set(per_list_giant_union_conj))
    print("needed_all_union_sorted_conj:", _sorted_int_set(per_list_needed_union_conj))
    print("new_only_union_sorted_conj:", _sorted_int_set(per_list_new_union_conj))
    print("final_keyindex_union_sorted_conj:", _sorted_int_set(per_list_final_union_conj))

    print("key_count_reduction_vs_base_union:", len(per_list_needed_union) - len(per_list_needed_union_conj))
    if len(per_list_needed_union) > 0:
        print("key_count_reduction_ratio_vs_base_union:", f"{(len(per_list_needed_union) - len(per_list_needed_union_conj)) / len(per_list_needed_union):.4%}")

    print("\n[MODE A NEW: STRUCTURAL-CONJ-FOLD]")
    print("needed_all_union_count_struct:", len(per_list_needed_union_struct))
    print("needed_baby_union_count_struct:", len(per_list_baby_union_struct))
    print("needed_giant_union_count_struct:", len(per_list_giant_union_struct))
    print("new_only_union_count_struct:", len(per_list_new_union_struct))
    print("final_keyindex_union_count_struct:", len(per_list_final_union_struct))
    print("needed_baby_union_sorted_struct:", _sorted_int_set(per_list_baby_union_struct))
    print("needed_giant_union_sorted_struct:", _sorted_int_set(per_list_giant_union_struct))
    print("needed_all_union_sorted_struct:", _sorted_int_set(per_list_needed_union_struct))
    print("new_only_union_sorted_struct:", _sorted_int_set(per_list_new_union_struct))
    print("final_keyindex_union_sorted_struct:", _sorted_int_set(per_list_final_union_struct))

    print("key_count_reduction_vs_base_union_struct:", len(per_list_needed_union) - len(per_list_needed_union_struct))
    if len(per_list_needed_union) > 0:
        print("key_count_reduction_ratio_vs_base_union_struct:", f"{(len(per_list_needed_union) - len(per_list_needed_union_struct)) / len(per_list_needed_union):.4%}")

    print("\n[MODE A NEW: STRUCTURAL-CONJ-FOLD-OPT-BSTEP]")
    print("needed_all_union_count_struct_opt:", len(per_list_needed_union_struct_opt))
    print("needed_baby_union_count_struct_opt:", len(per_list_baby_union_struct_opt))
    print("needed_giant_union_count_struct_opt:", len(per_list_giant_union_struct_opt))
    print("new_only_union_count_struct_opt:", len(per_list_new_union_struct_opt))
    print("final_keyindex_union_count_struct_opt:", len(per_list_final_union_struct_opt))
    print("needed_baby_union_sorted_struct_opt:", _sorted_int_set(per_list_baby_union_struct_opt))
    print("needed_giant_union_sorted_struct_opt:", _sorted_int_set(per_list_giant_union_struct_opt))
    print("needed_all_union_sorted_struct_opt:", _sorted_int_set(per_list_needed_union_struct_opt))
    print("new_only_union_sorted_struct_opt:", _sorted_int_set(per_list_new_union_struct_opt))
    print("final_keyindex_union_sorted_struct_opt:", _sorted_int_set(per_list_final_union_struct_opt))

    print("key_count_reduction_vs_base_union_struct_opt:", len(per_list_needed_union) - len(per_list_needed_union_struct_opt))
    if len(per_list_needed_union) > 0:
        print("key_count_reduction_ratio_vs_base_union_struct_opt:", f"{(len(per_list_needed_union) - len(per_list_needed_union_struct_opt)) / len(per_list_needed_union):.4%}")

    # Mode B: merge all lists, deduplicate, then run one simulation.
    raw_total = sum(len(g) for g in a_diags_groups)
    merged_a_diags = sorted({x & (slots - 1) for g in a_diags_groups for x in g})
    if not merged_a_diags:
        raise ValueError("Merged diagonal list is empty after deduplication.")

    print("\n===== MODE B: MERGED-DEDUP SINGLE SIMULATION =====")
    print("input_groups:", len(a_diags_groups))
    print("raw_total_count:", raw_total)
    print("merged_unique_count:", len(merged_a_diags))

    b_step_merged = _find_best_bsgs_ratio(merged_a_diags, slots, log_max_ratio=0)
    stat_merged = evaluate_strategy(
        merged_a_diags,
        n,
        bsKey,
        b_step_merged,
        slots,
        name="baseline_ratio(merged_dedup_A)",
    )
    final_keyindex_merged = stat_merged["new_only"] | stat_merged["bs_norm"]

    print_stats("BASE", stat_merged)
    # print("merged_a_diags_sorted:", merged_a_diags) # 合并diagonal列表的索引（已去重）
    print("needed_baby_sorted:", _sorted_int_set(stat_merged["needed_baby"]))
    print("needed_giant_sorted:", _sorted_int_set(stat_merged["needed_giant"]))
    print("needed_all_sorted:", _sorted_int_set(stat_merged["needed_all"]))
    print("new_only_sorted:", _sorted_int_set(stat_merged["new_only"]))
    print("final_keyindex_sorted:", _sorted_int_set(final_keyindex_merged))
    print("key reused:", set(stat_merged["needed_all"]) & set(bsKey))

    stat_merged_conj = evaluate_with_conjugation(stat_merged, bsKey, slots)

    stat_merged_struct = evaluate_strategy_structural_conj(
        merged_a_diags,
        n,
        bsKey,
        b_step_merged,
        slots,
        name="baseline_ratio(merged_dedup_A)+structural_conj",
    )

    stat_merged_struct_opt = find_best_bstep_structural_conj(
        merged_a_diags,
        n,
        bsKey,
        slots,
        mode="strict-min-new",
    )
    print("\n[MODE B NEW: CONJUGATION-3STEP]")
    print(f"[CONJ] strategy={stat_merged_conj['strategy']}")
    print(f"bStep={stat_merged_conj['b_step']}, gStep={stat_merged_conj['g_step']}")
    print(f"needed_all={len(stat_merged_conj['needed_all'])}, new_only={len(stat_merged_conj['new_only'])}, reused={len(stat_merged_conj['reused'])}")
    print(f"needed_baby={len(stat_merged_conj['needed_baby'])}, needed_giant={len(stat_merged_conj['needed_giant'])}")
    print("needed_baby_sorted_conj:", _sorted_int_set(stat_merged_conj["needed_baby"]))
    print("needed_giant_sorted_conj:", _sorted_int_set(stat_merged_conj["needed_giant"]))
    print("needed_all_sorted_conj:", _sorted_int_set(stat_merged_conj["needed_all"]))
    print("new_only_sorted_conj:", _sorted_int_set(stat_merged_conj["new_only"]))
    print("final_keyindex_sorted_conj:", _sorted_int_set(stat_merged_conj["final_keyindex"]))
    print("key_count_reduction_vs_base:", len(stat_merged["needed_all"]) - len(stat_merged_conj["needed_all"]))
    if len(stat_merged["needed_all"]) > 0:
        print("key_count_reduction_ratio_vs_base:", f"{(len(stat_merged["needed_all"]) - len(stat_merged_conj["needed_all"])) / len(stat_merged["needed_all"]):.4%}")
    print("final_keyindex_count_conj:", len(stat_merged_conj["final_keyindex"]))

    print("\n[MODE B NEW: STRUCTURAL-CONJ-FOLD]")
    print(f"[STRUCT] strategy={stat_merged_struct['strategy']}")
    print(f"bStep={stat_merged_struct['b_step']}, gStep={stat_merged_struct['g_step']}")
    print(f"needed_all={len(stat_merged_struct['needed_all'])}, new_only={len(stat_merged_struct['new_only'])}, reused={len(stat_merged_struct['reused'])}")
    print(f"needed_baby={len(stat_merged_struct['needed_baby'])}, needed_giant={len(stat_merged_struct['needed_giant'])}")
    print("needed_baby_sorted_struct:", _sorted_int_set(stat_merged_struct["needed_baby"]))
    print("needed_giant_sorted_struct:", _sorted_int_set(stat_merged_struct["needed_giant"]))
    print("needed_all_sorted_struct:", _sorted_int_set(stat_merged_struct["needed_all"]))
    print("new_only_sorted_struct:", _sorted_int_set(stat_merged_struct["new_only"]))
    final_keyindex_struct = stat_merged_struct["new_only"] | stat_merged_struct["bs_norm"]
    print("final_keyindex_sorted_struct:", _sorted_int_set(final_keyindex_struct))
    print("key_count_reduction_vs_base_struct:", len(stat_merged["needed_all"]) - len(stat_merged_struct["needed_all"]))
    if len(stat_merged["needed_all"]) > 0:
        print("key_count_reduction_ratio_vs_base_struct:", f"{(len(stat_merged["needed_all"]) - len(stat_merged_struct["needed_all"])) / len(stat_merged["needed_all"]):.4%}")
    print("final_keyindex_count_struct:", len(final_keyindex_struct))

    print("\n[MODE B NEW: STRUCTURAL-CONJ-FOLD-OPT-BSTEP]")
    print(f"[STRUCT-OPT] strategy={stat_merged_struct_opt['strategy']}")
    print(f"bStep={stat_merged_struct_opt['b_step']}, gStep={stat_merged_struct_opt['g_step']}")
    print(f"needed_all={len(stat_merged_struct_opt['needed_all'])}, new_only={len(stat_merged_struct_opt['new_only'])}, reused={len(stat_merged_struct_opt['reused'])}")
    print(f"needed_baby={len(stat_merged_struct_opt['needed_baby'])}, needed_giant={len(stat_merged_struct_opt['needed_giant'])}")
    print("needed_baby_sorted_struct_opt:", _sorted_int_set(stat_merged_struct_opt["needed_baby"]))
    print("needed_giant_sorted_struct_opt:", _sorted_int_set(stat_merged_struct_opt["needed_giant"]))
    print("needed_all_sorted_struct_opt:", _sorted_int_set(stat_merged_struct_opt["needed_all"]))
    print("new_only_sorted_struct_opt:", _sorted_int_set(stat_merged_struct_opt["new_only"]))
    final_keyindex_struct_opt = stat_merged_struct_opt["new_only"] | stat_merged_struct_opt["bs_norm"]
    print("final_keyindex_sorted_struct_opt:", _sorted_int_set(final_keyindex_struct_opt))
    print("key_count_reduction_vs_base_struct_opt:", len(stat_merged["needed_all"]) - len(stat_merged_struct_opt["needed_all"]))
    if len(stat_merged["needed_all"]) > 0:
        print("key_count_reduction_ratio_vs_base_struct_opt:", f"{(len(stat_merged["needed_all"]) - len(stat_merged_struct_opt["needed_all"])) / len(stat_merged["needed_all"]):.4%}")
    print("final_keyindex_count_struct_opt:", len(final_keyindex_struct_opt))

    multi_step_results: List[Tuple[str, Dict[str, object]]] = []
    print("\n===== MODE B: MULTI-STEP SEARCH =====")
    for levels in (2, 3, 4):
        best_multi = find_best_multistep_strategy(
            merged_a_diags,
            n,
            bsKey,
            slots=slots,
            levels=levels,
            mode="weighted",
            alpha=1.0,
            beta=3.0,
            use_conjugation=False,
        )
        multi_step_results.append((f"multi-{levels}", best_multi))
        print_stats(f"MULTI-{levels}STEP", best_multi)
        for stage_idx, stage_set in enumerate(best_multi["needed_stages"], start=1):
            print(f"needed_stage_{stage_idx}_count:", len(stage_set))
            print(f"needed_stage_{stage_idx}_sorted:", _sorted_int_set(stage_set))
        print("final_keyindex_count:", len(best_multi["final_keyindex"]))
        print("final_keyindex_sorted:", _sorted_int_set(best_multi["final_keyindex"]))

        best_multi_conj = find_best_multistep_strategy(
            merged_a_diags,
            n,
            bsKey,
            slots=slots,
            levels=levels,
            mode="weighted",
            alpha=1.0,
            beta=3.0,
            use_conjugation=True,
        )
        multi_step_results.append((f"multi-{levels}-conj", best_multi_conj))
        print_stats(f"MULTI-{levels}STEP-CONJ", best_multi_conj)
        for stage_idx, stage_set in enumerate(best_multi_conj["needed_stages"], start=1):
            print(f"needed_stage_{stage_idx}_count_conj:", len(stage_set))
            print(f"needed_stage_{stage_idx}_sorted_conj:", _sorted_int_set(stage_set))
        print("final_keyindex_count_conj:", len(best_multi_conj["final_keyindex"]))
        print("final_keyindex_sorted_conj:", _sorted_int_set(best_multi_conj["final_keyindex"]))

    base_summary = dict(stat_merged)
    base_summary["final_keyindex"] = final_keyindex_merged
    base_summary["online_rotations_avg"] = 2.0
    base_summary["online_rotations_total"] = 2 * len(merged_a_diags)
    conj_summary = dict(stat_merged_conj)
    conj_summary["online_rotations_avg"] = 3.0
    conj_summary["online_rotations_total"] = 3 * len(merged_a_diags)
    struct_summary = dict(stat_merged_struct)
    struct_summary["final_keyindex"] = final_keyindex_struct
    struct_summary["online_rotations_avg"] = 3.0
    struct_summary["online_rotations_total"] = 3 * len(merged_a_diags)
    struct_opt_summary = dict(stat_merged_struct_opt)
    struct_opt_summary["final_keyindex"] = final_keyindex_struct_opt
    struct_opt_summary["online_rotations_avg"] = 3.0
    struct_opt_summary["online_rotations_total"] = 3 * len(merged_a_diags)

    comparison_rows: List[Tuple[str, Dict[str, object]]] = [
        ("base-2step", base_summary),
        ("conj-post", conj_summary),
        ("struct-conj", struct_summary),
        ("struct-conj-opt", struct_opt_summary),
    ] + multi_step_results
    print_strategy_table("MODE B COMPARISON TABLE", comparison_rows)

    csv_rows: List[Dict[str, object]] = [
        _strategy_csv_row("mode_b_comparison", label, stat)
        for label, stat in comparison_rows
    ]

    beta_sweep_best_rows: List[Tuple[str, Dict[str, object]]] = []
    for beta in BETA_SWEEP_VALUES:
        sweep_candidates: List[Tuple[str, Dict[str, object]]] = []
        for levels in (2, 3, 4):
            for use_conjugation in (False, True):
                sweep_stat = find_best_multistep_strategy(
                    merged_a_diags,
                    n,
                    bsKey,
                    slots=slots,
                    levels=levels,
                    mode="weighted",
                    alpha=1.0,
                    beta=beta,
                    use_conjugation=use_conjugation,
                )
                sweep_label = f"multi-{levels}{'-conj' if use_conjugation else ''}"
                sweep_candidates.append((sweep_label, sweep_stat))
                csv_rows.append(_strategy_csv_row("beta_sweep_candidate", sweep_label, sweep_stat))

        best_label, best_stat = _best_scored_row(sweep_candidates)
        beta_sweep_best_rows.append((f"beta={beta:g}:{best_label}", best_stat))
        csv_rows.append(_strategy_csv_row("beta_sweep_best", best_label, best_stat))

    print_strategy_table("BETA SWEEP BEST BY WEIGHT", beta_sweep_best_rows)
    write_results_csv(RESULTS_CSV_PATH, csv_rows)
    print(f"\n[CSV] wrote {len(csv_rows)} rows to {RESULTS_CSV_PATH}")
