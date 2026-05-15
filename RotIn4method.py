def canonicalize_signed_step(s: int, slots_live: int) -> int:
    s %= slots_live
    if s > slots_live // 2:
        s -= slots_live
    return s

def map_logical_to_physical(logical_steps, stride=1, offset=0, slots_phys=None):
    out = []
    for s in logical_steps:
        if s == "CONJ":
            out.append("CONJ")
        else:
            v = offset + s * stride
            out.append(v if slots_phys is None else ((v % slots_phys + slots_phys) % slots_phys))
    return sorted(set(out), key=lambda x: (x == "CONJ", x))

# SEAL -----------------------------------------------------------------
# Source semantics: get_elt_from_step(step), get_elts_all(), generator_=3
def seal_extract(slots_live: int, requested_steps: list[int], include_conj=False):
    # CKKS complex case: coeff_count = 2 * slots_live, m = 2 * coeff_count = 4 * slots_live
    coeff_count = 2 * slots_live
    m = 2 * coeff_count
    generator = 3

    logical = sorted(set(canonicalize_signed_step(s, slots_live) for s in requested_steps))
    key_tokens = []

    if include_conj:
        key_tokens.append(RotKeyToken("conjugation", "CONJ"))
        key_tokens.append(RotKeyToken("galois_elt", m - 1))

    for s in logical:
        # step==0 must NOT be used for no-op; reserve special CONJ token instead
        if s < 0:
            step_eff = slots_live - abs(s)
        else:
            step_eff = s
        elt = pow(generator, step_eff, m)
        key_tokens.append(RotKeyToken("rotation_step", s))
        key_tokens.append(RotKeyToken("galois_elt", elt))

    return RotIndexResult(
        logical_steps=logical + (["CONJ"] if include_conj else []),
        physical_steps=logical + (["CONJ"] if include_conj else []),
        key_tokens=key_tokens,
        groups={"power2_default": [1 << i for i in range(slots_live.bit_length()-1)]},
        meta={"backend": "SEAL"}
    )

# Lattigo --------------------------------------------------------------
# Source semantics: BSGSIndex(nonZeroDiags, slots, N1) -> index, rotN1, rotN2
#                  GaloisElementsForLinearTransformation(...)
def lattigo_extract(nonZeroDiags: list[int], slots: int, N1: int, params):
    D = sorted(set(d % slots for d in nonZeroDiags))
    index = {}
    rotN1, rotN2 = set(), set()

    for d in D:
        baby = d % N1
        giant = d - baby
        index.setdefault(giant, []).append(baby)
        if baby != 0:
            rotN1.add(baby)
        if giant != 0:
            rotN2.add(giant)

    rotN1 = sorted(rotN1)
    rotN2 = sorted(rotN2)
    steps = sorted(set(rotN1 + rotN2))
    galels = [params.GaloisElement(s) for s in steps]

    return RotIndexResult(
        logical_steps=sorted(set(nonZeroDiags)),
        physical_steps=steps,
        key_tokens=[RotKeyToken("galois_elt", g) for g in galels],
        groups={"index": index, "baby": rotN1, "giant": rotN2},
        meta={"backend": "Lattigo", "bsgs_n1": N1}
    )

# OpenFHE --------------------------------------------------------------
# Publicly stable entry points are EvalBootstrapSetup / EvalBootstrapKeyGen.
# Internal helper names may vary by version; the robust extraction path is:
#   setup -> layer plans -> per-layer diagonals -> ReduceRotation -> EvalAtIndexKeyGen
def openfhe_bootstrap_extract(levelBudget, numSlots, dim1=None, layersCollapse=None, packing_layout="boot-dft"):
    layer_diags = plan_boot_layers(levelBudget, numSlots, dim1, packing_layout)  # framework/version dependent
    if layersCollapse:
        layer_diags = collapse_boot_layers(layer_diags, layersCollapse)
    S = set()
    for layer_id, diags in enumerate(layer_diags):
        mapped = [canonicalize_signed_step(d, numSlots) for d in diags]
        S.update(mapped)
    S = reduce_rotations_if_available(S)  # mirrors ReduceRotation-style post-process
    auto_indices = [find_automorphism_index(s) for s in sorted(S)]

    return RotIndexResult(
        logical_steps=sorted(S),
        physical_steps=sorted(S),
        key_tokens=[RotKeyToken("automorphism_index", a) for a in auto_indices],
        groups={"per_layer": layer_diags},
        meta={"backend": "OpenFHE", "note": "internal helper names vary by version"}
    )

# EVA ------------------------------------------------------------------
# Public source does not expose one stable helper like Lattigo.BSGSIndex.
# Robust extraction = compiler IR pass over rotate nodes -> dedup -> SEAL mapping.
def eva_extract(compiled_rotate_nodes, slots_live):
    logical = set()
    include_conj = False
    for node in compiled_rotate_nodes:
        if node.op == "rotate_left":
            logical.add(canonicalize_signed_step(+node.amount, slots_live))
        elif node.op == "rotate_right":
            logical.add(canonicalize_signed_step(-node.amount, slots_live))
        elif node.op == "conjugate":
            include_conj = True
    return seal_extract(slots_live, sorted(logical), include_conj)

# TenSEAL --------------------------------------------------------------
# Publicly visible:
#  - matmul_plain_inplace -> diagonal_ct_vector_matmul(matrix)
#  - enc_matmul_plain_inplace pads plain vector to next power of two, replicates,
#    then rotates by rows_nb * chunks_nb while chunks shrink by powers of two
#  - replicate_first_slot_inplace uses negative powers of two
def tenseal_matmul_plain_extract(matrix_shape, slots_live):
    rows, cols = matrix_shape
    logical_diags = [d for d in range(-(rows-1), cols) if diagonal_is_nonzero(d)]  # matrix dependent
    physical = [canonicalize_signed_step(d, slots_live) for d in logical_diags]
    return RotIndexResult(
        logical_steps=sorted(set(logical_diags)),
        physical_steps=sorted(set(physical)),
        key_tokens=[RotKeyToken("rotation_step", s) for s in sorted(set(physical))],
        groups={"diagonals": sorted(set(logical_diags))},
        meta={"backend": "TenSEAL", "note": "diagonal_ct_vector_matmul body not fully exposed in web index"}
    )

def tenseal_enc_matmul_plain_extract(rows_nb, plain_vec_size, slot_count):
    chunks_nb = 1 << ceil_log2(plain_vec_size)
    rots = []
    while chunks_nb > 1:
        rots.append(rows_nb * chunks_nb)
        chunks_nb >>= 1
    repl = [-(1 << i) for i in range(ceil_log2(slot_count))]
    return RotIndexResult(
        logical_steps=rots + repl,
        physical_steps=rots + repl,
        key_tokens=[RotKeyToken("rotation_step", s) for s in sorted(set(rots + repl))],
        groups={"reduction": rots, "replicate_first_slot": repl},
        meta={"backend": "TenSEAL"}
    )

# matrix-mult-fhe ------------------------------------------------------
# Exact source logic visible in rotation.h
def matrix_mult_fhe_binary_extract(batch_size):
    max_bit = floor_log2(batch_size)
    atoms = [1 << i for i in range(max_bit) if (1 << i) < batch_size]
    return RotIndexResult(
        logical_steps=atoms,
        physical_steps=atoms,
        key_tokens=[RotKeyToken("rotation_step", s) for s in atoms],
        groups={"atoms": atoms},
        meta={"backend": "matrix-mult-fhe"}
    )

def matrix_mult_fhe_decompose(rotation, batch_size):
    steps = []
    for i in reversed(range(floor_log2(batch_size))):
        step = 1 << i
        if step < batch_size and (abs(rotation) & step):
            steps.append(step if rotation > 0 else -step)
    return steps

# NEXUS ----------------------------------------------------------------
# Exact source logic visible:
#   rots.push_back((degree + 2^i) / 2^i)
# and expand_ciphertext applies ckks->rots[i] directly with apply_galois.
def nexus_extract(degree):
    rots = [ (degree + (1 << i)) // (1 << i) for i in range(ceil_log2(degree)) ]
    return RotIndexResult(
        logical_steps=[f"expand_level_{i}" for i in range(len(rots))],
        physical_steps=[],
        key_tokens=[RotKeyToken("galois_elt", r) for r in rots],
        groups={"expand_levels": list(range(len(rots)))},
        meta={"backend": "NEXUS", "note": "source stores direct galois tokens, not ordinary slot steps"}
    )

# Orion ----------------------------------------------------------------
# Exact source logic visible:
#  - AddPo2RotationKeys(): add 1,2,4,...,maxSlots
#  - AddRotationKey(rotation): galEl = Params.GaloisElement(rotation); generate if absent
def orion_extract(maxSlots, runtime_trace=None, params=None):
    base = []
    i = 1
    while i < maxSlots:
        base.append(i)
        i <<= 1
    requested = sorted(set(runtime_trace or []))
    all_steps = sorted(set(base + requested))
    galels = [params.GaloisElement(s) for s in all_steps] if params else []
    return RotIndexResult(
        logical_steps=all_steps,
        physical_steps=all_steps,
        key_tokens=[RotKeyToken("galois_elt", g) for g in galels] if galels else [RotKeyToken("rotation_step", s) for s in all_steps],
        groups={"preloaded_power2": base, "runtime_extra": requested},
        meta={"backend": "Orion", "policy": "preload + on-demand"}
    )
