import re
from pathlib import Path
from math import isclose
from collections import OrderedDict
import time
import numpy as np
import tensorflow as tf
from pymatgen.symmetry.groups import SpaceGroup
from typing import Callable, List, Tuple
import xrayutilities as xu
import pandas as pd

# --- config --------------------------------------------------------------
# Atomic numbers
Z = {"Pr": 59, "O": 8, "Ni": 28, "Cu": 29, "Ba": 56, "La":57}


# Skip rows whose displacements are all zero
DROP_ZERO_DISP = False

# Name blocks as M1, M2, ...
BLOCK_VAR_PREFIX = "M"

# Emit comments mapping each block header -> its variable
EMIT_BLOCK_COMMENTS = True
# ------------------------------------------------------------------------
def get_df(path):
    df = pd.read_csv(
    path,
    sep=r"\s+",
    engine="python",
    header=None,
    names=["h", "k", "l", "intensity", "std"],
    comment="#",
    )
    int_cols = ["h", "k", "l"]
    float_cols = ["intensity", "std"]

    for c in int_cols + float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2) Drop rows where ints or intensity are missing after coercion
    bad = df[df[int_cols + ["intensity"]].isna().any(axis=1)]
    print(f"Dropping {len(bad)} malformed rows")
    df = df.dropna(subset=int_cols + ["intensity"]).copy()

    # 3) Make h,k,l real integers; keep intensity as float
    df[int_cols] = df[int_cols].astype(int)
    df["intensity"] = df["intensity"].astype(float)
    df["std"] = df["std"].astype(float)
    print(f"Loaded {len(df)} reflections from: {path}")
    print(df.describe())
    return df

def nice_float(x):
    s = f"{round(float(x), 8):.8f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s

def label_to_element(label):
    m = re.match(r"([A-Za-z]+)", label)
    if not m:
        raise ValueError(f"Cannot parse element from label: {label}")
    return m.group(1)


def coeff_term(coeff, block_var):
    """
    Turn a numeric coeff into ' ± var', ' ± 0.5*var', or ' ± mag*var'.
    Return '' if coeff == 0.
    """
    if isclose(coeff, 0.0, abs_tol=1e-12):
        return ""
    sign = "+" if coeff > 0 else "-"
    mag = abs(coeff)
    if isclose(mag, 1.0, abs_tol=1e-9):
        return f" {sign} {block_var}"
    elif isclose(mag, 0.5, abs_tol=1e-9):
        return f" {sign} 0.5*{block_var}"
    else:
        return f" {sign} {nice_float(mag)}*{block_var}"

def parse(lines):
    """
    Yield (block_index, block_header, label, [x,y,z,dx,dy,dz]).
    block_index increments at each new mode header line.
    """
    current_label = None
    current_block_header = None
    block_idx = 0

    header_re = re.compile(r".*normfactor\s*=\s*[-\d.]+")
    start_re = re.compile(
        r"^\s*([A-Za-z]+\d+_\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$"
    )
    cont_re = re.compile(
        r"^\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$"
    )

    for raw in lines:
        line = raw.rstrip("\n")

        if not line.strip():
            current_label = None
            continue

        if line.startswith("Displacive mode"):
            current_label = None
            current_block_header = None
            continue

        if header_re.match(line):
            current_block_header = line.strip()
            block_idx += 1
            current_label = None
            continue

        m = start_re.match(line)
        if m:
            current_label = m.group(1)
            nums = list(map(float, m.groups()[1:]))
            yield block_idx, current_block_header, current_label, nums
            continue

        if current_label:
            m2 = cont_re.match(line)
            if m2:
                nums = list(map(float, m2.groups()))
                yield block_idx, current_block_header, current_label, nums
                continue
        # ignore anything else


def parse_linear_expr(expr: str):
    """
    Parse linear expression in a,b,c like '2c', '-2a-2b', 'a-b', '0'
    and return coefficients (ca, cb, cc) such that
    expr = ca*a + cb*b + cc*c.
    """
    expr = expr.strip()
    if expr == "" or expr == "0":
        return 0.0, 0.0, 0.0

    expr_nospace = expr.replace(" ", "")
    tokens = re.findall(r'[+-]?[^+-]+', expr_nospace)

    coef = {'a': 0.0, 'b': 0.0, 'c': 0.0}
    for tok in tokens:
        # find variable (a, b or c)
        m_var = re.search(r'([abc])$', tok)
        if not m_var:
            # constant term – should normally be 0 for hkl transforms
            val = float(tok)
            if abs(val) > 1e-12:
                raise ValueError(f"Unexpected constant term {tok} in expression {expr}")
            continue

        var = m_var.group(1)
        coef_str = tok[:m_var.start()]  # e.g. '', '2', '-2', '3/2', '-'

        # handle sign
        sign = 1.0
        if coef_str.startswith('+'):
            coef_str = coef_str[1:]
        elif coef_str.startswith('-'):
            sign = -1.0
            coef_str = coef_str[1:]

        # empty => coefficient 1
        if coef_str == "":
            val = 1.0
        else:
            if '/' in coef_str:
                num, den = coef_str.split('/')
                val = float(num) / float(den)
            else:
                val = float(coef_str)

        val *= sign
        coef[var] += val

    return coef['a'], coef['b'], coef['c']


def parse_cif_cell_params_and_transform(cif_path):
    """
    Open a CIF file, read all six lattice parameters (a, b, c, alpha, beta, gamma),
    the space group number, and the _iso_parent-to-child.transform_Pp_abc string.

    Returns
    -------
    tuple
        (cell_length_a, cell_length_b, cell_length_c,
         cell_angle_alpha, cell_angle_beta, cell_angle_gamma,
         space_group_number,
         transform_list_hkl)
    """
    cell_a = cell_b = cell_c = None
    cell_alpha = cell_beta = cell_gamma = None
    space_group_number = None
    transform_str = None

    with open(cif_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()

            # --- Cell Lengths ---
            if stripped.startswith("_cell_length_a"):
                parts = stripped.split()
                if len(parts) >= 2:
                    cell_a = float(parts[1].split('(')[0])

            elif stripped.startswith("_cell_length_b"):
                parts = stripped.split()
                if len(parts) >= 2:
                    cell_b = float(parts[1].split('(')[0])

            elif stripped.startswith("_cell_length_c"):
                parts = stripped.split()
                if len(parts) >= 2:
                    cell_c = float(parts[1].split('(')[0])

            # --- Cell Angles ---
            elif stripped.startswith("_cell_angle_alpha"):
                parts = stripped.split()
                if len(parts) >= 2:
                    cell_alpha = float(parts[1].split('(')[0])

            elif stripped.startswith("_cell_angle_beta"):
                parts = stripped.split()
                if len(parts) >= 2:
                    cell_beta = float(parts[1].split('(')[0])

            elif stripped.startswith("_cell_angle_gamma"):
                parts = stripped.split()
                if len(parts) >= 2:
                    cell_gamma = float(parts[1].split('(')[0])

            # --- Space Group Number (NEW) ---
            # Checks for standard tag and common alternative
            elif stripped.startswith("_symmetry_Int_Tables_number") or stripped.startswith("_space_group_IT_number"):
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        # Split on '(' to handle uncertainties like 65(2)
                        space_group_number = int(parts[1].split('(')[0])
                    except ValueError:
                        pass

            # --- Transform String ---
            elif stripped.startswith("_iso_parent-to-child.transform_Pp_abc"):
                # everything after the tag is the transform string
                parts = stripped.split(maxsplit=1)
                if len(parts) == 2:
                    transform_str = parts[1].strip().strip("'\"")

    # --- Validation and Error Handling ---
    if cell_a is None or cell_b is None or cell_c is None:
        raise ValueError("Could not find all _cell_length_{a,b,c} entries in CIF.")
    if cell_alpha is None or cell_beta is None or cell_gamma is None:
        raise ValueError("Could not find all _cell_angle_{alpha,beta,gamma} entries in CIF.")
    if space_group_number is None:
        # Warning instead of Error, or set a default if strictly required
        print("Warning: Could not find _symmetry_Int_Tables_number in CIF. Defaulting to None.")
    if transform_str is None:
        raise ValueError("Could not find _iso_parent-to-child.transform_Pp_abc in CIF.")

    # --- Transform Parsing ---

    basis_part = transform_str.split(";", 1)[0]
    comps = [c.strip() for c in basis_part.split(",")]
    if len(comps) != 3:
        raise ValueError(f"Unexpected transform format: {transform_str}")

    # Note: Requires your parse_linear_expr function to be defined in scope
    ca_h, cb_h, cc_h = parse_linear_expr(comps[0])
    ca_k, cb_k, cc_k = parse_linear_expr(comps[1])
    ca_l, cb_l, cc_l = parse_linear_expr(comps[2])

    def funky(h, k, l):
        """
        Function to transform an (h, k, l) from the old cell to the new cell.
        """
        h_new = ca_h * h + cb_h * k + cc_h * l
        k_new = ca_k * h + cb_k * k + cc_k * l
        l_new = ca_l * h + cb_l * k + cc_l * l
        return h_new, k_new, l_new

    # --- Final Return (Updated) ---
    return cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma, space_group_number, funky

def build_res_per_block_var(path_txt, cif_path):
    """
    1. Pre-populates all atoms in the fully expanded unit cell from a CIF.
    2. Deduplicates atoms by (elem, Z, x0, y0, z0).
    3. Accumulates mode displacement contributions from the txt file.
    """
    atoms = OrderedDict()
    block_meta = OrderedDict()

    # ---------------------------------------------------------
    # 1. BASELINE PARSE: Generate full unit cell from CIF
    # ---------------------------------------------------------
    crystal = xu.materials.Crystal.fromCIF(cif_path)

    for atom_obj, pos, _, _ in crystal.lattice.base():
        # 1. Cast the xrayutilities object to a string
        atom_str = str(atom_obj)

        # 2. Extract just the element (e.g., 'La' from 'La3+ (57)')
        elem = label_to_element(atom_str)
        Znum = Z.get(elem)

        if Znum is None:
            continue

        x, y, z = pos[0], pos[1], pos[2]
        x0, y0, z0 = nice_float(x), nice_float(y), nice_float(z)

        key = (elem, Znum, x0, y0, z0)

        # Pre-fill the dictionary with 0 displacement
        if key not in atoms:
            atoms[key] = {
                "elem": elem,
                "Z": Znum,
                "x0": x0,
                "y0": y0,
                "z0": z0,
                "terms": [[], [], []],
                # Fallback to the element symbol since xu drops the '1_1' site suffix
                "occ_var": f"occ_{elem}",
                "b_var": f"B_{elem}"
            }

    # ---------------------------------------------------------
    # 2. MODE PARSE: Add displacements from the text file
    # ---------------------------------------------------------
    lines = Path(path_txt).read_text(encoding="utf-8", errors="ignore").splitlines()

    for block_idx, block_header, label, (x, y, z, dx, dy, dz) in parse(lines):
        if block_idx == 0:
            continue

        if block_idx not in block_meta:
            block_var = f"{BLOCK_VAR_PREFIX}{block_idx}"
            block_meta[block_idx] = (block_var, block_header or f"Block {block_idx}")

        if DROP_ZERO_DISP and all(isclose(v, 0.0, abs_tol=1e-12) for v in (dx, dy, dz)):
            continue

        elem = label_to_element(label)
        Znum = Z.get(elem)
        if Znum is None:
            continue  # Safe skip if something weird is parsed

        x0, y0, z0 = nice_float(x), nice_float(y), nice_float(z)
        key = (elem, Znum, x0, y0, z0)
        if key in atoms:
            atoms[key]["occ_var"] = f"occ_{label}"
            atoms[key]["b_var"] = f"B_{label}"

        # If somehow a mode atom wasn't in the CIF, add it
        if key not in atoms:
            atoms[key] = {
                "elem": elem,
                "Z": Znum,
                "x0": x0,
                "y0": y0,
                "z0": z0,
                "terms": [[], [], []],
                "occ_var": f"occ_{label}",
                "b_var": f"B_{label}"
            }

        block_var = block_meta[block_idx][0]

        tx = coeff_term(dx, block_var)
        ty = coeff_term(dy, block_var)
        tz = coeff_term(dz, block_var)

        # Append displacements
        if tx:
            atoms[key]["terms"][0].append(tx)
        if ty:
            atoms[key]["terms"][1].append(ty)
        if tz:
            atoms[key]["terms"][2].append(tz)

    return atoms, block_meta


def emit_python(mode_file, cif_path):
    # Now passing both the mode file and the CIF file
    atoms, block_meta = build_res_per_block_var(mode_file, cif_path)

    modes = [var for var, _ in block_meta.values()]

    # Gather unique occupancy and B-factor variables sequentially
    occ_factors = []
    b_factors = []
    for payload in atoms.values():
        if payload["occ_var"] not in occ_factors:
            occ_factors.append(payload["occ_var"])
        if payload["b_var"] not in b_factors:
            b_factors.append(payload["b_var"])

    # Combine lists for the function arguments
    all_vars = modes + occ_factors + b_factors
    vars_list = ", ".join(all_vars)

    out = []
    out.append(f"def shift_atoms({vars_list}):")
    out.append('    """')
    out.append("    Function to shift atoms in the structure.")
    out.append("    :return: Structure of shifted atoms")
    out.append('    """')

    if EMIT_BLOCK_COMMENTS and block_meta:
        out.append("    # Mode blocks (one variable per block):")
        for var, header in block_meta.values():
            out.append(f"    #   {var}: {header}")

    out.append("    res = [")

    current_elem = None
    for (elem, Znum, x0, y0, z0), payload in atoms.items():
        # Aesthetic empty line between different elements
        if current_elem is not None and elem != current_elem:
            out.append("")
        current_elem = elem

        tx_list, ty_list, tz_list = payload["terms"]

        # Construct algebraic string for displaced coordinates
        x_expr = x0 + "".join(tx_list) if tx_list else x0
        y_expr = y0 + "".join(ty_list) if ty_list else y0
        z_expr = z0 + "".join(tz_list) if tz_list else z0

        occ_var = payload["occ_var"]
        b_var = payload["b_var"]

        # Output with occ and B variables included
        out.append(f"        ['{elem}', {Znum}, [{x_expr}, {y_expr}, {z_expr}], {occ_var}, {b_var}],")

    out.append("    ]")
    out.append("    return res")

    # Join into source code
    source = "\n".join(out)

    # Execute the source in a fresh namespace
    ns = {}
    exec(source, ns)

    # Return the function, the source, and the specific lists of variables
    return ns["shift_atoms"], source, modes, b_factors, occ_factors


def hkl_LTT_to_HTT(hL, kL, lL):
    """
    Transforms experimental LTT coordinates to HTT parent coordinates,
    aligned with the ISODISTORT basis vector convention (a-b, a+b, c).
    """
    h = (hL + kL) / 2.0
    k = (kL - hL) / 2.0
    return h, k, lL


def extract_mode_lists(path_modes_txt, path_str, path_cif):
    """
    From the ISODISTORT modes_*.txt file and the corresponding .str file,
    return three parallel lists:

        mode_names   : ['M1', 'M2', ...]
        norm_factors : [0.06831, 0.06402, ...]
        max_params   : [2.0, 2.0, ...]

    Matching is done by comparing the description strings in the .str file
    with the mode headers in the .txt file (ignoring whitespace and the
    'normfactor = ...' part).
    """
    # We only need block_meta from the .txt
    _, block_meta = build_res_per_block_var(path_modes_txt, path_cif)

    # Read the .str file
    text = Path(path_str).read_text(encoding="utf-8", errors="ignore")

    # Lines like:
    # prm  !a1       0.00000 min  -2.00 max  2.00 'P4/mmm[...] A2u(a)
    prm_re = re.compile(
        r"prm\s+!a(\d+)\s+[-\d.]+\s+min\s+[-\d.]+\s+max\s+([-\d.]+)\s+'(.+)"
    )

    prm_matches = prm_re.findall(text)
    if not prm_matches:
        raise ValueError("No 'prm !a#' parameter lines found in .str file")

    def normalize_key(s: str) -> str:
        """
        Remove 'normfactor' part if present and strip all whitespace,
        so that e.g.
        '...[Pr1:d:dsp]A2u(a) normfactor = 0.06831'
        and
        '...[Pr1:d:dsp] A2u(a)'
        match.
        """
        s = s.split("normfactor")[0]
        return re.sub(r"\s+", "", s)

    # Map normalized description from .str -> max value
    desc_to_max = {}
    for idx_str, max_str, desc in prm_matches:
        key = normalize_key(desc)
        desc_to_max[key] = float(max_str)

    mode_names: list[str] = []
    norm_factors: list[float] = []
    max_params: list[float] = []

    # Go through blocks in the same order as in modes_*.txt
    for block_idx, (mode_var, header) in block_meta.items():
        # Extract normfactor from the header
        m = re.search(r"normfactor\s*=\s*([-\d.]+)", header)
        if not m:
            raise ValueError(f"No normfactor found in block header: {header!r}")
        norm_val = float(m.group(1))

        key = normalize_key(header)
        max_val = desc_to_max.get(key)
        if max_val is None:
            raise KeyError(
                f"Could not match block header from txt to prm line in str:\n  {header!r}"
            )

        mode_names.append(header.split()[0])
        norm_factors.append(norm_val)
        max_params.append(max_val)
        mode_dict = {
                name: {"normfactor": nf, "max_param": mp}
                for name, nf, mp in zip(mode_names, norm_factors, max_params)
                }

    return mode_names, norm_factors, max_params, mode_dict


def compute_qnorms_general(hkl_batch, cell_lengths, cell_angles):
    """
    Compute |q| = 2π/d_hkl for a batch of Miller indices in an arbitrary unit cell.

    The function computes 1/d_hkl using the general formula incorporating all six
    lattice parameters (a, b, c, alpha, beta, gamma) and then multiplies by 2π.

    Parameters
    ----------
    hkl_batch : tf.Tensor
        Tensor of shape [N, 3] containing Miller indices (h, k, l).
    cell_lengths : list or tuple
        [a, b, c] lattice parameters in Å.
    cell_angles : list or tuple
        [alpha, beta, gamma] lattice angles in degrees.

    Returns
    -------
    qnorms : tf.Tensor
        Tensor of shape [N] with |q| = 2π/d_hkl values in Å⁻¹.
    """
    a, b, c = [tf.constant(x, dtype=tf.float32) for x in cell_lengths]
    alpha, beta, gamma = [tf.constant(np.deg2rad(x), dtype=tf.float32) for x in cell_angles]
    h, k, l = tf.unstack(tf.cast(hkl_batch, tf.float32), axis=1)

    cos_a, cos_b, cos_g = tf.cos(alpha), tf.cos(beta), tf.cos(gamma)

    sin_sq_term = 1.0 - cos_a ** 2 - cos_b ** 2 - cos_g ** 2 + 2.0 * cos_a * cos_b * cos_g

    V = a * b * c * tf.sqrt(sin_sq_term)

    # Sine terms for the diagonal part
    sin_sq_a = tf.sin(alpha) ** 2
    sin_sq_b = tf.sin(beta) ** 2
    sin_sq_g = tf.sin(gamma) ** 2

    T1 = h ** 2 * (b * c) ** 2 * sin_sq_a
    T2 = k ** 2 * (a * c) ** 2 * sin_sq_b
    T3 = l ** 2 * (a * b) ** 2 * sin_sq_g
    T4 = 2.0 * h * k * a * b * c ** 2 * (cos_a * cos_b - cos_g)
    T5 = 2.0 * k * l * a ** 2 * b * c * (cos_b * cos_g - cos_a)
    T6 = 2.0 * l * h * a * b ** 2 * c * (cos_g * cos_a - cos_b)

    # The full numerator
    Numerator = T1 + T2 + T3 + T4 + T5 + T6

    # Reciprocal squared spacing: 1/d^2_hkl = Numerator / V^2
    inv_d_sq = Numerator / (V ** 2)

    # |q| = 2π/d_hkl = 2π * sqrt(1/d^2_hkl)
    qnorms = 2.0 * tf.constant(np.pi, dtype=tf.float32) * tf.sqrt(inv_d_sq)
    return qnorms


def get_atomic_form_factor(qnorm, atom):
    """
    Function to calculate the relativistic atomic form factor using the 5-term
    and 4-term expansions from Olukayode et al. (2023).

    :param qnorm: Norm of the hkl vector |Q| = 4 * pi * sin(theta) / lambda
    :param atom: Type of atom
    :return: The atomic form factor
    """
    # Convert |Q| to s = sin(theta)/lambda
    s = qnorm / (4.0 * np.pi)

    # ========================================================================
    # 1. RANGE: s <= 2.0 (Uses Table S5: 5-term Gaussian expansion + c)
    # Formula: f(s) = sum(a_i * exp(-b_i * s^2)) + c
    # ========================================================================
    S5_vals = {
        # Format: 'a': [a1, a2, a3, a4, a5], 'b': [b1, b2, b3, b4, b5], 'c': c
        "Ba": {
            'a': tf.constant([24.493274, 21.465054, 2.540001, 9.630249, -9.087347], dtype=tf.float32),
            'b': tf.constant([2.821126, 0.4013595, 33.888096, 18.080001, 1.265068], dtype=tf.float32),
            'c': tf.constant(4.958608, dtype=tf.float32)
        },
        "Cu": {
            'a': tf.constant([6.059708, 11.467765, 7.51863, 1.289357, 0.0000003103094], dtype=tf.float32),
            'b': tf.constant([7.667164, 3.271697, 0.2148349, 19.409789, -2.559956], dtype=tf.float32),
            'c': tf.constant(0.6638525, dtype=tf.float32)
        },
        "O": {
            'a': tf.constant([1.77979, 1.952228, 1.54706, 0.3774445, 3.090503], dtype=tf.float32),
            'b': tf.constant([30.929378, 5.484135, 0.3263118, 87.668801, 12.394058], dtype=tf.float32),
            'c': tf.constant(0.2524927, dtype=tf.float32)
        },
        "La": {
            'a': tf.constant([10.600986, 1.977582, 19.915393, 205.320762, -188.399766], dtype=tf.float32),
            'b': tf.constant([16.113839, 29.273523, 0.3467648, 2.100354, 2.010199], dtype=tf.float32),
            'c': tf.constant(4.585139, dtype=tf.float32)
        },
    }

    # ========================================================================
    # Formula: f(s) = exp(a0 + a1*s + a2*s^2 + a3*s^3 + a4*s^4)
    # ========================================================================
    S7_vals = {
        # PASTE YOUR VALUES FROM TABLE S7 HERE
        # Format: 'a': [a0, a1, a2, a3, a4]
        "Ba": {
            'a': tf.constant([6.40931900, -3.97920200, 1.289065, -0.1940966, 0.01067194], dtype=tf.float32)
        },
        "Cu": {
            'a': tf.constant([2.91085500, -0.58133620, -0.227822300, 0.0770442700, -0.006138708], dtype=tf.float32)
        },
        "O": {
            'a': tf.constant([0.98878340, -0.42671010, -0.205752800, 0.0412286500, -0.002501271], dtype=tf.float32)
        },
        "La": {
            'a': tf.constant([6.29129700, -3.74758100, 1.171638000, -0.1706972000, 0.009078393], dtype=tf.float32)
        },
    }

    # Fallback default (e.g., Oxygen) if atom is not in dictionary
    if atom not in S5_vals:
        atom = "O"

    # --- Calculate Expansion 1 (s <= 2.0) ---
    a5 = S5_vals[atom]['a']
    b5 = S5_vals[atom]['b']
    c5 = S5_vals[atom]['c']

    # tf.vectorized_map passes qnorm as a scalar, so s is a scalar
    fq_1 = tf.reduce_sum(a5 * tf.exp(-b5 * (s ** 2))) + c5

    # --- Calculate Expansion 2 (s > 2.0) ---
    a4 = S7_vals[atom]['a']

    # Calculate polynomial: a0*1 + a1*s + a2*s^2 + a3*s^3 + a4*s^4
    s_powers = tf.stack([1.0, s, s ** 2, s ** 3, s ** 4])
    fq_2 = tf.exp(tf.reduce_sum(a4 * s_powers))

    # --- Piecewise condition ---
    # If s <= 2.0, return Expansion 1, else return Expansion 2
    fq = tf.where(s <= 2.0, fq_1, fq_2)

    return fq


def get_structure_factors(hkl_batch, structure, qnorms):
    """
    Vectorized structure factor calculation including occupancy and B-factors.

    Parameters
    ----------
    hkl_batch : Tensor [N, 3]
        List of N hkl vectors
    structure : List of (atom, Z, position, occupancy, B)
        Atomic basis of the crystal (outputted by shift_atoms)
    qnorms : Tensor [N]
        Norms of the hkl vectors |Q| (assumes q = sin(theta)/lambda)
    Returns
    -------
    Tensor [N] (complex64)
        Structure factors for each hkl
    """
    # 1. Unpack the structure generated by the shift_atoms function
    # Expected format: ['elem', Z, [x, y, z], occ, B]
    atoms = [item[0] for item in structure]
    positions = tf.stack([item[2] for item in structure])  # [A, 3]
    occupancies = tf.stack([item[3] for item in structure])  # [A]
    b_factors = tf.stack([item[4] for item in structure])  # [A]

    # Ensure tensors are float32 for mathematical operations
    positions = tf.cast(positions, tf.float32)
    occupancies = tf.cast(occupancies, tf.float32)
    b_factors = tf.cast(b_factors, tf.float32)
    qnorms = tf.cast(qnorms, tf.float32)

    # 2. Get per-atom form factors per hkl
    fq_table = {
        "La": tf.vectorized_map(lambda q: tf.cast(get_atomic_form_factor(q, "La"), tf.complex64), qnorms),
        "Cu": tf.vectorized_map(lambda q: tf.cast(get_atomic_form_factor(q, "Cu"), tf.complex64), qnorms),
        "O": tf.vectorized_map(lambda q: tf.cast(get_atomic_form_factor(q, "O"), tf.complex64), qnorms),
        "Ba": tf.vectorized_map(lambda q: tf.cast(get_atomic_form_factor(q, "Ba"), tf.complex64), qnorms),
    }  # Each: [N]

    # Build full form factor matrix [N, A]
    fq_matrix = tf.stack([fq_table[atom] for atom in atoms], axis=1)

    # 3. Calculate Isotropic Temperature Factor: exp(-B * q^2)
    # qnorms is [N] -> expand to [N, 1]
    # b_factors is [A] -> expand to [1, A]
    # Resulting temp_factor matrix broadcasts perfectly to [N, A]
    q_squared = tf.square(qnorms / (4.0 * np.pi))  # Convert back to s^2 for the formula
    temp_factor_arg = -tf.expand_dims(q_squared, 1) * tf.expand_dims(b_factors, 0)
    temp_factor = tf.exp(temp_factor_arg)  # [N, A]

    # 4. Compute phase terms: [N, A]
    # Matches equation: exp[ +2 * pi * i * (hx + ky + lz) ]
    phase_arg = tf.tensordot(tf.cast(hkl_batch, tf.float32), tf.transpose(positions), axes=1)  # [N, A]
    phase = tf.exp(tf.complex(0.0, 2.0 * np.pi) * tf.cast(phase_arg, tf.complex64))  # [N, A]

    # 5. Convert occupancies and temp_factor to complex64 for multiplication
    occ_c = tf.cast(occupancies, tf.complex64)  # [A] -> broadcasts to [N, A]
    temp_factor_c = tf.cast(temp_factor, tf.complex64)  # [N, A]

    # Element-wise multiply: occupancy * temp_factor * form_factor * phase
    # Then sum over atoms (axis=1) to get the final F for each hkl reflection
    F_hkl = tf.reduce_sum(occ_c * temp_factor_c * fq_matrix * phase, axis=1)  # [N]

    return F_hkl


def get_structure_factors_optimized(hkl_batch, structure, qnorms, fq_matrix):
    """
    Optimized version using pre-calculated form factors.
    """
    # 1. Unpack positions, occupancies, and B-factors from the structure
    # These are the things that might be changing/trainable
    positions = tf.stack([item[2] for item in structure])  # [A, 3]
    occupancies = tf.stack([item[3] for item in structure])  # [A]
    b_factors = tf.stack([item[4] for item in structure])  # [A]

    positions = tf.cast(positions, tf.float32)
    occupancies = tf.cast(occupancies, tf.float32)
    b_factors = tf.cast(b_factors, tf.float32)

    # 2. Calculate Isotropic Temperature Factor: exp(-B * q^2)
    # Use the pre-calculated qnorms
    q_squared = tf.square(qnorms / (4.0 * np.pi))
    temp_factor_arg = -tf.expand_dims(q_squared, 1) * tf.expand_dims(b_factors, 0)
    temp_factor = tf.exp(temp_factor_arg)

    # 3. Compute phase terms
    phase_arg = tf.tensordot(tf.cast(hkl_batch, tf.float32), tf.transpose(positions), axes=1)
    phase = tf.exp(tf.complex(0.0, 2.0 * np.pi) * tf.cast(phase_arg, tf.complex64))

    # 4. Multiply with pre-calculated fq_matrix
    occ_c = tf.cast(occupancies, tf.complex64)
    temp_factor_c = tf.cast(temp_factor, tf.complex64)

    # All matrices [N, A] are now ready for the final reduction
    F_hkl = tf.reduce_sum(occ_c * temp_factor_c * fq_matrix * phase, axis=1)

    return F_hkl