# Superstructure-Refinement-Algorithms# Superstructure Refinement Algorithms

Code developed for a master's thesis on refining candidate low-symmetry ("child") crystal
structures against single-crystal diffraction data, using symmetry-mode decompositions
from **ISODISTORT** and structure-factor calculations implemented in **TensorFlow**.

The current test system is **LBCO** (La–Ba–Cu–O), where a set of candidate distorted structures
(one per irreducible representation / isotropy subgroup, e.g. `X1+`, `X2+`, `X2-`, `X3+`, `X3-`,
`X4+`, `X4-`) is fit against a measured set of `(h, k, l, intensity)` reflections to determine
which distortion mode(s) best explain the observed superstructure peaks.

## How it works

1. **Mode parsing (`functions.py`).** For a given child structure, ISODISTORT's
   `mode_details.txt` (atomic displacement decomposition) and `subgroup.cif` (undistorted basis)
   + cell/symmetry metadata) are parsed to build a Python function `shift_atoms(*mode_amplitudes,
   *B_factors, *occupancies)` that returns the atomic basis `[element, Z, [x, y, z], occupancy,
   B]` as a function of the trainable symmetry-mode amplitudes. Mode normalization factors and
   per-mode amplitude bounds are cross-checked against the corresponding TOPAS `.str` file.
2. **Structure factors.** `compute_qnorms_general` and `get_atomic_form_factor` compute |Q| and
   relativistic atomic form factors (5-term Gaussian + polynomial expansions) for every
   reflection, so that `get_structure_factors_optimized` can evaluate F(hkl) with pre-cached form
   factors for speed. Intensities are `|F|²`, scaled and compared to the measured data with a
   normalized R-factor loss: `R = Σ|√I_obs − √I_calc| / Σ√I_obs`.
3. **Optimization.** Each child structure is refined with many independent random-start runs in
   parallel (`joblib` + `multiprocess`), and the best (lowest R-factor) run is kept. Two
   interchangeable optimizers are provided:
   - `train_model.py` – wraps the model as a Keras layer and trains mode amplitudes,
     B-factors, occupancies and a global scale factor with the Adam optimizer, a custom
     warmup/cosine-decay learning-rate schedule, and reparameterizations (`tanh`/`sigmoid`) to
     keep parameters within physical bounds.
   - `train_model_leastsq.py` – the same physical model refined with SciPy's
     `scipy.optimize.minimize` (L-BFGS-B, bounded, per-parameter step sizes), which is
     considerably faster and used for the larger multi-start batches.
4. **Batch driving.** `batch_runner.py` / `batch_runner_leastsq.py` (and the `*_noocc` variants,
   which run the same refinements with occupancies held fixed rather than trained) loop over all
   child structures and both reflection subsets — all reflections, and superstructure-only
   reflections (peaks whose parent-cell indices are non-integer) — invoking the corresponding
   `train_model*.py` script as a subprocess for each configuration.
5. **Analysis.** `analyze_fit.ipynb` is used to inspect the resulting CSVs (best R-factors,
   refined parameters, simulated vs. observed intensities) across children and settings.

## Repository structure

```
LBCO/
├── Children/                      # One subfolder per candidate child structure, e.g.
│   ├── X1+/C1_47/                 #   mode_details.txt, topas.str, subgroup.cif
│   ├── X1+/P1_123/
│   ├── X1+/P3_65/
│   └── ...                        # X2+, X2-, X3+, X3-, X4+, X4- (each with C1/P1/P3 settings)
├── data/
│   └── lbco10kbc.hkl              # Measured (h, k, l, intensity, sigma) reflection list
├── Results/                       # Output of train_model.py / batch_runner.py runs
├── results_leastsq/               # Output of train_model_leastsq.py / batch_runner_leastsq.py runs
├── functions.py                   # Mode parsing, structure-factor & form-factor math, I/O helpers
├── train_model.py                 # Single-run TensorFlow/Keras (Adam) refinement, CLI entry point
├── train_model_leastsq.py         # Single-run SciPy (L-BFGS-B) refinement, CLI entry point
├── batch_runner.py                # Drives train_model.py over all children (occupancy trained)
├── batch_runner_noocc.py          # Same, with occupancies held fixed
├── batch_runner_leastsq.py        # Drives train_model_leastsq.py over all children (occupancy trained)
├── batch_runner_leastsq_noocc.py  # Same, with occupancies held fixed
├── analyze_fit.ipynb              # Notebook for inspecting/plotting refinement results
└── .gitignore
```

> Each child directory is expected to contain the three files exported from ISODISTORT/TOPAS for
> that candidate structure: `mode_details.txt`, `topas.str`, and `subgroup.cif`.

## Requirements

- Python 3.9+
- `tensorflow`
- `numpy`, `pandas`, `scipy`
- `joblib`, `multiprocess`
- `pymatgen`, `xrayutilities`
- `matplotlib` (loss-curve plotting in `train_model.py`)

```bash
pip install tensorflow numpy pandas scipy joblib multiprocess pymatgen xrayutilities matplotlib
```

`KMP_DUPLICATE_LIB_OK=TRUE` is set at the top of both training scripts as a workaround for an
OpenMP library clash that can occur on Windows when TensorFlow, SciPy and pymatgen are all
linked against their own copies of OpenMP.

## Usage

### Single refinement run

```bash
python train_model_leastsq.py \
    --child_path Children/X1+/C1_47 \
    --max_amount_dist 0.5 \
    --n_iter 3000 \
    --train_occupancy \
    --superstructure_only \
    --results_csv_path results_leastsq/occ_fit_superpoints/X1+_C1_47_results.csv
```

The TensorFlow/Adam version (`train_model.py`) takes the same arguments and additionally saves a
publication-style loss-curve PDF alongside the results CSV.

**Arguments**

| Flag | Description | Default |
|---|---|---|
| `--child_path` | Path to the child directory containing `mode_details.txt`, `topas.str`, `subgroup.cif` | *required* |
| `--results_csv_path` | Where to write the per-run results CSV | *required* |
| `--max_amount_dist` | Scales the maximum allowed mode amplitude (fraction of ISODISTORT's max value) | `1.0` |
| `--train_occupancy` | If set, La/Ba site occupancies are refined; otherwise held at their initial values | off |
| `--n_iter` | Number of independent random-start optimization runs (parallelized with joblib) | `50` (leastsq) / `100` (TF) |
| `--superstructure_only` | If set, fit only reflections with non-integer parent-cell indices | off |

### Batch runs

To reproduce a full overnight sweep across all child structures, both reflection subsets, and
both occupancy settings, edit the `runs` list at the top of the desired batch script (paths,
`n_iter`, `max_amount_dist`, etc.) and run:

```bash
python batch_runner_leastsq.py
```

Each entry in `runs` is passed to `train_model_leastsq.py` as a subprocess; a failed run logs its
exit code and the batch continues with the next configuration.

## Outputs

For each run, two CSVs are produced next to `--results_csv_path`:

- `<name>_results.csv` — one row per random-start iteration: seed, fit time, refined mode
  amplitudes / B-factors / occupancies, and final R-factor.
- `<name>_fit_intensities.csv` — the experimental reflection list with an added
  `intensity_sim` column, computed from the best (lowest R-factor) run.

`train_model.py` additionally writes `<name>_loss_plot.pdf`, showing the R-factor trajectory of
every random-start run (light grey) with the best-converging run highlighted in black.

## Notes

- Both optimizers minimize the same physical R-factor loss and should, in principle, converge to
  similar minima; the SciPy/L-BFGS-B version is faster per run and is used for larger
  multi-start batches (thousands of starts), while the TensorFlow/Adam version profed to be more consistent in finding a minima, but taking longer.
- `BASE_SEED = 521651` is fixed in both training scripts so that iteration `i` always uses seed
  `BASE_SEED + i`, making individual runs reproducible.