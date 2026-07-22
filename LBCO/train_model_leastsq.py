import os
import sys

# --- HOTFIX FOR OPENMP CRASH ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import multiprocess as mp
from joblib import Parallel, delayed
from time import time
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from functions import *

# ==========================================
# ARGUMENT PARSING
# ==========================================
parser = argparse.ArgumentParser(description="Run structure refinement using SciPy optimization.")
parser.add_argument("--child_path", type=str, required=True, help="Path to the child directory")
parser.add_argument("--max_amount_dist", type=float, default=1.0, help="Max amount of distortion")
parser.add_argument("--train_occupancy", action="store_true", help="Include this flag to train occupancy")
parser.add_argument("--n_iter", type=int, default=50, help="Number of iterations to run")
parser.add_argument("--results_csv_path", type=str, required=True, help="Path to save the results CSV")
parser.add_argument("--superstructure_only", action="store_true",
                    help="If set, only fit against superstructure reflections.")

args = parser.parse_args()

child_path = args.child_path
max_amount_dist = args.max_amount_dist
train_occupancy = args.train_occupancy
n_iter = args.n_iter
results_csv_path = args.results_csv_path
superstructure_only = args.superstructure_only

cpu_count = os.cpu_count()
print(f"Number of CPU cores: {cpu_count}")

# load the necessary files
mode_path = os.path.join(child_path, "mode_details.txt")
topas_path = os.path.join(child_path, "topas.str")
cif_path = os.path.join(child_path, "subgroup.cif")

# generate the function
shift_atoms, source_code, modes, b_factors, occ_factors = emit_python(mode_path, cif_path)
cell_length_a, cell_length_b, cell_length_c, cell_alpha, cell_beta, cell_gamma, sg_nmb, transform_list_hkl = parse_cif_cell_params_and_transform(cif_path)
mode_names, norm_factors, max_A_list, mode_dict = extract_mode_lists(mode_path, topas_path, cif_path)
n_modes = len(mode_names)

# Load the data
experimental_data = get_df("data/lbco10kbc.hkl")

experimental_data['h_p'], experimental_data['k_p'], experimental_data['l_p'] = hkl_LTT_to_HTT(
    experimental_data['h'], experimental_data['k'], experimental_data['l'])

experimental_data['h'], experimental_data['k'], experimental_data['l'] = transform_list_hkl(
    experimental_data['h_p'], experimental_data['k_p'], experimental_data['l_p'])

experimental_data = experimental_data[~((experimental_data['h'] == 0) & (experimental_data['k'] == 0))]
experimental_data = experimental_data[
    ~((experimental_data['h'] == 1) & (experimental_data['k'] == 1) & (experimental_data['l'] == 1))]

# --- SUPERSTRUCTURE FILTERING ---
if superstructure_only:
    tol = 1e-4
    is_hp_non_int = np.abs(experimental_data['h_p'] - np.round(experimental_data['h_p'])) > tol
    is_kp_non_int = np.abs(experimental_data['k_p'] - np.round(experimental_data['k_p'])) > tol
    is_lp_non_int = np.abs(experimental_data['l_p'] - np.round(experimental_data['l_p'])) > tol

    superstructure_mask = is_hp_non_int | is_kp_non_int | is_lp_non_int
    experimental_data = experimental_data[superstructure_mask]
    print(f"Filtered to superstructure peaks only. Remaining reflections: {len(experimental_data)}")

if len(experimental_data) == 0:
    print("ERROR: No reflections remaining after filtering! Exiting cleanly.")
    sys.exit(1)

hkl_list = experimental_data[["h", "k", "l"]].values.tolist()
features = tf.convert_to_tensor(hkl_list, dtype=tf.float32)
n_features = experimental_data.shape[0]

labels = experimental_data["intensity"].tolist()
labels = np.abs(labels) / np.max(labels) * 10e3
labels = tf.convert_to_tensor(labels, dtype=tf.float32)

qnorms_precalc = compute_qnorms_general(features, [cell_length_a, cell_length_b, cell_length_c],
                                        [cell_alpha, cell_beta, cell_gamma])

unique_atoms = ["La", "Cu", "O", "Ba"]
fq_precalc_dict = {}
for atom in unique_atoms:
    fq_precalc_dict[atom] = tf.vectorized_map(
        lambda q: tf.cast(get_atomic_form_factor(q, atom), tf.complex64),
        qnorms_precalc
    )

example_pars = [0.0] * n_modes + [1.0] * (len(b_factors) + len(occ_factors))
base_structure = shift_atoms(*example_pars)

final_base_structure = []
for atom in base_structure:
    final_base_structure.append(atom)
    if atom[0] == 'La':
        ba_atom = ['Ba', 56, atom[2], 1.0 - atom[3], atom[4]]
        final_base_structure.append(ba_atom)

atoms_list = [item[0] for item in final_base_structure]
fq_matrix_precalc = tf.stack([fq_precalc_dict[atom] for atom in atoms_list], axis=1)

np_qnorms = qnorms_precalc.numpy()
np_fq_matrix = fq_matrix_precalc.numpy()

del shift_atoms

# ==========================================
# GLOBAL DEFINITIONS FOR TENSORFLOW WORKERS
# ==========================================
W_SHIFT_ATOMS = None
W_FEATURES = None
W_LABELS = None
W_Q_PRECALC = None
W_FQ_PRECALC = None

@tf.function(reduce_retracing=True)
def compiled_objective(tf_params):
    pars = tf.unstack(tf_params)

    modified_struct = W_SHIFT_ATOMS(*pars[:-1])

    final_struct = []
    for atom in modified_struct:
        final_struct.append(atom)
        if atom[0] == 'La':
            ba_atom = ['Ba', 56, atom[2], 1.0 - atom[3], atom[4]]
            final_struct.append(ba_atom)

    sf = get_structure_factors_optimized(W_FEATURES, final_struct, W_Q_PRECALC, W_FQ_PRECALC)

    intensity = tf.math.real(sf * tf.math.conj(sf))
    intensity = tf.maximum(intensity, 0.0)

    scale = pars[-1]
    y_pred = tf.cast(scale, tf.float32) * intensity / (tf.reduce_max(intensity) + 1e-12) * 10e3

    eps = 1e-9
    num = tf.reduce_sum(tf.abs(tf.sqrt(W_LABELS + eps) - tf.sqrt(y_pred + eps)))
    den = tf.reduce_sum(tf.sqrt(W_LABELS + eps))

    return num / den

# ==========================================
# WORKER FUNCTION
# ==========================================
def run_single_training(config):
    import os
    import random
    import time
    from scipy.optimize import minimize

    global W_SHIFT_ATOMS, W_FEATURES, W_LABELS, W_Q_PRECALC, W_FQ_PRECALC

    run_seed = config['seed']
    os.environ['PYTHONHASHSEED'] = str(run_seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRA_OP_PARALLELISM_THREADS"] = "1"
    os.environ["TF_NUM_INTER_OP_PARALLELISM_THREADS"] = "1"

    import tensorflow as tf
    import numpy as np

    random.seed(run_seed)
    np.random.seed(run_seed)
    tf.random.set_seed(run_seed)

    from functions import get_structure_factors_optimized

    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass

    exec(config['source_code'], globals())
    W_SHIFT_ATOMS = globals()['shift_atoms']
    W_Q_PRECALC = tf.constant(config['qnorms_precalc'], dtype=tf.float32)
    W_FQ_PRECALC = tf.constant(config['fq_matrix_precalc'], dtype=tf.complex64)
    W_FEATURES = tf.constant(config['features'], dtype=tf.float32)
    W_LABELS = tf.constant(config['combined_labels'], dtype=tf.float32)

    len_modes = len(config['modes'])
    len_occ = len(config['occ_factors'])
    len_b = len(config['b_factors'])

    # 3. Define Bounds, Initial Guess, and Step Sizes (eps)
    p0 = []
    bounds = []
    eps_array = []  # <--- Create the array to hold our custom step sizes
    fixed_occs = []

    # Init Modes
    for a_max, n_fac in zip(config['max_A_list'], config['norm_factors']):
        limit = config['max_par_value'] * a_max * n_fac
        p0.append(np.random.normal(0.0, 0.1 * limit))
        bounds.append((-limit, limit))
        eps_array.append(5e-6)  # Small step for highly sensitive structural modes

    # Init B-factors
    for _ in config['b_factors']:
        p0.append(np.random.normal(1, 0.5))
        bounds.append((0.0, 4.0))
        eps_array.append(1e-3)  # Large step for B-factors so the optimizer can "feel" them

    # Init Occ
    for occ_name in config['occ_factors']:
        init_val = 0.875 if 'La' in occ_name else 1.0
        if config['train_occupancy']:
            p0.append(init_val)
            bounds.append((0.0, 1.0))
            eps_array.append(1e-3)  # Large step for occupancies if we are training them
        else:
            fixed_occs.append(init_val)

    # Init Scale
    p0.append(1.0)
    bounds.append((0.2, 20.0))
    eps_array.append(5e-6)  # Small step for the global scale factor

    # 4. Direct Physical Objective Function with Parameter Masking
    def objective_function(params):
        if not config['train_occupancy']:
            # Reconstruct the full list: [modes..., b_factors..., fixed_occs..., scale]
            # params[:-1] gets modes + b_factors. params[-1] is the scale.
            full_params = list(params[:-1]) + fixed_occs + [params[-1]]
        else:
            full_params = list(params)

        tf_params = tf.convert_to_tensor(full_params, dtype=tf.float32)
        loss = compiled_objective(tf_params)

        if np.isnan(loss.numpy()) or np.isinf(loss.numpy()):
            return 1e6

        return float(loss.numpy())

    ## 5. Execute SciPy Optimization
    fit_start_time = time.time()
    try:
        res = minimize(
            objective_function,
            x0=p0,
            bounds=bounds,
            method='L-BFGS-B',
            options={
                'ftol': 1e-7,       # Matches float32 precision limits
                'gtol': 1e-7,
                'maxiter': 10000,
                'disp': False,
                'eps': eps_array    # <--- Pass the array instead of a single float
            }
        )
        final_pars_raw = res.x.tolist()
        final_loss = res.fun
    except Exception as e:
        print(f"Run {config['iter_index']} failed: {e}")
        final_pars_raw = p0
        final_loss = 1e6

    fit_duration = time.time() - fit_start_time

    # 6. Reconstruct the Final Variables for saving
    if not config['train_occupancy']:
        final_pars = list(final_pars_raw[:-1]) + fixed_occs + [final_pars_raw[-1]]
    else:
        final_pars = final_pars_raw

    current_modes = final_pars[:len_modes]
    current_bs = final_pars[len_modes:len_modes + len_b]
    current_occ = final_pars[len_modes + len_b:len_modes + len_b + len_occ]
    current_scale = final_pars[-1]

    # Save output data
    temp_data = {
        'Run': [config['iter_index']],
        'Seed': [run_seed],
        'Fit_Time_s': [fit_duration]
    }

    for i, m_name in enumerate(config['mode_names']):
        temp_data[m_name] = [current_modes[i]]

    for i, B in enumerate(config['b_factors']):
        temp_data[f'{B}'] = [current_bs[i]]

    for i, occ in enumerate(config['occ_factors']):
        temp_data[f'{occ}'] = [current_occ[i]]

    temp_data['R_factor'] = [final_loss]

    return {
        'final_loss': final_loss,
        'pars': final_pars, # Return the fully reconstructed parameters
        'temp_data': temp_data
    }

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == '__main__':
    time_start = time.time()

    BASE_SEED = 521651

    print(f"Packaging tasks for {child_path} and starting Joblib parallel pool...")

    tasks = []
    for i in range(n_iter):
        tasks.append({
            'iter_index': i,
            'seed': BASE_SEED + i,
            'max_par_value': max_amount_dist,
            'train_occupancy': train_occupancy,
            'features': np.array(features),
            'combined_labels': np.array(labels),
            'max_A_list': np.array(max_A_list),
            'norm_factors': np.array(norm_factors),
            'source_code': source_code,
            'modes': modes,
            'mode_names': mode_names,
            'b_factors': b_factors,
            'occ_factors': occ_factors,
            'qnorms_precalc': np_qnorms,
            'fq_matrix_precalc': np_fq_matrix
        })

    results = Parallel(n_jobs=cpu_count, verbose=49)(delayed(run_single_training)(task) for task in tasks)

    best_result = min(results, key=lambda x: x['final_loss'])
    best_model_pars = best_result['pars']
    min_loss = best_result['final_loss']

    sim_data = {key: [] for key in results[0]['temp_data'].keys()}
    for r in results:
        for key in sim_data:
            sim_data[key].extend(r['temp_data'][key])

    sim_df = pd.DataFrame(sim_data)
    sim_df.to_csv(results_csv_path, index=False)

    print(f"Time elapsed for {child_path}: {time.time() - time_start:.2f} seconds")
    print(f"Best Final Loss: {min_loss:.3e}")

    print("Maximum parameter value:", max_amount_dist)
    print("Number of iterations:", n_iter)
    print("Number of features:", n_features)

    # ==========================================
    # CALCULATE AND SAVE BEST FIT INTENSITIES
    # ==========================================
    print("Calculating final simulated intensities for the best fit...")

    exec(source_code, globals())

    best_modified_struct = shift_atoms(*best_model_pars[:-1])

    best_final_struct = []
    for atom in best_modified_struct:
        best_final_struct.append(atom)
        if atom[0] == 'La':
            ba_atom = ['Ba', 56, atom[2], 1.0 - atom[3], atom[4]]
            best_final_struct.append(ba_atom)

    best_sf = get_structure_factors_optimized(features, best_final_struct, qnorms_precalc, fq_matrix_precalc)

    best_intensity = tf.math.real(best_sf * tf.math.conj(best_sf))
    best_intensity = tf.maximum(best_intensity, 0.0)

    best_scale = best_model_pars[-1]
    best_y_pred = tf.cast(best_scale, tf.float32) * best_intensity / (tf.reduce_max(best_intensity) + 1e-12) * 10e3

    experimental_data['intensity_sim'] = best_y_pred.numpy()
    intensities_csv_path = results_csv_path.replace(".csv", "_fit_intensities.csv")
    experimental_data.to_csv(intensities_csv_path, index=False)
    print(f"Saved experimental and simulated intensities to: {intensities_csv_path}")