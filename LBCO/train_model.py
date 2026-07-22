import os

# --- FIX: Prevent OpenMP Library Clash on Windows ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import multiprocess as mp
from joblib import Parallel, delayed
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
import tensorflow as tf
from functions import *

# ==========================================
# ARGUMENT PARSING
# ==========================================
parser = argparse.ArgumentParser(description="Run structure refinement.")
parser.add_argument("--child_path", type=str, required=True, help="Path to the child directory")
parser.add_argument("--max_amount_dist", type=float, default=1.0, help="Max amount of distortion")
parser.add_argument("--train_occupancy", action="store_true", help="Include this flag to train occupancy")
parser.add_argument("--n_iter", type=int, default=100, help="Number of iterations to run")
parser.add_argument("--results_csv_path", type=str, required=True, help="Path to save the results CSV")
parser.add_argument("--superstructure_only", action="store_true",
                    help="If set, only fit against superstructure reflections (non-integer parent h, k, or l).")

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
cell_length_a, cell_length_b, cell_length_c, cell_alpha, cell_beta, cell_gamma, sg_nmb, transform_list_hkl = parse_cif_cell_params_and_transform(
    cif_path)
mode_names, norm_factors, max_A_list, mode_dict = extract_mode_lists(mode_path, topas_path, cif_path)
n_modes = len(mode_names)

# Load the data
experimental_data = get_df("data/lbco10kbc.hkl")
experimental_data["intensity"] = np.abs(experimental_data["intensity"] / np.max(experimental_data["intensity"]) * 1e3)

# Special case here, because the data is in LTT coordinates, but we need HTT coordinates for the model
experimental_data['h_p'], experimental_data['k_p'], experimental_data['l_p'] = hkl_LTT_to_HTT(experimental_data['h'],
                                                                                              experimental_data['k'],
                                                                                              experimental_data['l'])

# --- SUPERSTRUCTURE FILTERING ---
if superstructure_only:
    # A peak is a superstructure peak if h_p, k_p, or l_p is not an integer in the parent cell.
    # We use a small tolerance (1e-4) to account for floating point inaccuracies.
    tol = 1e-4
    is_hp_non_int = np.abs(experimental_data['h_p'] - np.round(experimental_data['h_p'])) > tol
    is_kp_non_int = np.abs(experimental_data['k_p'] - np.round(experimental_data['k_p'])) > tol
    is_lp_non_int = np.abs(experimental_data['l_p'] - np.round(experimental_data['l_p'])) > tol

    superstructure_mask = is_hp_non_int | is_kp_non_int | is_lp_non_int
    experimental_data = experimental_data[superstructure_mask]
    print(f"Filtered to superstructure peaks only. Remaining reflections: {len(experimental_data)}")

# Transform to child cell coordinates using the transformation matrix from the CIF
experimental_data['h'], experimental_data['k'], experimental_data['l'] = transform_list_hkl(experimental_data['h_p'],
                                                                                            experimental_data['k_p'],
                                                                                            experimental_data['l_p'])

# Filter out the (0,0,l) and (1,1,1) reflections
experimental_data = experimental_data[~((experimental_data['h'] == 0) & (experimental_data['k'] == 0))]
experimental_data = experimental_data[
    ~((experimental_data['h'] == 1) & (experimental_data['k'] == 1) & (experimental_data['l'] == 1))]

hkl_list = experimental_data[["h", "k", "l"]].values.tolist()
features = tf.convert_to_tensor(hkl_list, dtype=tf.float32)
n_features = experimental_data.shape[0]

labels = experimental_data["intensity"].tolist()
labels = np.abs(labels) / np.max(labels) * 10e3
labels = tf.convert_to_tensor(labels, dtype=tf.float32)

# Precalc to enhance efficiency during training
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

# --- Inject Ba into the base structure so the precalc matrix matches the 36 atoms! ---
final_base_structure = []
for atom in base_structure:
    final_base_structure.append(atom)
    if atom[0] == 'La':
        # Same logic used inside fun_tf
        ba_atom = ['Ba', 56, atom[2], 1.0 - atom[3], atom[4]]
        final_base_structure.append(ba_atom)

# Build the precalc matrix from the 36-atom list
atoms_list = [item[0] for item in final_base_structure]
fq_matrix_precalc = tf.stack([fq_precalc_dict[atom] for atom in atoms_list], axis=1)

# Convert tensors to numpy for safe multiprocessing transit
np_qnorms = qnorms_precalc.numpy()
np_fq_matrix = fq_matrix_precalc.numpy()

# Clean up global namespace to prevent pickling errors
del shift_atoms


# ==========================================
# WORKER FUNCTION (Encapsulates everything)
# ==========================================
def run_single_training(config):
    import os
    import random
    import time  # Imported locally to track fit duration

    # 1. Extract Seed and force Determinism
    run_seed = config['seed']
    os.environ['PYTHONHASHSEED'] = str(run_seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRA_OP_PARALLELISM_THREADS"] = "1"
    os.environ["TF_NUM_INTER_OP_PARALLELISM_THREADS"] = "1"

    import tensorflow as tf
    import numpy as np

    # Set seeds for all random number generators
    random.seed(run_seed)
    np.random.seed(run_seed)
    tf.random.set_seed(run_seed)

    # Needs to be imported inside worker
    from functions import get_structure_factors_optimized

    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass

    # 2. Recreate the dynamic function securely inside this worker's namespace
    exec(config['source_code'], globals())

    # 3. Re-cast precalculated matrices back to Tensors
    q_precalc = tf.constant(config['qnorms_precalc'], dtype=tf.float32)
    fq_precalc = tf.constant(config['fq_matrix_precalc'], dtype=tf.complex64)
    features = tf.constant(config['features'], dtype=tf.float32)
    combined_labels = tf.constant(config['combined_labels'], dtype=tf.float32)

    # 4. Define the TF function locally
    def fun_tf(x, pars):
        modified_struct = globals()['shift_atoms'](*pars[:-1])

        # --- FIX 1: LBCO Barium Injection ---
        final_struct = []
        for atom in modified_struct:
            final_struct.append(atom)
            if atom[0] == 'La':
                # atom is formatted: ['La', 57, [x, y, z], occ, B]
                # We append Ba (Z=56) at the same xyz, with occ = 1 - La_occ, and the same B-factor
                ba_atom = ['Ba', 56, atom[2], 1.0 - atom[3], atom[4]]
                final_struct.append(ba_atom)

        # Use final_struct instead of modified_struct
        sf = get_structure_factors_optimized(x, final_struct, q_precalc, fq_precalc)

        intensity = tf.math.real(sf * tf.math.conj(sf))
        intensity = tf.maximum(intensity, 0.0)

        scale = pars[-1]
        return tf.cast(scale, tf.float32) * intensity / (tf.reduce_max(intensity) + 1e-12) * 10e3

    # 5. Define the Layer locally
    class FunAsLayer(tf.keras.layers.Layer):
        def __init__(self, modes, b_factors, occ_factors, max_A_list, norm_factors, max_par_value_t, train_occ,
                     **kwargs):
            super().__init__(**kwargs)
            self.modes = modes
            self.b_factors = b_factors
            self.occ_factors = occ_factors
            self.max_A_list = max_A_list
            self.norm_factors = norm_factors
            self.max_par_value = max_par_value_t
            self.train_occ = train_occ

        def build(self, input_shape):
            self.mode_weights = [
                self.add_weight(name=m, shape=(),
                                initializer=tf.keras.initializers.RandomNormal(0.0, 0.4, seed=run_seed),
                                trainable=True) for m in self.modes]

            # --- FIX 3: Initialize using inverse sigmoid ---
            def inv_sigmoid(y):
                if y >= 1.0: return 15.0  # sigmoid(15) is effectively 1.0
                if y <= 0.0: return -15.0
                return np.log(y / (1.0 - y))

            self.occ_weights = []
            for o in self.occ_factors:
                target_val = 0.875 if 'La' in o else 1.0
                init_val = inv_sigmoid(target_val)
                self.occ_weights.append(
                    self.add_weight(name=o, shape=(),
                                    initializer=tf.keras.initializers.Constant(init_val),
                                    trainable=self.train_occ))

            self.b_weights = [
                self.add_weight(name=b, shape=(), initializer=tf.keras.initializers.Constant(0.95), trainable=True) for
                b in self.b_factors]
            self.scale_weight = self.add_weight(name='scale', shape=(), initializer=tf.keras.initializers.Constant(1.0),
                                                trainable=True)
            super().build(input_shape)

        def call(self, inputs_t):
            # --- FIX 2: Multiply by norm_factors here! ---
            mode_vals = [
                self.max_par_value * tf.tanh(w) * a_max * n_fac
                for w, a_max, n_fac in zip(self.mode_weights, self.max_A_list, self.norm_factors)
            ]
            occ_vals = [tf.math.sigmoid(w) for w in self.occ_weights]
            b_vals = [4.0 * tf.math.sigmoid(w) for w in self.b_weights]
            scale_val = 0.2 + 19.8 * tf.math.sigmoid(self.scale_weight)

            pars = mode_vals + occ_vals + b_vals + [scale_val]
            return fun_tf(inputs_t, pars)

        def compute_output_shape(self, input_shape):
            return input_shape[0], 1

    # 6. Define Losses/Metrics locally
    class RFactorLoss(tf.keras.losses.Loss):
        def call(self, y_true, y_pred):
            eps = 1e-9
            num = tf.reduce_sum(tf.abs(tf.sqrt(y_true + eps) - tf.sqrt(y_pred + eps)))
            den = tf.reduce_sum(tf.sqrt(y_true + eps))
            return num / den

    def r_factor_metric(y_true, y_pred):
        eps = 1e-9
        num = tf.reduce_sum(tf.abs(tf.sqrt(y_true + eps) - tf.sqrt(y_pred + eps)))
        den = tf.reduce_sum(tf.sqrt(y_true + eps))
        return num / den

    def custom_lr_schedule(epoch, current_lr):
        warmup_epochs, decay_epochs = 2000, 13000
        max_lr, min_lr = 1e-1, 1e-6
        if epoch < warmup_epochs:
            return min_lr + (max_lr - min_lr) * (epoch / warmup_epochs)
        else:
            progress = min(1.0, (epoch - warmup_epochs) / decay_epochs)
            return min_lr + (max_lr - min_lr) * (0.5 * (1.0 + np.cos(np.pi * progress)))

    # --- MODEL SETUP & FIT ---
    inputs = tf.keras.Input(shape=(config['n_dim'],))
    outputs = FunAsLayer(
        modes=config['modes'],
        b_factors=config['b_factors'],
        occ_factors=config['occ_factors'],
        max_A_list=config['max_A_list'],
        norm_factors=config['norm_factors'],
        max_par_value_t=config['max_par_value'],
        train_occ=config['train_occupancy']
    )(inputs)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=RFactorLoss(), metrics=[r_factor_metric])

    # Start timing right before fit execution
    fit_start_time = time.time()

    history = model.fit(
        x=features, y=combined_labels,
        batch_size=features.shape[0],
        epochs=config['n_epochs'],
        verbose=0, shuffle=False,
        callbacks=[tf.keras.callbacks.LearningRateScheduler(custom_lr_schedule)]
    )

    # Calculate duration
    fit_end_time = time.time()
    fit_duration = fit_end_time - fit_start_time

    # --- EXTRACT WEIGHTS ---
    final_loss = history.history['loss'][-1]

    # Grab the custom layer directly
    custom_layer = model.layers[-1]

    # Pull from the explicit class attributes instead of a flat list
    current_modes = [
        config['max_par_value'] * np.tanh(w.numpy()) * config['max_A_list'][j] * config['norm_factors'][j]
        for j, w in enumerate(custom_layer.mode_weights)
    ]

    current_occ = [tf.math.sigmoid(w).numpy() for w in custom_layer.occ_weights]
    current_bs = [(4.0 * tf.math.sigmoid(w)).numpy() for w in custom_layer.b_weights]
    current_scale = [(0.2 + 19.8 * tf.math.sigmoid(custom_layer.scale_weight)).numpy()]

    # Save the seed and timing into the output data!
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

    temp_data['R_factor'] = [history.history['r_factor_metric'][-1]]

    return {
        'loss_history': history.history['loss'],
        'final_loss': final_loss,
        'pars': current_modes + current_occ + current_bs + current_scale,
        'temp_data': temp_data
    }


# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == '__main__':
    time_start = time.time()
    n_epochs = 15000

    # Establish a base seed for reproducibility across the entire run
    BASE_SEED = 521651

    print(f"Packaging tasks for {child_path} and starting Joblib parallel pool...")

    tasks = []
    for i in range(n_iter):
        tasks.append({
            'iter_index': i,
            'seed': BASE_SEED + i,
            'n_dim': 3,
            'max_par_value': max_amount_dist,
            'train_occupancy': train_occupancy,
            'features': np.array(features),
            'combined_labels': np.array(labels),
            'n_epochs': n_epochs,
            'max_A_list': np.array(max_A_list),
            'norm_factors': np.array(norm_factors),
            'source_code': source_code,
            'modes': modes,
            'mode_names': mode_names,  # Passed here so workers can use it for column headers
            'b_factors': b_factors,
            'occ_factors': occ_factors,
            'qnorms_precalc': np_qnorms,
            'fq_matrix_precalc': np_fq_matrix
        })

    # The magic line
    results = Parallel(n_jobs=cpu_count, verbose=49)(delayed(run_single_training)(task) for task in tasks)

    # --- PROCESS RESULTS ---
    all_losses = [r['loss_history'] for r in results]
    best_result = min(results, key=lambda x: x['final_loss'])
    best_model_pars = best_result['pars']
    min_loss = best_result['final_loss']

    # Combine dictionary results
    sim_data = {key: [] for key in results[0]['temp_data'].keys()}
    for r in results:
        for key in sim_data:
            sim_data[key].extend(r['temp_data'][key])

    sim_df = pd.DataFrame(sim_data)
    sim_df.to_csv(results_csv_path, index=False)

    print(f"Time elapsed for {child_path}: {time.time() - time_start:.2f} seconds")
    print(f"Best Final Loss: {min_loss:.3e}")

    # ── Publication-Quality Matplotlib Settings ──────────────────────────────────────
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10

    # Create the figure with thesis-friendly dimensions and high DPI
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # 1. Plot all trajectories as a transparent background ensemble
    for loss_vals in all_losses:
        ax.plot(loss_vals, color='dimgrey', alpha=0.15, linewidth=0.8)

    # 2. Highlight the best converging trajectory (lowest final R-factor)
    best_loss_idx = np.argmin([loss[-1] for loss in all_losses])
    ax.plot(all_losses[best_loss_idx], color='black', alpha=1.0, linewidth=1.5, label='Best Fit')

    # 3. Scientific Formatting
    ax.set_xscale('log')
    ax.set_xlabel(r"Epoch")
    ax.set_ylabel(r"$R$-Factor")
    ax.grid(False)
    ax.legend(frameon=False, loc='upper right')

    plt.tight_layout()

    # Save as PDF for lossless vector insertion into LaTeX
    plot_name = results_csv_path.replace(".csv", "_loss_plot.pdf")
    plt.savefig(plot_name, bbox_inches='tight')
    plt.close()

    # Print the model summary
    print("Maximum parameter value:", max_amount_dist)
    print("Number of iterations:", n_iter)
    print("Number of features:", n_features)

    # ==========================================
    # CALCULATE AND SAVE BEST FIT INTENSITIES
    # ==========================================
    print("Calculating final simulated intensities for the best fit...")

    # Re-instantiate the dynamic function in the main thread
    exec(source_code, globals())

    # Generate the displaced structure using the best parameters
    best_modified_struct = shift_atoms(*best_model_pars[:-1])

    best_final_struct = []
    for atom in best_modified_struct:
        best_final_struct.append(atom)
        if atom[0] == 'La':
            ba_atom = ['Ba', 56, atom[2], 1.0 - atom[3], atom[4]]
            best_final_struct.append(ba_atom)

    # Calculate structure factors
    best_sf = get_structure_factors_optimized(features, best_final_struct, qnorms_precalc, fq_matrix_precalc)

    best_intensity = tf.math.real(best_sf * tf.math.conj(best_sf))
    best_intensity = tf.maximum(best_intensity, 0.0)

    best_scale = best_model_pars[-1]
    best_y_pred = tf.cast(best_scale, tf.float32) * best_intensity / (tf.reduce_max(best_intensity) + 1e-12) * 10e3

    # Append to the dataframe
    experimental_data['intensity_sim'] = best_y_pred.numpy()

    # Save the dataframe
    intensities_csv_path = results_csv_path.replace(".csv", "_fit_intensities.csv")
    experimental_data.to_csv(intensities_csv_path, index=False)
    print(f"Saved experimental and simulated intensities to: {intensities_csv_path}")