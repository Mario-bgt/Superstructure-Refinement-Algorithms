import subprocess
import sys
import os

# Define all the iterations you want to run overnight in this list
# n_iter is set to 50 for the SciPy random start guesses
runs = [
    ### Fit childs with occupancy and for all points
    # --- X1+ ---
    {"child_path": "Children/X1+/C1_47", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X1+_C1_47_results.csv", "superstructure_only": False},
    {"child_path": "Children/X1+/P1_123", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X1+_P1_123_results.csv", "superstructure_only": False},
    {"child_path": "Children/X1+/P3_65", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X1+_P3_65_results.csv", "superstructure_only": False},

    # --- X2+ ---
    {"child_path": "Children/X2+/C1_55", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X2+_C1_55_results.csv", "superstructure_only": False},
    {"child_path": "Children/X2+/P1_127", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X2+_P1_127_results.csv", "superstructure_only": False},
    {"child_path": "Children/X2+/P3_64", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X2+_P3_64_results.csv", "superstructure_only": False},

    # --- X2- ---
    {"child_path": "Children/X2-/C1_59", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X2-_C1_59_results.csv", "superstructure_only": False},
    {"child_path": "Children/X2-/P1_129", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X2-_P1_129_results.csv", "superstructure_only": False},
    {"child_path": "Children/X2-/P3_63", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X2-_P3_63_results.csv", "superstructure_only": False},

    # --- X3+ ---
    {"child_path": "Children/X3+/C1_56", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X3+_C1_56_results.csv", "superstructure_only": False},
    {"child_path": "Children/X3+/P1_138", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X3+_P1_138_results.csv", "superstructure_only": False},
    {"child_path": "Children/X3+/P3_64", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X3+_P3_64_results.csv", "superstructure_only": False},

    # --- X3- ---
    {"child_path": "Children/X3-/C1_58", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X3-_C1_58_results.csv", "superstructure_only": False},
    {"child_path": "Children/X3-/P1_136", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X3-_P1_136_results.csv", "superstructure_only": False},
    {"child_path": "Children/X3-/P3_63", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X3-_P3_63_results.csv", "superstructure_only": False},

    # --- X4+ ---
    {"child_path": "Children/X4+/C1_48", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X4+_C1_48_results.csv", "superstructure_only": False},
    {"child_path": "Children/X4+/P1_134", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X4+_P1_134_results.csv", "superstructure_only": False},
    {"child_path": "Children/X4+/P3_66", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X4+_P3_66_results.csv", "superstructure_only": False},

    # --- X4- ---
    {"child_path": "Children/X4-/C1_49", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X4-_C1_49_results.csv", "superstructure_only": False},
    {"child_path": "Children/X4-/P1_132", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X4-_P1_132_results.csv", "superstructure_only": False},
    {"child_path": "Children/X4-/P3_67", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_all_points/X4-_P3_67_results.csv", "superstructure_only": False},

    ### Fit childs with occupancy and for superstructure only
    # --- X1+ ---
    {"child_path": "Children/X1+/C1_47", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X1+_C1_47_results.csv", "superstructure_only": True},
    {"child_path": "Children/X1+/P1_123", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X1+_P1_123_results.csv", "superstructure_only": True},
    {"child_path": "Children/X1+/P3_65", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X1+_P3_65_results.csv", "superstructure_only": True},

    # --- X2+ ---
    {"child_path": "Children/X2+/C1_55", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X2+_C1_55_results.csv", "superstructure_only": True},
    {"child_path": "Children/X2+/P1_127", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X2+_P1_127_results.csv", "superstructure_only": True},
    {"child_path": "Children/X2+/P3_64", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X2+_P3_64_results.csv", "superstructure_only": True},

    # --- X2- ---
    {"child_path": "Children/X2-/C1_59", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X2-_C1_59_results.csv", "superstructure_only": True},
    {"child_path": "Children/X2-/P1_129", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X2-_P1_129_results.csv", "superstructure_only": True},
    {"child_path": "Children/X2-/P3_63", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X2-_P3_63_results.csv", "superstructure_only": True},

    # --- X3+ ---
    {"child_path": "Children/X3+/C1_56", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X3+_C1_56_results.csv", "superstructure_only": True},
    {"child_path": "Children/X3+/P1_138", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X3+_P1_138_results.csv", "superstructure_only": True},
    {"child_path": "Children/X3+/P3_64", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X3+_P3_64_results.csv", "superstructure_only": True},

    # --- X3- ---
    {"child_path": "Children/X3-/C1_58", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X3-_C1_58_results.csv", "superstructure_only": True},
    {"child_path": "Children/X3-/P1_136", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X3-_P1_136_results.csv", "superstructure_only": True},
    {"child_path": "Children/X3-/P3_63", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X3-_P3_63_results.csv", "superstructure_only": True},

    # --- X4+ ---
    {"child_path": "Children/X4+/C1_48", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X4+_C1_48_results.csv", "superstructure_only": True},
    {"child_path": "Children/X4+/P1_134", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X4+_P1_134_results.csv", "superstructure_only": True},
    {"child_path": "Children/X4+/P3_66", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X4+_P3_66_results.csv", "superstructure_only": True},

    # --- X4- ---
    {"child_path": "Children/X4-/C1_49", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X4-_C1_49_results.csv", "superstructure_only": True},
    {"child_path": "Children/X4-/P1_132", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X4-_P1_132_results.csv", "superstructure_only": True},
    {"child_path": "Children/X4-/P3_67", "max_amount_dist": 0.5, "train_occupancy": True, "n_iter": 3000,
     "results_csv_path": "results_leastsq/occ_fit_superpoints/X4-_P3_67_results.csv", "superstructure_only": True},
]


def main():
    # Ensure the target directory for the CSVs exists
    os.makedirs("results", exist_ok=True)

    print("========================================")
    print(f"Starting SciPy overnight batch process for {len(runs)} configurations.")
    print("========================================\n")

    for run_config in runs:
        print(f"--> Initiating run for: {run_config['child_path']}")

        # Build the command arguments
        cmd = [
            sys.executable, "train_model_leastsq.py",
            "--child_path", run_config["child_path"],
            "--max_amount_dist", str(run_config["max_amount_dist"]),
            "--n_iter", str(run_config["n_iter"]),
            "--results_csv_path", run_config["results_csv_path"]
        ]

        # If train_occupancy is True, append the flag
        if run_config["train_occupancy"]:
            cmd.append("--train_occupancy")

        # If superstructure_only is True, append the flag
        if run_config.get("superstructure_only"):
            cmd.append("--superstructure_only")

        try:
            # Run the child process
            subprocess.run(cmd, check=True)
            print(f"--> Completed run for: {run_config['child_path']}\n")
            print("-" * 50)

        except subprocess.CalledProcessError as e:
            print(f"--> ERROR: Process failed for {run_config['child_path']} with exit code {e.returncode}.")
            print("--> Moving to the next configuration...\n")
            print("-" * 50)

    print("\n========================================")
    print("Overnight batch process finished!")
    print("========================================")


if __name__ == "__main__":
    main()