#!/usr/bin/env python3
import os
import argparse
import yaml
import uproot
import numpy as np
import onnxruntime as ort

# Import shared physics utilities and dataset loaders from model_helper
from src.model_helper import (
    load_physical_fold,
    update_mass_dependent_features,
)


# ==============================================================================
# Main ONNX Application Loop
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run ONNX evaluation using model_config.yaml"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="model_config.yaml",
        help="Path to YAML model configuration file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found at: {args.config}")

    print(f"--> Loading model configuration from: {args.config}")
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve paths & target evaluation folders
    input_folder = cfg["input_folder"]
    output_folder = cfg["output_folder"]
    tree_name = cfg.get("tree_name", "AppliedTree")
    dedicated_mass = cfg.get("dedicated_mass", None)

    if dedicated_mass is not None:
        eval_mass_points = [dedicated_mass]
        output_folder = f"{output_folder}_M{dedicated_mass}"
    else:
        eval_mass_points = cfg["signal_mass_points"]

    # Construct feature maps per regime dynamically
    feature_map = {}
    for regime in cfg["regimes"]:
        feature_map[regime] = cfg["common_features"] + cfg["regime_features"].get(
            regime, []
        )

    extra_vars = cfg["extra_variables"]
    eps = 1e-15

    # Run inference loop across all 4 parity folds
    for train_parity in range(4):
        val_parity = (train_parity + 2) % 4

        val_dir = os.path.join(output_folder, f"nParity{train_parity}_validation")
        output_root_filename = os.path.join(val_dir, "validation_applied_onnx.root")

        print("\n" + "=" * 80)
        print(
            f"ONNX EVALUATION FOR PARITY CYCLE {train_parity} (Val Fold: {val_parity})"
        )
        print("=" * 80)

        val_data_path = os.path.join(input_folder, f"nParity{val_parity}_Merged.root")
        val_weight_path = os.path.join(
            input_folder, f"nParity{val_parity}_Merged_weight.root"
        )

        if not os.path.exists(val_data_path):
            print(
                f"[WARNING] Validation path '{val_data_path}' not found. Skipping fold."
            )
            continue

        # Load physical fold using loaded config properties
        df_val_raw = load_physical_fold(
            val_data_path,
            val_weight_path,
            feature_map,
            extra_vars,
            dedicated_mass=dedicated_mass,
        )

        val_output_data = {
            "class_value": df_val_raw["class_value"].values.astype(np.int32),
            "weight_Central": df_val_raw["weight_Central"].values.astype(np.float32),
            "DeepHME_mass": df_val_raw["DeepHME_mass"].values.astype(np.float32),
            "res2b": df_val_raw["res2b"].values.astype(np.int32),
            "recovery": df_val_raw["recovery"].values.astype(np.int32),
            "boosted": df_val_raw["boosted"].values.astype(np.int32),
            "channelId": df_val_raw["channelId"].values.astype(np.int32),
            "X_mass": df_val_raw["X_mass"].values.astype(np.float32),
        }

        for regime in cfg["regimes"]:
            onnx_model_path = os.path.join(
                val_dir, f"pdnn_model_{regime}_nparity{train_parity}.onnx"
            )

            if not os.path.exists(onnx_model_path):
                print(
                    f"[WARNING] ONNX model missing at '{onnx_model_path}'. Skipping regime {regime}."
                )
                continue

            print(f"\n--> Running ONNX Session for Regime: {regime.upper()}")
            current_features = feature_map[regime]
            val_mask = (
                (df_val_raw["boosted"] == 1)
                if regime == "boosted"
                else (df_val_raw["boosted"] == 0)
            )
            val_indices = np.where(val_mask)[0]

            if len(val_indices) == 0:
                print(
                    f"    No events found matching topology mask for {regime}. Skipping."
                )
                continue

            # Load ONNX Session
            session = ort.InferenceSession(
                onnx_model_path, providers=["CPUExecutionProvider"]
            )
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            df_val_temp = df_val_raw[val_mask].copy()

            for mass in eval_mass_points:
                df_val_temp["X_mass"] = mass
                df_val_temp = update_mass_dependent_features(
                    df_val_temp, feature_map[regime]
                )

                # Unscaled raw features passed directly to ONNX graph (embedded scaler node handles normalization)
                X_raw_mat = df_val_temp[current_features].values.astype(np.float32)
                print(f"Using current features {current_features}")
                print(f"And X_raw_mat is:")
                print(X_raw_mat)

                batch_size = 32768
                outputs_list = []

                for i in range(0, len(X_raw_mat), batch_size):
                    batch_x = X_raw_mat[i : i + batch_size]
                    probs = session.run([output_name], {input_name: batch_x})[0]
                    outputs_list.append(probs)

                outputs = np.concatenate(outputs_list, axis=0)
                val_prob_sig = outputs[:, 0]
                logit_score = np.log(
                    np.clip(val_prob_sig, eps, 1.0 - eps)
                    / np.clip(1.0 - val_prob_sig, eps, 1.0 - eps)
                )

                for key, source_arr in [
                    (f"DNN_M{mass}", logit_score),
                    (f"DNN_M{mass}_prob_sig", val_prob_sig),
                    (f"DNN_M{mass}_prob_tt", outputs[:, 1]),
                    (f"DNN_M{mass}_prob_other", outputs[:, 2]),
                ]:
                    if key not in val_output_data:
                        val_output_data[key] = np.zeros(
                            len(df_val_raw), dtype=np.float32
                        )
                    val_output_data[key][val_indices] = source_arr.astype(np.float32)

        print(f"\n--> Saving ONNX validation tree to '{output_root_filename}'...")
        with uproot.recreate(output_root_filename) as f_out:
            f_out.mktree(tree_name, {k: v.dtype for k, v in val_output_data.items()})
            f_out[tree_name].extend(val_output_data)
        print(f" Successfully written: {output_root_filename}")


if __name__ == "__main__":
    main()
