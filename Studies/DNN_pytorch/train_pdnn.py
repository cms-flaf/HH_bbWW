#!/usr/bin/env python3
import os
import argparse
import yaml
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import refactored helper functions, data loaders, and model architectures
from src.model_helper import (
    eval_parity_expr,
    load_parametric_fold,
    load_physical_fold,
    update_mass_dependent_features,
    DynamicMassManager,
    ParametricDNN,
    ONNXDeployWrapper,
    WeightedStandardCrossEntropyLoss,
)

# Use Apple Silicon MPS if available, otherwise GPU/CPU
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


def main():
    parser = argparse.ArgumentParser(
        description="Train Parametric DNNs using model_config.yaml"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="model_config.yaml",
        help="Path to YAML model configuration file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    print(f"--> Loading configuration settings from: {args.config}")
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve paths & target evaluation parameters
    input_folder = cfg["input_folder"]
    output_folder = cfg["output_folder"]
    tree_name = cfg.get("tree_name", "AppliedTree")
    dedicated_mass = cfg.get("dedicated_mass", None)

    # Class mapping & configuration
    class_names = cfg.get("class_names", ["Signal", "TT", "Other"])
    num_classes = len(class_names)

    # ONNX Model Naming Pattern
    model_name_fmt = cfg.get("model_name", "pdnn_model_{regime}_nparity{nParity}.onnx")

    # Parity settings
    n_parity_folds = cfg.get("nParity", 4)
    train_parity_expr = cfg.get("train_parity").get("index")
    test_parity_expr = cfg.get("test_parity").get("index")
    val_parity_expr = cfg.get("val_parity").get("index")
    app_parity_expr = cfg.get("app_parity").get("index")

    if dedicated_mass is not None:
        eval_mass_points = [dedicated_mass]
        output_folder = f"{output_folder}_M{dedicated_mass}"
    else:
        eval_mass_points = cfg["signal_mass_points"]

    # Map dynamic regime feature vectors
    feature_map = {}
    for regime in cfg["regimes"]:
        feature_map[regime] = cfg["common_features"] + cfg["regime_features"].get(
            regime, []
        )

    extra_vars = cfg["extra_variables"]
    train_cfg = cfg["training"]

    print(f"Using compute acceleration device: {device}")
    print(f"Configured output classes ({num_classes}): {class_names}")
    feature_importance_records = {regime: [] for regime in cfg["regimes"]}

    # Loop over cross-validation parities defined in YAML configuration
    for i_fold in range(n_parity_folds):
        # Dynamically evaluate target parity indices using string expressions from config
        train_parity = eval_parity_expr(train_parity_expr, i_fold)
        test_parity = eval_parity_expr(test_parity_expr, i_fold)
        val_parity = eval_parity_expr(val_parity_expr, i_fold)
        app_parity = eval_parity_expr(app_parity_expr, i_fold)

        out_dir = os.path.join(output_folder, f"nParity{train_parity}_validation")
        os.makedirs(out_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print(
            f"RUNNING DUAL-REGIME PARITY CYCLE {i_fold} --> Train: {train_parity} | Test: {test_parity} | Val: {val_parity} | App: {app_parity}"
        )
        print("=" * 80 + "\n")

        files = {
            "train": {
                "data": os.path.join(
                    input_folder, f"nParity{train_parity}_Merged.root"
                ),
                "weight": os.path.join(
                    input_folder, f"nParity{train_parity}_Merged_weight.root"
                ),
            },
            "test": {
                "data": os.path.join(input_folder, f"nParity{test_parity}_Merged.root"),
                "weight": os.path.join(
                    input_folder, f"nParity{test_parity}_Merged_weight.root"
                ),
            },
            "val": {
                "data": os.path.join(input_folder, f"nParity{val_parity}_Merged.root"),
                "weight": os.path.join(
                    input_folder, f"nParity{val_parity}_Merged_weight.root"
                ),
            },
        }

        df_train_raw = load_parametric_fold(
            files["train"]["data"],
            files["train"]["weight"],
            feature_map,
            extra_vars,
            cfg["signal_mass_points"],
            seed=100,
            dedicated_mass=dedicated_mass,
        )
        df_test_raw = load_parametric_fold(
            files["test"]["data"],
            files["test"]["weight"],
            feature_map,
            extra_vars,
            cfg["signal_mass_points"],
            seed=200,
            dedicated_mass=dedicated_mass,
        )
        df_val_raw = load_physical_fold(
            files["val"]["data"],
            files["val"]["weight"],
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
            print(f"\n--- Initiating Network Training for Regime: {regime.upper()} ---")
            current_features = feature_map[regime]

            val_mask = (
                (df_val_raw["boosted"] == 1)
                if regime == "boosted"
                else (df_val_raw["boosted"] == 0)
            )
            train_mask = (
                (df_train_raw["boosted"] == 1)
                if regime == "boosted"
                else (df_train_raw["boosted"] == 0)
            )
            test_mask = (
                (df_test_raw["boosted"] == 1)
                if regime == "boosted"
                else (df_test_raw["boosted"] == 0)
            )

            df_train = (
                df_train_raw[train_mask]
                .copy()
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
            )
            df_test = (
                df_test_raw[test_mask]
                .copy()
                .sample(frac=1, random_state=200)
                .reset_index(drop=True)
            )
            df_val = (
                df_val_raw[val_mask]
                .copy()
                .sample(frac=1, random_state=300)
                .reset_index(drop=True)
            )

            if len(df_train) == 0 or len(df_test) == 0:
                print(
                    f"[WARNING] Insufficient statistics for regime {regime}. Skipping."
                )
                continue

            # Class rebalancing dynamically across num_classes
            for df_split in [df_train, df_test]:
                y_arr = df_split["class_targets_multiclass"]
                w_arr = df_split["class_weight"]
                sums = {c: w_arr[y_arr == c].sum() for c in range(num_classes)}
                target_sum = sums.get(0, 1.0)  # Signal class index 0 as target baseline
                if target_sum <= 0:
                    target_sum = 1.0

                for c in range(num_classes):
                    mask = y_arr == c
                    if sums[c] > 0:
                        df_split.loc[mask, "class_weight"] *= target_sum / sums[c]

            scaler = StandardScaler()
            scaler.fit(df_train[current_features])

            train_manager = DynamicMassManager(
                df_train,
                current_features,
                scaler,
                cfg["signal_mass_points"],
                is_training=True,
            )
            test_manager = DynamicMassManager(
                df_test,
                current_features,
                scaler,
                cfg["signal_mass_points"],
                is_training=False,
            )

            X_test_tensor, y_test_tensor, w_test_tensor = (
                test_manager.get_scaled_tensors()
            )
            test_dataset = torch.utils.data.TensorDataset(
                X_test_tensor, y_test_tensor, w_test_tensor
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=train_cfg["eval_batch_size"],
                shuffle=False,
                num_workers=0,
            )

            X_tr_init, y_tr_init, w_tr_init = train_manager.get_scaled_tensors()
            epoch_dataset = torch.utils.data.TensorDataset(
                X_tr_init, y_tr_init, w_tr_init
            )
            train_loader = DataLoader(
                epoch_dataset,
                batch_size=train_cfg["batch_size"],
                shuffle=True,
                drop_last=True,
                num_workers=0,
            )

            model = ParametricDNN(
                input_dim=len(current_features),
                num_classes=num_classes,
                hidden_dims=train_cfg.get("hidden_dims", [512, 256, 128, 64]),
                dropout_rate=train_cfg.get("dropout_rate", 0.35),
            ).to(device)

            criterion = WeightedStandardCrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=train_cfg["learning_rate"],
                weight_decay=train_cfg["weight_decay"],
            )

            sched_cfg = train_cfg.get("lr_scheduler", {})
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=sched_cfg.get("patience", 6),
                factor=sched_cfg.get("factor", 0.5),
            )

            best_loss = float("inf")
            epochs = train_cfg["epochs"]
            early_stopping_patience = train_cfg.get("early_stopping_patience", 10)
            epochs_no_improve = 0

            for epoch in range(1, epochs + 1):
                train_manager.randomize_background_masses()
                X_train_epoch, y_train_epoch, w_train_epoch = (
                    train_manager.get_scaled_tensors()
                )
                train_loader.dataset.tensors = (
                    X_train_epoch,
                    y_train_epoch,
                    w_train_epoch,
                )

                model.train()
                train_loss = 0.0
                for batch_x, batch_y, batch_w in train_loader:
                    batch_x, batch_y, batch_w = (
                        batch_x.to(device),
                        batch_y.to(device),
                        batch_w.to(device),
                    )
                    optimizer.zero_grad()

                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y, batch_w)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * len(batch_y)

                train_loss /= len(X_train_epoch)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y, batch_w in test_loader:
                        batch_x, batch_y, batch_w = (
                            batch_x.to(device),
                            batch_y.to(device),
                            batch_w.to(device),
                        )
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y, batch_w)
                        val_loss += loss.item() * len(batch_y)
                val_loss /= len(test_dataset)

                scheduler.step(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_no_improve = 0
                    torch.save(
                        model.state_dict(),
                        os.path.join(out_dir, f"best_pdnn_model_{regime}.pth"),
                    )
                else:
                    epochs_no_improve += 1

                if epoch % 10 == 0 or epoch == 1:
                    print(
                        f"    [{regime.upper()}] Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.5f} | Test Loss: {val_loss:.5f}"
                    )

                if epochs_no_improve >= early_stopping_patience:
                    print(
                        f"    [{regime.upper()}] Early stopping triggered at epoch {epoch}. Restoring best weights."
                    )
                    break

            model.load_state_dict(
                torch.load(os.path.join(out_dir, f"best_pdnn_model_{regime}.pth"))
            )
            model.eval()

            # Export ONNX model with embedded standard scaler dynamically formatted
            onnx_wrapper = ONNXDeployWrapper(model.to("cpu"), scaler)
            onnx_wrapper.eval()

            dummy_input = torch.randn(1, len(current_features), dtype=torch.float32)
            onnx_filename = model_name_fmt.format(nParity=i_fold, regime=regime)
            onnx_path = os.path.join(out_dir, onnx_filename)

            torch.onnx.export(
                onnx_wrapper,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=["raw_inputs"],
                output_names=["probabilities"],
                dynamic_axes={
                    "raw_inputs": {0: "batch_size"},
                    "probabilities": {0: "batch_size"},
                },
            )
            print(
                f"    [{regime.upper()}] Successfully exported ONNX model to: {onnx_path}"
            )

            model = model.to(device)
            model.eval()

            eps = 1e-15
            val_indices = np.where(val_mask)[0]
            df_val_temp = df_val_raw[val_mask].copy()

            val_scaled_mat_placeholder = torch.zeros(
                (1, len(current_features)), dtype=torch.float32
            )
            v_dataset = torch.utils.data.TensorDataset(val_scaled_mat_placeholder)
            v_loader = DataLoader(
                v_dataset, batch_size=32768, shuffle=False, num_workers=0
            )

            for mass in eval_mass_points:
                df_val_temp["X_mass"] = mass
                df_val_temp = update_mass_dependent_features(
                    df_val_temp, current_features
                )

                val_scaled_mat = torch.tensor(
                    scaler.transform(df_val_temp[current_features]), dtype=torch.float32
                )
                v_loader.dataset.tensors = (val_scaled_mat,)

                outputs_list = []
                with torch.no_grad():
                    for (batch_x,) in v_loader:
                        batch_x = batch_x.to(device)
                        logits = model(batch_x)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                        outputs_list.append(probs)

                outputs = np.concatenate(outputs_list, axis=0)
                val_prob_sig = outputs[:, 0]
                logit_score = np.log(
                    np.clip(val_prob_sig, eps, 1.0 - eps)
                    / np.clip(1.0 - val_prob_sig, eps, 1.0 - eps)
                )

                # Dynamic branch generation based on class_names
                targets_to_save = [(f"DNN_M{mass}", logit_score)]
                for c_idx, c_name in enumerate(class_names):
                    targets_to_save.append(
                        (f"DNN_M{mass}_prob_{c_name.lower()}", outputs[:, c_idx])
                    )

                for key, source_arr in targets_to_save:
                    if key not in val_output_data:
                        val_output_data[key] = np.zeros(
                            len(df_val_raw), dtype=np.float32
                        )
                    val_output_data[key][val_indices] = source_arr.astype(np.float32)

            # Feature Importance Scanning
            perm_frac = train_cfg.get("perm_subsample_fraction", 0.20)
            if len(df_val) > 0:
                print(
                    f"--> Initiating feature importance scans for {regime.upper()}..."
                )

                df_sub_base = (
                    df_val.sample(frac=perm_frac, random_state=42).copy()
                    if perm_frac < 1.0
                    else df_val.copy()
                )
                mass_importance_records = {mass: {} for mass in eval_mass_points}

                tensor_mat_placeholder = torch.zeros(
                    (1, len(current_features)), dtype=torch.float32
                )
                perm_dataset = torch.utils.data.TensorDataset(tensor_mat_placeholder)
                perm_loader = DataLoader(
                    perm_dataset, batch_size=32768, shuffle=False, num_workers=0
                )

                def compute_batched_loss(X_matrix, y_targets):
                    tensor_mat = torch.tensor(X_matrix, dtype=torch.float32)
                    perm_loader.dataset.tensors = (tensor_mat,)
                    prob_list = []
                    with torch.no_grad():
                        for (bx,) in perm_loader:
                            bx = bx.to(device)
                            lg = model(bx)
                            prob_list.append(torch.softmax(lg, dim=1).cpu().numpy())
                    return log_loss(
                        y_targets,
                        np.concatenate(prob_list, axis=0),
                        labels=list(range(num_classes)),
                    )

                for mass in eval_mass_points:
                    df_mass_eval = df_sub_base.copy()
                    df_mass_eval["X_mass"] = mass
                    df_mass_eval = update_mass_dependent_features(
                        df_mass_eval, current_features
                    )

                    X_importance_base = scaler.transform(df_mass_eval[current_features])
                    y_importance_base = df_mass_eval["class_targets_multiclass"].values

                    base_loss_val = compute_batched_loss(
                        X_importance_base, y_importance_base
                    )

                    for feat_name in current_features:
                        if feat_name == "X_mass":
                            mass_importance_records[mass][feat_name] = 0.0
                            continue

                        df_permuted = df_mass_eval.copy()
                        permuted_vals = df_permuted[feat_name].values.copy()
                        np.random.shuffle(permuted_vals)
                        df_permuted[feat_name] = permuted_vals

                        df_permuted = update_mass_dependent_features(
                            df_permuted, current_features
                        )

                        X_permuted_scaled = scaler.transform(
                            df_permuted[current_features]
                        )
                        perm_loss = compute_batched_loss(
                            X_permuted_scaled, y_importance_base
                        )

                        delta_loss = max(0.0, perm_loss - base_loss_val)
                        mass_importance_records[mass][feat_name] = delta_loss

                feature_importance_records[regime].append(mass_importance_records)

        # Output ROOT File Construction
        output_root_filename = os.path.join(out_dir, "validation_applied.root")
        print(f"\n--> Saving applied dataset tree to '{output_root_filename}'...")
        with uproot.recreate(output_root_filename) as f_out:
            f_out.mktree(tree_name, {k: v.dtype for k, v in val_output_data.items()})
            f_out[tree_name].extend(val_output_data)

    # Summary Plotting
    print("\n" + "=" * 80)
    print("GENERATING FEATURE IMPORTANCE REPORTS & TREND PLOTS")
    print("=" * 80)

    for regime in cfg["regimes"]:
        if not feature_importance_records[regime]:
            continue

        pdf_path = os.path.join(
            output_folder, f"per_mass_feature_importance_{regime}.pdf"
        )
        print(f"--> Compiling multi-page report document: '{pdf_path}'")

        folds_list = feature_importance_records[regime]
        regime_mass_averages = {}

        with PdfPages(pdf_path) as pdf:
            for mass in eval_mass_points:
                feat_averages = {}
                sample_features = list(folds_list[0][mass].keys())

                for feat in sample_features:
                    feat_vals = [fold[mass][feat] for fold in folds_list]
                    feat_averages[feat] = np.mean(feat_vals)

                regime_mass_averages[mass] = feat_averages.copy()

                s_importance = pd.Series(feat_averages).sort_values(ascending=True)
                s_importance = s_importance.drop(labels=["X_mass"], errors="ignore")

                fig, ax = plt.subplots(figsize=(12, 8))
                ax.barh(
                    s_importance.index,
                    s_importance.values,
                    color="crimson" if regime == "boosted" else "royalblue",
                    edgecolor="black",
                    alpha=0.8,
                )
                ax.set_xlabel(
                    "Mean Shift in Cross-Entropy Loss (Permuted - Baseline)",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.set_ylabel("Kinematic Feature Name", fontsize=11, fontweight="bold")
                ax.set_title(
                    f"{regime.upper()} Topology Feature Importance (M_X = {mass} GeV)",
                    fontsize=13,
                    fontweight="bold",
                    pad=15,
                )
                ax.grid(True, axis="x", linestyle="--", alpha=0.5)
                plt.tight_layout()

                pdf.savefig(fig, dpi=150)
                plt.close(fig)

        if len(eval_mass_points) > 1:
            df_summary = pd.DataFrame.from_dict(regime_mass_averages, orient="index")
            if "X_mass" in df_summary.columns:
                df_summary = df_summary.drop(columns=["X_mass"])

            df_features_vs_mass = df_summary.T
            top_n = min(15, len(df_features_vs_mass))
            top_features = df_features_vs_mass.mean(axis=1).nlargest(top_n).index
            df_top_features = df_features_vs_mass.loc[top_features]

            plt.figure(figsize=(12, 7))
            for feature in df_top_features.index:
                plt.plot(
                    df_top_features.columns,
                    df_top_features.loc[feature],
                    marker="o",
                    markersize=5,
                    linewidth=2,
                    label=feature,
                )

            plt.title(
                f"{regime.upper()} Topology: Top {top_n} Feature Importances Across Mass Points",
                fontsize=14,
                fontweight="bold",
                pad=15,
            )
            plt.xlabel("Mass Point $M_X$ (GeV)", fontsize=12)
            plt.ylabel("Mean Shift in Cross-Entropy Loss", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.xticks(eval_mass_points, rotation=45)
            plt.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                title="Kinematic Features",
                frameon=True,
            )
            plt.tight_layout()

            trend_plot_path = os.path.join(
                output_folder, f"feature_importance_evolution_lines_{regime}.png"
            )
            plt.savefig(trend_plot_path, dpi=300)
            plt.close()

    print("\n--> Training execution complete.")


if __name__ == "__main__":
    main()
