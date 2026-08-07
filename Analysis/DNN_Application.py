#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import gc
import yaml
import numpy as np
import awkward as ak
import onnxruntime as ort


class DNNProducer:
    def __init__(self, cfg: dict, payload_name: str, period: str):
        """
        Inference producer for Parametric DNN models with NaN/Inf diagnostics.

        Parameters
        ----------
        cfg : dict
            Master payload configuration dictionary containing channel definitions and columns.
        payload_name : str
            Prefix used for output branches in the snapshot array.
        period : str
            Data-taking period string (e.g. "e2018").
        """
        self.cfg = cfg
        self.payload_name = payload_name
        self.period = period

        # Ensure analysis module path is loaded
        analysis_path = os.environ.get("ANALYSIS_PATH", ".")
        if analysis_path not in sys.path:
            sys.path.append(analysis_path)

        self.channels = ["DL_Resolved", "DL_Boosted", "SL_Resolved", "SL_Boosted"]
        self.cfg_dict = {ch: self.cfg.get(ch, None) for ch in self.channels}

        self.target_columns = self.cfg.get("columns", [])
        self.target_masses = set()
        self.classes_to_save = set()

        # Parse requested target masses and class names from the column list
        for col in self.target_columns:
            parts = col.split("_", 1)
            if len(parts) == 2 and parts[0].startswith("M"):
                try:
                    self.target_masses.add(float(parts[0][1:]))
                    self.classes_to_save.add(parts[1])
                except ValueError:
                    pass

        self.target_masses = sorted(list(self.target_masses))
        self.dnnConfig = {}
        load_features = set()

        # Build channel-specific configurations and locate ONNX models
        for channel, ch_cfg in self.cfg_dict.items():
            if ch_cfg is None:
                continue

            version = ch_cfg.get("version", "")
            dnnFolder = os.path.join(analysis_path, "config", "DNN", version)
            config_file = os.path.join(dnnFolder, "dnn_config.yaml")

            if not os.path.exists(config_file):
                print(
                    f"[WARNING] Config not found: '{config_file}'. Skipping channel {channel}."
                )
                continue

            with open(config_file, "r") as f:
                model_config = yaml.safe_load(f)

            nParity = model_config.get("nParity", 4)
            features = model_config.get("features", [])
            regime = "boosted" if "Boosted" in channel else "resolved"
            model_name = model_config.get(
                "model_name", "pdnn_model_{regime}_nparity{fold}.onnx"
            )

            # `{fold}` is the cross-validation fold index (0..nParity-1). It is
            # deliberately not called `{nParity}`: nParity is the *number* of
            # folds, and conflating the two is what allows a fold index to be
            # silently substituted where a fold count is meant.
            model_paths = []
            for fold_idx in range(nParity):
                onnx_path = os.path.join(
                    dnnFolder,
                    model_name.format(fold=fold_idx, nParity=nParity, regime=regime),
                )
                model_paths.append(onnx_path)

            model_config["model_paths"] = model_paths
            self.dnnConfig[channel] = model_config

            load_features.update(features)

        self.vars_to_save = set()
        for f in load_features:
            if f == "X_mass":
                continue
            if f.endswith("_scaled"):
                # Dynamically determine unscaled base branch
                self.vars_to_save.add(f.replace("_scaled", ""))
            else:
                self.vars_to_save.add(f)

        mandatory_branches = {"FullEventId", "event", "SL", "DL", "boosted"}
        self.vars_to_save.update(mandatory_branches)

        print(f"[pDNNProducer] Initialized for payload '{self.payload_name}'.")
        print(f" -> Configured Masses : {self.target_masses}")
        print(f" -> Output Classes    : {self.classes_to_save}")
        print(f" -> Input Variables   : {len(self.vars_to_save)} branches")

    def run(self, array: ak.Array) -> ak.Array:
        """
        Executes inference on an input Awkward Array and returns a pruned payload array.
        """
        print(f"[pDNNProducer] Processing batch of {len(array)} events...")

        # 1. Apply ONNX evaluation across configured channels/models
        array = self.ApplyDNN(array)

        # 2. Select prediction per event based on SL/DL and Boosted/Resolved topology
        array = self.selectDNN(array)

        # 3. Assemble and prefix final output columns
        out_fields = {}
        if "FullEventId" in array.fields:
            out_fields["FullEventId"] = array["FullEventId"]

        for col in self.target_columns:
            target_name = f"{self.payload_name}_{col}"
            if col in array.fields:
                out_fields[target_name] = array[col]
            else:
                print(
                    f"[WARNING] Column '{col}' missing in predictions! Filling with zeros."
                )
                out_fields[target_name] = np.zeros(len(array), dtype=np.float32)

        return ak.Array(out_fields)

    def _audit_and_sanitize_inputs(
        self, X_mat: np.ndarray, features: list[str], channel: str, mass: float
    ) -> np.ndarray:
        """
        Scans input feature matrix for NaN/Inf, reports offending variables, and sanitizes array.
        """
        nan_mask = np.isnan(X_mat)
        inf_mask = np.isinf(X_mat)

        if nan_mask.any() or inf_mask.any():
            print("\n" + "!" * 80)
            print(
                f"[pDNNProducer DIAGNOSTIC ERROR] Invalid values detected in Input Features!"
            )
            print(f" -> Channel: {channel} | Mass Point: {mass} GeV")

            for col_idx, feat_name in enumerate(features):
                col_nans = nan_mask[:, col_idx].sum()
                col_infs = inf_mask[:, col_idx].sum()

                if col_nans > 0 or col_infs > 0:
                    bad_rows = np.where(nan_mask[:, col_idx] | inf_mask[:, col_idx])[0]
                    sample_events = bad_rows[:5]  # Display up to first 5 row indices
                    print(
                        f"   [!] Feature '{feat_name}' (col {col_idx}): {col_nans} NaNs, {col_infs} Infs. (Example row indices: {sample_events})"
                    )

            print(
                " -> Automating sanitization: Replacing NaNs/Infs with 0.0 to prevent model failure."
            )
            print("!" * 80 + "\n")

            X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)

        return X_mat

    def _audit_predictions(
        self, array: np.ndarray, stage_label: str, channel: str, mass: float
    ) -> None:
        """
        Checks intermediate predictions and logits for NaN or Inf values.
        """
        if np.isnan(array).any() or np.isinf(array).any():
            nan_count = np.isnan(array).sum()
            inf_count = np.isinf(array).sum()
            print("\n" + "!" * 80)
            print(
                f"[pDNNProducer DIAGNOSTIC ERROR] Invalid outputs found at stage '{stage_label}'!"
            )
            print(f" -> Channel: {channel} | Mass: {mass} GeV")
            print(f" -> NaNs count: {nan_count} | Infs count: {inf_count}")
            print("!" * 80 + "\n")

    def _compute_fold_masks(
        self,
        dnnConfig: dict,
        channel: str,
        event_number: np.ndarray,
        nParity: int,
    ) -> list[np.ndarray]:
        """
        Builds the per-fold application masks, one boolean array per fold.

        Fold `f` is applied to events satisfying
        ``(event_number + f + offset) % nParity == 0``, where ``offset`` comes
        from the ``app_parity`` section of the model configuration. The masks
        are required to partition the events: every event must be claimed by
        exactly one fold, or the ensemble below would silently drop or
        double-count it.
        """
        app_parity_cfg = dnnConfig.get("app_parity")
        if app_parity_cfg is None:
            raise RuntimeError(
                f"[pDNNProducer] Missing required 'app_parity' section in channel '{channel}' configuration!"
            )

        offset = (
            app_parity_cfg.get("offset")
            if isinstance(app_parity_cfg, dict)
            else app_parity_cfg
        )

        if isinstance(offset, bool) or not isinstance(offset, int):
            # Legacy configurations expressed the fold assignment as a Python
            # snippet under 'func'. Support it, but evaluate it with no builtins
            # and only the fold variables in scope -- a model directory is data,
            # and data must not be able to execute arbitrary code.
            legacy_expr = (
                app_parity_cfg.get("func")
                if isinstance(app_parity_cfg, dict)
                else app_parity_cfg
            )
            if not isinstance(legacy_expr, str):
                raise RuntimeError(
                    f"[pDNNProducer] Channel '{channel}': 'app_parity' must provide an integer "
                    f"'offset' (or a legacy 'func' string); got {app_parity_cfg!r}."
                )

            print(
                f"[WARNING] Channel '{channel}': 'app_parity' uses the deprecated 'func' "
                "expression. Replace it with an integer 'offset' -- see the comment in "
                "dnn_config.yaml."
            )

            masks = []
            for fold_idx in range(nParity):
                try:
                    formatted = legacy_expr.format(fold=fold_idx, nParity=nParity)
                except (KeyError, IndexError) as exc:
                    raise RuntimeError(
                        f"[pDNNProducer] Channel '{channel}': cannot substitute fold index into "
                        f"app_parity func {legacy_expr!r} ({exc!r}). It must contain '{{fold}}'."
                    ) from exc

                value = eval(
                    formatted,
                    {"__builtins__": {}},
                    {"event_number": event_number, "nParity": nParity},
                )
                masks.append(np.asarray(value) == 0)
        else:
            masks = [
                ((event_number + fold_idx + offset) % nParity) == 0
                for fold_idx in range(nParity)
            ]

        # The folds must tile the events exactly once each. This single check
        # catches a wrong modulus, a fold offset that does not vary with the
        # fold index, and an event_number that does not match the training one.
        coverage = np.sum(np.stack(masks, axis=0), axis=0)
        if not np.all(coverage == 1):
            unclaimed = int(np.count_nonzero(coverage == 0))
            multiclaimed = int(np.count_nonzero(coverage > 1))
            raise RuntimeError(
                f"[pDNNProducer] Channel '{channel}': the {nParity} application folds do not "
                f"partition the events ({unclaimed} events claimed by no fold, {multiclaimed} "
                f"claimed by more than one). Check 'nParity' and the 'app_parity' offset in the "
                f"model configuration."
            )

        return masks

    def ApplyDNN(self, branches: ak.Array) -> ak.Array:
        """
        Evaluates ONNX inference sessions across channels and parametric mass hypotheses.
        """
        nEvents = len(branches)
        if nEvents == 0:
            return branches

        # The fold assignment must be computed from the *same* quantity that
        # defined the training folds. There is deliberately no fallback to
        # FullEventId: that is a packed (crc16(dataset), crc16(file), entry)
        # identifier with no relation to the event number, so falling back to it
        # would still produce a valid-looking partition while scoring events
        # with folds that were trained on them.
        if "event" not in branches.fields:
            raise RuntimeError(
                "[pDNNProducer] Required branch 'event' is missing from the input array; "
                "it defines the cross-validation folds and has no safe substitute."
            )

        event_number = np.asarray(branches.event)
        if not np.issubdtype(event_number.dtype, np.integer):
            raise RuntimeError(
                f"[pDNNProducer] Branch 'event' has non-integer dtype '{event_number.dtype}'. "
                "Fold assignment uses modular arithmetic and needs exact integers "
                "(float64 silently loses the low bits of large event numbers)."
            )

        output_fields = {}

        for channel, dnnConfig in self.dnnConfig.items():
            if not dnnConfig:
                continue

            features = dnnConfig.get("features", [])
            class_names = dnnConfig.get("class_names", ["Signal", "TT", "Other"])
            nClasses = len(class_names)
            nParity = dnnConfig.get("nParity", 4)
            model_paths = dnnConfig.get("model_paths", [])

            # Fold assignment depends only on the event number, not on the mass
            # hypothesis, so it is computed once per channel rather than once
            # per (mass, fold) pair.
            fold_masks = self._compute_fold_masks(
                dnnConfig, channel, event_number, nParity
            )

            # Prepare ONNX Inference Sessions. Every fold must be present: the
            # fold masks partition the events, so a single missing model leaves
            # its events with all-zero probabilities, which the logit transform
            # below turns into a finite, plausible-looking score (~-16.1) rather
            # than an obvious failure.
            missing = [path for path in model_paths if not os.path.exists(path)]
            if missing or len(model_paths) != nParity:
                raise RuntimeError(
                    f"[pDNNProducer] Channel '{channel}': expected {nParity} ONNX models, "
                    f"found {len(model_paths) - len(missing)}. Missing: {missing}"
                )

            sessions = []
            for p_idx, path in enumerate(model_paths):
                sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                sessions.append((p_idx, sess))

            for mass in self.target_masses:
                # Build feature inputs per event
                feature_cols = []
                for feat in features:
                    if feat == "X_mass":
                        feature_cols.append(np.full(nEvents, mass, dtype=np.float32))
                    elif feat.endswith("_scaled"):
                        base_feat = feat.replace("_scaled", "")
                        if base_feat in branches.fields:
                            raw_val = np.asarray(branches[base_feat], dtype=np.float32)
                        else:
                            print(
                                f"[WARNING] Missing base feature '{base_feat}' for '{feat}'! Filling with 0.0."
                            )
                            raw_val = np.zeros(nEvents, dtype=np.float32)
                        feature_cols.append(raw_val * mass)
                    elif feat in branches.fields:
                        feature_cols.append(
                            np.asarray(branches[feat], dtype=np.float32)
                        )
                    else:
                        print(
                            f"[WARNING] Missing feature branch '{feat}' in input array! Filling with 0.0."
                        )
                        feature_cols.append(np.zeros(nEvents, dtype=np.float32))

                X_mat = np.column_stack(feature_cols).astype(np.float32)

                # AUDIT STEP 1: Audit and clean input feature matrix
                X_mat = self._audit_and_sanitize_inputs(X_mat, features, channel, mass)

                all_predictions = np.zeros((nEvents, nClasses), dtype=np.float32)

                # Run inference fold by fold. Each fold is evaluated only on the
                # events it owns, rather than on all events followed by masking,
                # which costs nParity times less ONNX work.
                for parity_idx, sess in sessions:
                    app_mask = fold_masks[parity_idx]
                    if not app_mask.any():
                        continue

                    input_name = sess.get_inputs()[0].name
                    preds = sess.run(None, {input_name: X_mat[app_mask]})[0]

                    # AUDIT STEP 2: Audit raw ONNX model prediction outputs
                    self._audit_predictions(
                        preds, f"Raw ONNX fold {parity_idx}", channel, mass
                    )

                    all_predictions[app_mask] = preds

                # AUDIT STEP 3: Audit ensemble probabilities before logit transformation
                self._audit_predictions(
                    all_predictions, "Ensembled Probabilities", channel, mass
                )

                # The models emit a softmax over the classes, so every event must
                # carry exactly one fold's worth of probability. A row summing to
                # 0 means an event was scored by no fold; anything else means the
                # ensemble is not the per-event probability it is treated as below.
                prob_sums = all_predictions.sum(axis=1)
                if not np.allclose(prob_sums, 1.0, atol=1e-4):
                    bad = np.flatnonzero(~np.isclose(prob_sums, 1.0, atol=1e-4))
                    raise RuntimeError(
                        f"[pDNNProducer] Channel '{channel}', mass {mass}: {bad.size} events have "
                        f"class probabilities summing to something other than 1 "
                        f"(e.g. rows {bad[:5].tolist()} -> {prob_sums[bad[:5]].tolist()}). "
                        "The fold ensemble is incomplete or the models are not emitting probabilities."
                    )

                # Transform raw probabilities to logit scores (numerically safe subtraction)
                probs_clipped = np.clip(all_predictions, 1e-7, 1.0 - 1e-7)
                logits = np.log(probs_clipped) - np.log(1.0 - probs_clipped)

                # AUDIT STEP 4: Audit final logit values
                self._audit_predictions(logits, "Logit Calculation", channel, mass)

                mass_str = f"{mass:.0f}" if mass.is_integer() else f"{mass}"
                for class_idx, class_name in enumerate(class_names):
                    field_name = f"{channel}_M{mass_str}_{class_name}"
                    output_fields[field_name] = logits[:, class_idx].copy()

        # Attach intermediate outputs to array
        for field_name, values in output_fields.items():
            branches[field_name] = values

        gc.collect()
        return branches

    def selectDNN(self, branches: ak.Array) -> ak.Array:
        """
        Combines topology and lepton channel predictions into unified branches (e.g., M300_Signal).
        """
        nEvents = len(branches)
        output_fields = {}

        sl_mask = (
            np.asarray(branches.SL, dtype=bool)
            if "SL" in branches.fields
            else np.zeros(nEvents, dtype=bool)
        )
        boosted_mask = (
            np.asarray(branches.boosted, dtype=bool)
            if "boosted" in branches.fields
            else np.zeros(nEvents, dtype=bool)
        )

        for mass in self.target_masses:
            mass_str = f"{mass:.0f}" if mass.is_integer() else f"{mass}"
            for class_name in self.classes_to_save:
                target_field = f"M{mass_str}_{class_name}"

                sl_boosted = (
                    np.asarray(branches[f"SL_Boosted_{target_field}"])
                    if f"SL_Boosted_{target_field}" in branches.fields
                    else np.zeros(nEvents, dtype=np.float32)
                )
                sl_resolved = (
                    np.asarray(branches[f"SL_Resolved_{target_field}"])
                    if f"SL_Resolved_{target_field}" in branches.fields
                    else np.zeros(nEvents, dtype=np.float32)
                )
                dl_boosted = (
                    np.asarray(branches[f"DL_Boosted_{target_field}"])
                    if f"DL_Boosted_{target_field}" in branches.fields
                    else np.zeros(nEvents, dtype=np.float32)
                )
                dl_resolved = (
                    np.asarray(branches[f"DL_Resolved_{target_field}"])
                    if f"DL_Resolved_{target_field}" in branches.fields
                    else np.zeros(nEvents, dtype=np.float32)
                )

                sl_sel = np.where(boosted_mask, sl_boosted, sl_resolved)
                dl_sel = np.where(boosted_mask, dl_boosted, dl_resolved)

                output_fields[target_field] = np.where(sl_mask, sl_sel, dl_sel)

        for field_name, values in output_fields.items():
            branches[field_name] = values

        return branches
