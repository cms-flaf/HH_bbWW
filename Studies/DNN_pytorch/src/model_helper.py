#!/usr/bin/env python3
import uproot
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# ==============================================================================
# 1. Parity Expression & Feature Calculation Helpers
# ==============================================================================
def eval_parity_expr(expr_str, n_parity_val):
    """
    Evaluates parity string expressions from configuration file (e.g. '(nParity + 1) % 4')
    by injecting 'nParity' into context.
    """
    return eval(expr_str, {"nParity": n_parity_val, "np": np})


def delta_r(eta1, phi1, eta2, phi2):
    dphi = phi1 - phi2
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)


def compute_static_features(df):
    safe_met = np.clip(df["PuppiMET_pt"], 1e-3, None)
    safe_ht = np.clip(df["HT"], 1.0, None)

    dr_l1_b1 = delta_r(df["lep1_eta"], df["lep1_phi"], df["bjet1_eta"], df["bjet1_phi"])
    dr_l1_b2 = delta_r(df["lep1_eta"], df["lep1_phi"], df["bjet2_eta"], df["bjet2_phi"])
    dr_l2_b1 = delta_r(df["lep2_eta"], df["lep2_phi"], df["bjet1_eta"], df["bjet1_phi"])
    dr_l2_b2 = delta_r(df["lep2_eta"], df["lep2_phi"], df["bjet2_eta"], df["bjet2_phi"])

    df["min_dR_l_b"] = np.minimum(
        np.minimum(dr_l1_b1, dr_l1_b2), np.minimum(dr_l2_b1, dr_l2_b2)
    )
    df["lep_b_centrality"] = (df["lep1_pt"] + df["lep2_pt"]) / np.clip(
        df["bjet1_pt"] + df["bjet2_pt"] + df["PuppiMET_pt"], 1.0, None
    )
    df["MET_over_sqrt_HT"] = df["PuppiMET_pt"] / np.sqrt(safe_ht)
    df["HT_total"] = df["HT"] + df["PuppiMET_pt"] + df["lep1_pt"] + df["lep2_pt"]

    if "bjet1_btagPNetB" in df.columns:
        df["bb_btag_product"] = df["bjet1_btagPNetB"] * df["bjet2_btagPNetB"]
    if "bb_pt" in df.columns:
        df["pt_ratio_ll_bb"] = (df["ll_pt"]) / np.clip(df["bb_pt"], 1.0, None)

    mt_l1_met = np.sqrt(
        2 * df["lep1_pt"] * safe_met * (1 - np.cos(df["lep1_phi"] - df["PuppiMET_phi"]))
    )
    mt_l2_met = np.sqrt(
        2 * df["lep2_pt"] * safe_met * (1 - np.cos(df["lep2_phi"] - df["PuppiMET_phi"]))
    )
    df["min_MT_lep_met"] = np.minimum(mt_l1_met, mt_l2_met)

    df["cos_theta_star_ll"] = np.tanh((df["lep1_eta"] - df["lep2_eta"]) / 2.0)

    dphi_l1_met = np.abs(
        (df["lep1_phi"] - df["PuppiMET_phi"] + np.pi) % (2 * np.pi) - np.pi
    )
    dphi_l2_met = np.abs(
        (df["lep2_phi"] - df["PuppiMET_phi"] + np.pi) % (2 * np.pi) - np.pi
    )
    df["min_dphi_l_met"] = np.minimum(dphi_l1_met, dphi_l2_met)

    px_ll = df["lep1_pt"] * np.cos(df["lep1_phi"]) + df["lep2_pt"] * np.cos(
        df["lep2_phi"]
    )
    py_ll = df["lep1_pt"] * np.sin(df["lep1_phi"]) + df["lep2_pt"] * np.sin(
        df["lep2_phi"]
    )

    pt_ll = np.sqrt(px_ll**2 + py_ll**2)
    m_ll = df["ll_mass"]
    e_ll = np.sqrt(pt_ll**2 + m_ll**2)

    df["m_ll_met_transverse"] = np.sqrt(
        np.clip(
            (e_ll + safe_met) ** 2
            - (px_ll + safe_met * np.cos(df["PuppiMET_phi"])) ** 2
            - (py_ll + safe_met * np.sin(df["PuppiMET_phi"])) ** 2,
            0.0,
            None,
        )
    )

    dphi_leptons = df["lep1_phi"] - df["lep2_phi"]
    df["ll_dphi"] = np.abs((dphi_leptons + np.pi) % (2 * np.pi) - np.pi)
    phi_ll = np.arctan2(py_ll, px_ll)
    df["met_ll_dphi"] = np.abs(
        (phi_ll - df["PuppiMET_phi"] + np.pi) % (2 * np.pi) - np.pi
    )
    return df


def update_mass_dependent_features(df, features):
    """
    Dynamically creates scaled features for parametric inference.
    If a required feature contains '_scaled', its base counterpart is multiplied by mX.
    """
    mX = df["X_mass"].values
    for feature in features:
        if "_scaled" in feature:
            base_feature = feature.replace("_scaled", "")
            if base_feature in df.columns:
                df[feature] = df[base_feature].values * mX
            else:
                print(
                    f"[WARNING] Base feature '{base_feature}' for '{feature}' not found in DataFrame!"
                )
    return df


# ==============================================================================
# 2. Data Loader Functions
# ==============================================================================
def load_parametric_fold(
    data_path,
    weight_path,
    feature_map,
    extra_vars,
    mass_points,
    seed=42,
    dedicated_mass=None,
):
    computed_static_features = [
        "min_dR_l_b",
        "lep_b_centrality",
        "MET_over_sqrt_HT",
        "HT_total",
        "bb_btag_product",
        "min_MT_lep_met",
        "cos_theta_star_ll",
        "min_dphi_l_met",
        "pt_ratio_ll_bb",
        "m_ll_met_transverse",
        "ll_dphi",
        "met_ll_dphi",
    ]
    all_possible_features = list(set(feature_map["resolved"] + feature_map["boosted"]))

    # Exclude calculated static features and dynamic '_scaled' features from ROOT branch loading
    parent_features = [
        f
        for f in all_possible_features
        # if f not in computed_static_features and "_scaled" not in f
        if "_scaled" not in f
    ]
    load_branches = list(set(parent_features + extra_vars + ["weight_Central"]))

    with uproot.open(data_path) as f_data:
        available = [b for b in load_branches if b in f_data["Events"]]
        df_feats = f_data["Events"].arrays(available, library="pd")
    with uproot.open(weight_path) as f_weight:
        df_wghts = f_weight["weight_tree"].arrays(
            ["class_targets_binary", "class_weight"], library="pd"
        )

    df_merged = pd.concat([df_feats, df_wghts], axis=1)
    df_merged["class_targets_binary"] = 1 - df_merged["class_targets_binary"]

    is_bkg = df_merged["class_targets_binary"] == 0
    df_sig = df_merged[~is_bkg].copy()
    df_bkg = df_merged[is_bkg].copy()

    if dedicated_mass is not None:
        df_sig = df_sig[df_sig["X_mass"] == dedicated_mass].copy()
        df_bkg["X_mass"] = dedicated_mass
    else:
        np.random.seed(seed)
        random_masses = np.random.choice(mass_points, size=len(df_bkg))
        df_bkg["X_mass"] = random_masses.astype(np.int32)

    df_final = pd.concat([df_sig, df_bkg], ignore_index=True)
    df_final["X_mass"] = df_final["X_mass"].astype(df_merged["X_mass"].dtype)

    # df_final = compute_static_features(df_final)
    df_final = update_mass_dependent_features(df_final, all_possible_features)

    df_final["class_targets_multiclass"] = np.where(
        df_final["class_targets_binary"] == 1,
        0,
        np.where(df_final["class_value"] == 1, 1, 2),
    )
    return df_final


def load_physical_fold(
    data_path, weight_path, feature_map, extra_vars, dedicated_mass=None
):
    computed_static_features = [
        "min_dR_l_b",
        "lep_b_centrality",
        "MET_over_sqrt_HT",
        "HT_total",
        "bb_btag_product",
        "min_MT_lep_met",
        "cos_theta_star_ll",
        "min_dphi_l_met",
        "pt_ratio_ll_bb",
        "m_ll_met_transverse",
        "ll_dphi",
        "met_ll_dphi",
    ]
    all_possible_features = list(set(feature_map["resolved"] + feature_map["boosted"]))

    parent_features = [
        f
        for f in all_possible_features
        # if f not in computed_static_features and "_scaled" not in f
        if "_scaled" not in f
    ]
    load_branches = list(set(parent_features + extra_vars + ["weight_Central"]))

    with uproot.open(data_path) as f_data:
        available = [b for b in load_branches if b in f_data["Events"]]
        df_merged = f_data["Events"].arrays(available, library="pd")
    with uproot.open(weight_path) as f_weight:
        df_wghts = f_weight["weight_tree"].arrays(
            ["class_targets_binary", "class_weight"], library="pd"
        )

    df_merged = pd.concat([df_merged, df_wghts], axis=1)
    df_merged["class_targets_binary"] = 1 - df_merged["class_targets_binary"]

    if dedicated_mass is not None:
        is_bkg = df_merged["class_targets_binary"] == 0
        is_target_sig = (df_merged["class_targets_binary"] == 1) & (
            df_merged["X_mass"] == dedicated_mass
        )
        df_merged = df_merged[is_bkg | is_target_sig].copy()

    # df_merged = compute_static_features(df_merged)
    df_merged = update_mass_dependent_features(df_merged, all_possible_features)

    df_merged["class_targets_multiclass"] = np.where(
        df_merged["class_targets_binary"] == 1,
        0,
        np.where(df_merged["class_value"] == 1, 1, 2),
    )
    return df_merged


# ==============================================================================
# 3. Dynamic Mass State Manager & PyTorch Network Classes
# ==============================================================================
class DynamicMassManager:
    def __init__(self, df, features, scaler, mass_points, is_training=True):
        self.df = df.copy().reset_index(drop=True)
        self.features = features
        self.scaler = scaler
        self.mass_points = mass_points
        self.is_training = is_training

        self.bkg_mask = (self.df["class_targets_multiclass"] != 0).values
        self.num_bkg = self.bkg_mask.sum()

    def randomize_background_masses(self):
        if self.is_training and self.num_bkg > 0:
            new_masses = np.random.choice(self.mass_points, size=self.num_bkg)
            self.df.loc[self.bkg_mask, "X_mass"] = new_masses.astype(np.int32)
            self.df = update_mass_dependent_features(self.df, self.features)

    def get_scaled_tensors(self):
        X_scaled = self.scaler.transform(self.df[self.features])
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(
            self.df["class_targets_multiclass"].values, dtype=torch.long
        )
        w_tensor = torch.tensor(self.df["class_weight"].values, dtype=torch.float32)
        return X_tensor, y_tensor, w_tensor


class ParametricDNN(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=3,
        hidden_dims=[512, 256, 128, 64],
        dropout_rate=0.35,
    ):
        super(ParametricDNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        return self.output_layer(self.network(x))


class ONNXDeployWrapper(nn.Module):
    """Wraps PyTorch model with embedded StandardScaler parameters for standalone ONNX deployment."""

    def __init__(self, trained_model, scaler):
        super(ONNXDeployWrapper, self).__init__()
        self.model = trained_model
        self.register_buffer(
            "mean", torch.tensor(scaler.mean_, dtype=torch.float32).unsqueeze(0)
        )
        self.register_buffer(
            "scale", torch.tensor(scaler.scale_, dtype=torch.float32).unsqueeze(0)
        )

    def forward(self, x):
        x_scaled = (x - self.mean) / self.scale
        logits = self.model(x_scaled)
        return torch.softmax(logits, dim=1)


class WeightedStandardCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightedStandardCrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets, weights):
        loss = self.ce(logits, targets)
        return (loss * weights).mean()
