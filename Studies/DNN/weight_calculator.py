import uproot
import numpy as np
import os
import awkward as ak


def create_weight_file(inName, outName, bb_low=70, bb_high=150, bb_min=70, bb_max=300):
    print(f"On file {inName}")
    in_file = uproot.open(inName)
    out_file = uproot.recreate(outName)

    tree = in_file["Events"]
    branches_to_load = [
        "sample_type",
        "bb_mass",
        "bb_mass_PNetRegPtRawCorr",
        "bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino",
        "X_mass",
        "centralJet_hadronFlavour",
        "SelectedFatJet_hadronFlavour",
        "weight_MC_Lumi_pu",
    ]
    branches = tree.arrays(branches_to_load)

    sample_type = branches["sample_type"]
    bb_mass = branches["bb_mass"]
    bb_mass = branches["bb_mass_PNetRegPtRawCorr"]
    bb_mass = branches["bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino"]

    X_mass = branches["X_mass"]

    hadronFlavour = ak.fill_none(
        ak.pad_none(branches["centralJet_hadronFlavour"], 2, axis=1), 0
    )
    ak8_hadronFlavour = ak.fill_none(
        ak.pad_none(branches["SelectedFatJet_hadronFlavour"], 1, axis=1), 0
    )

    type_to_name = {
        "1": "Signal",
        "2": "Signal",
        "8": "TT",
        "5": "DY",
        "9": "ST",
    }  # 1 is Radion, 2 is Graviton
    # type_to_target = {'1': 0, '2': 0, '8': 1, '5': 2, '9': 3} # Multiclass type-to-target
    type_to_target = {"1": 0, "2": 0, "8": 1, "5": 1, "9": 1}  # Binary type-to-target
    sample_name = np.array([type_to_name[str(sample)] for sample in sample_type])
    class_targets = np.array([type_to_target[str(sample)] for sample in sample_type])

    # Initialize the two branches, class weight and adv weight
    # Starting from their genWeight (includes XS and such)
    class_weight = branches["weight_MC_Lumi_pu"]
    adv_weight = branches["weight_MC_Lumi_pu"]

    # First step, remove any sample types we want to
    # samples_to_remove = [ 'DY' ]
    samples_to_remove = []

    for sample_to_remove in samples_to_remove:
        class_weight = np.where(sample_name == sample_to_remove, 0.0, class_weight)

        adv_weight = np.where(sample_name == sample_to_remove, 0.0, adv_weight)

    # Next normalize between sample types (class)

    # First remove the signal that is not gen bb
    class_weight = np.where(
        sample_name == "Signal",
        np.where(
            ((hadronFlavour[:, 0] == 5) & (hadronFlavour[:, 1] == 5))
            | (
                ak8_hadronFlavour[:, 0] == 5
            ),  # & (X_mass == 800), # For now, only train on m450
            class_weight,
            0.0,
        ),
        class_weight,
    )

    # Total_Signal == Total_DY + Total_TT (Equal weight of signal vs background in binary)
    total_signal = np.sum(np.where(sample_name == "Signal", class_weight, 0.0))
    total_background = np.sum(np.where(sample_name != "Signal", class_weight, 0.0))

    norm_factor = total_signal / total_background
    class_weight = np.where(
        sample_name == "Signal", class_weight, class_weight * norm_factor
    )

    # Next normalize between m_bb regions (adversarial)
    # TT_Low == TT_Mid == TT_High
    # DY_Low == DY_Mid == DY_High

    # TT_Total / DY_Total == TT_yield / DY_yield
    adv_weight = np.where(sample_name == "Signal", 0.0, adv_weight)
    # bb_low = 70
    # bb_high = 150

    # Set adv targets
    adv_targets = np.where(bb_mass < bb_low, -1, np.where(bb_mass < bb_high, 0, 1))

    # Option to set an lower and upper
    # bb_min = 70
    # bb_max = 300
    adv_weight = np.where(bb_mass > bb_min, adv_weight, 0.0)
    adv_weight = np.where(bb_mass < bb_max, adv_weight, 0.0)

    for this_name in np.unique(sample_name):
        if this_name == "Signal":
            continue
        print(f"On sample {this_name}")
        total_low = np.sum(
            np.where((sample_name == this_name) & (bb_mass < bb_low), adv_weight, 0.0)
        )
        total_mid = np.sum(
            np.where(
                (sample_name == this_name) & (bb_mass > bb_low) & (bb_mass < bb_high),
                adv_weight,
                0.0,
            )
        )
        total_high = np.sum(
            np.where((sample_name == this_name) & (bb_mass > bb_high), adv_weight, 0.0)
        )
        # norm to mid
        adv_weight = np.where(
            (sample_name == this_name) & (bb_mass < bb_low),
            # total_mid * adv_weight / total_low,
            0.0,  # For now, we will just ignore the down category
            adv_weight,
        )
        adv_weight = np.where(
            (sample_name == this_name) & (bb_mass > bb_high),
            total_mid * adv_weight / total_high,
            adv_weight,
        )

        total_scaled = np.sum(np.where(sample_name == this_name, adv_weight, 0.0))
        adv_weight = np.where(
            (sample_name == this_name), adv_weight / total_scaled, adv_weight
        )

    # Nan to num for any divide by 0 errors
    class_weight = np.nan_to_num(class_weight, 0.0)
    adv_weight = np.nan_to_num(adv_weight, 0.0)

    # Normalize both class weights and adv weights to nEvents
    print(
        f"Before normalization our class total {np.sum(class_weight)} and adv total {np.sum(adv_weight)}"
    )
    nEvents = len(class_weight)
    class_weight = (nEvents / np.sum(class_weight)) * class_weight
    adv_weight = (nEvents / np.sum(adv_weight)) * adv_weight
    print(
        f"After normalization our class total {np.sum(class_weight)} and adv total {np.sum(adv_weight)}"
    )

    out_dict = {
        "class_weight": class_weight,
        "adv_weight": adv_weight,
        "class_target": class_targets,
        "adv_target": adv_targets,
    }

    out_file["weight_tree"] = out_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create Weight Files for DNN.")
    parser.add_argument(
        "--inDir", required=True, type=str, help="Input Directory with batch files."
    )

    args = parser.parse_args()

    inDir = args.inDir

    # fname = "/afs/cern.ch/work/d/daebi/diHiggs/HH_bbWW/Studies/DNN/DNN_Datasets/Dataset_2025-03-28-12-49-16/batchfile{nParity}.root"
    # outname = "/afs/cern.ch/work/d/daebi/diHiggs/HH_bbWW/Studies/DNN/DNN_Datasets/Dataset_2025-03-28-12-49-16/weightfile{nParity}.root"

    batchfiles = [x for x in os.listdir(inDir) if "batchfile" in x]

    bb_low = 70
    bb_high = 150

    bb_min = 70
    bb_max = 300

    for batchfile_name in batchfiles:
        weightfile_name = f"weightfile{batchfile_name[-6:]}"

        in_file = os.path.join(inDir, batchfile_name)
        out_file = os.path.join(inDir, weightfile_name)

        print(f"Starting infile {in_file} and making outfile {out_file}")
        create_weight_file(in_file, out_file, bb_low, bb_high, bb_min, bb_max)
