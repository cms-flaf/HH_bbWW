from __future__ import annotations


class HMEinDNNBins:
    def __init__(self, cfg, payload_name):

        self.cfg = cfg
        self.payload_name = payload_name

        self.mass_list = cfg["masses"]
        self.bins = cfg["bins"]

    def run(self, dfw):
        for mass in self.mass_list:
            for binset in self.bins:
                SL_HME_var = "SingleLep_DeepHME_mass"
                SL_DNN_var = f"DNNParametric_SL_NoHME_M{mass}_Signal"
                DL_HME_var = "DoubleLep_DeepHME_mass"
                DL_DNN_var = f"DNNParametric_DL_NoHME_M{mass}_Signal"
                low_DNN = ".".join(binset[0].split("p"))
                high_DNN = ".".join(binset[1].split("p"))

                colName_SL = f"{self.payload_name}_HME_mass_SL_DNN_M{mass}_{binset[0]}_{binset[1]}"
                colName_DL = f"{self.payload_name}_HME_mass_DL_DNN_M{mass}_{binset[0]}_{binset[1]}"

                dfw.DefineAndAppend(
                    f"{colName_SL}",
                    f"({SL_DNN_var} >= {low_DNN} && {SL_DNN_var} <= {high_DNN}) ? {SL_HME_var} : -100.f",
                )

                dfw.DefineAndAppend(
                    f"{colName_DL}",
                    f"({DL_DNN_var} >= {low_DNN} && {DL_DNN_var} <= {high_DNN}) ? {DL_HME_var} : -100.f",
                )

        return dfw
