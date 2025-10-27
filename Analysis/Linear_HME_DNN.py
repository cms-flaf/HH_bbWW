from __future__ import annotations


class LinHMEDNN:
    def __init__(self, cfg, payload_name):

        self.cfg = cfg
        self.payload_name = payload_name

        self.hme_bins = [
            -10,
            225,
            275,
            325,
            375,
            425,
            525,
            675,
            850,
            1250,
            1750,
            2250,
            3250,
            10000,
        ]
        self.n_bins = len(self.hme_bins) - 1
        self.hme_bins_string = ",".join(map(str, self.hme_bins))

        self.channel = cfg["channel"]

    def run(self, dfw):
        for col in self.cfg["columns"]:
            mass = col.split("_")[-1][1:]
            HME_var = (
                "DoubleLep_DeepHME_mass"
                if self.channel == "DL"
                else "SingleLep_DeepHME_mass"
            )
            DNN_var = (
                f"DNNParametric_DL_NoHME_M{mass}_Signal"
                if self.channel == "DL"
                else f"DNNParametric_SL_NoHME_M{mass}_Signal"
            )

            dfw.Define(
                f"linear_hme_dnn_m{mass}",
                f"""
            static const Double_t bins[{self.n_bins+1}] = {{{self.hme_bins_string}}};
            static const TAxis axis({self.n_bins}, bins);
            return axis.FindFixBin({HME_var}) + {DNN_var};
            """,
            )
        for col in self.cfg["columns"]:
            dfw.DefineAndAppend(f"{self.payload_name}_{col}", f"return {col};")
        return dfw
