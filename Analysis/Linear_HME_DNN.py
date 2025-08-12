from __future__ import annotations


class LinHMEDNN:
    def __init__(self, cfg, payload_name):

        self.cfg = cfg
        self.payload_name = payload_name

        self.hme_bins = [
            -10,
            200,
            225,
            275,
            325,
            375,
            425,
            475,
            525,
            575,
            625,
            675,
            750,
            850,
            950,
            1100,
            1300,
            1500,
            1700,
            1900,
            2250,
            2750,
            3500,
            4500,
            10000,
        ]
        self.n_bins = len(self.hme_bins) - 1
        self.hme_bins_string = ",".join(map(str, self.hme_bins))

    def run(self, dfw):
        for col in self.cfg["columns"]:
            mass = col.split("_")[-1][1:]
            dfw.Define(
                f"linear_hme_dnn_m{mass}",
                f"""
            static const Double_t bins[{self.n_bins+1}] = {{{self.hme_bins_string}}};
            static const TAxis axis({self.n_bins}, bins);
            return axis.FindFixBin(DoubleLepHME_mass) + DNNParametric_M{mass}_Signal;
            """,
            )
        for col in self.cfg["columns"]:
            dfw.DefineAndAppend(f"{self.payload_name}_{col}", f"return {col};")
        return dfw
