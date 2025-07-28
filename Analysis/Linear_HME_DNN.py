from __future__ import annotations

class LinHMEDNN:
    def __init__(self, cfg, payload_name):

        self.cfg = cfg
        self.payload_name = payload_name

        self.hme_bins = "-10, 250, 280, 300, 350, 400, 500, 600, 800, 10000"
        self.n_bins = 9
        
    def run(self, dfw):
        dfw.Define("linear_hme_dnn", f"""
        const Double_t bins[{self.n_bins+1}] = {{{self.hme_bins}}};
        TAxis *axis = new TAxis({self.n_bins}, bins);
        return axis->FindBin(DoubleLepHME_mass) + DNN_M0_Signal;
        """)
        for col in self.cfg['columns']:
            dfw.DefineAndAppend(f"{self.payload_name}_{col}", f"return {col};")
        return dfw

