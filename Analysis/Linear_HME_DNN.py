from __future__ import annotations

class LinHMEDNN:
    def __init__(self, cfg, payload_name):

        self.cfg = cfg
        self.payload_name = payload_name

        self.hme_bins = [-10, 250, 280, 300, 350, 400, 500, 600, 800, 10000]
        self.n_bins = len(self.hme_bins) - 1
        self.hme_bins_string = ','.join(map(str, self.hme_bins))
        
    def run(self, dfw):
        for col in self.cfg['columns']:
            print(col)
            mass = col.split('_')[-1][1:]
            print(mass)
            dfw.Define(f"linear_hme_dnn_m{mass}", f"""
            static const Double_t bins[{self.n_bins+1}] = {{{self.hme_bins_string}}};
            static TAxis axis({self.n_bins}, bins);
            return axis.FindBin(DoubleLepHME_mass) + DNNParametric_M{mass}_Signal;
            """)
        for col in self.cfg['columns']:
            dfw.DefineAndAppend(f"{self.payload_name}_{col}", f"return {col};")
        return dfw

