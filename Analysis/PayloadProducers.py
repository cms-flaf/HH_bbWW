from Studies.HME.new.hmeVariables import GetHMEVariables

class HMEProducer:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, dfw):
        if "ncentralJet" not in dfw.df.GetColumnNames():
            dfw.Define("ncentralJet", "return centralJet_pt.size();")

        ch = self.cfg['channel']
        if ch == "DL":
            dfw.Define("has_necessary_inputs", "ncentralJet >= 2 && lep1_pt > 0.0 && lep2_pt > 0.0")
        elif ch == "SL":
            dfw.Define("has_necessary_inputs", "ncentralJet >= 4 && lep1_pt > 0.0")
        else:
            raise RuntimeError(f"Illegal channel in config: {ch}")

        dfw.df = GetHMEVariables(dfw.df, ch)
        for col in self.cfg['columns']:
            if col != 'valid':
                dfw.DefineAndAppend(f"HME_{ch}_{col}", f"return hme_output[static_cast<size_t>(HME::EstimOut::{col})];")
        if 'valid' in self.cfg['columns']:
            dfw.DefineAndAppend(f"HME_{ch}_valid", f"return HME_{ch}_mass > 0.0;")
        return dfw

class DNNProducer:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, dfw):
        print("Running DNN producer")
        return dfw