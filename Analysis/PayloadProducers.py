from Studies.HME.new.hmeVariables import GetHMEVariables

class HMEProducer:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, dfw):
        channel_name = None
        if "ncentralJet" not in dfw.df.GetColumnNames():
            dfw.Define("ncentralJet", "return centralJet_pt.size();")
        if self.cfg['channel'] == "DL":
            channel_name = 'DL'
            dfw.Define("has_necessary_inputs", "ncentralJet >= 2 && lep1_pt > 0.0 && lep2_pt > 0.0")
        elif self.cfg['channel'] == "SL":
            channel_name = 'SL'
            dfw.Define("has_necessary_inputs", "ncentralJet >= 4 && lep1_pt > 0.0")
        else:
            raise RuntimeError(f'Passed illegal option to HMEProducer config: channel {self.cfg['channel']}')
        
        dfw.df = GetHMEVariables(dfw.df, self.cfg['channel'])
        for col in self.cfg['columns']:
            if col != 'valid':
                dfw.DefineAndAppend(f"HME_{channel_name}_{col}", f"return hme_output[static_cast<size_t>(HME::EstimOut::{col})];")
        if 'valid' in self.cfg['columns']:
            dfw.DefineAndAppend(f"HME_{channel_name}_valid", "return HME_mass > 0.0;")
        return dfw

class DNNProducer:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, dfw):
        print("Running DNN producer")
        return dfw