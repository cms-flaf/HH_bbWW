import numpy as np
import Analysis.hh_bbww as analysis

class DeepHMEProducer:
    def __init__(self, cfg, payload_name):
        self.cfg = cfg
        self.payload_name = payload_name
    
    def prepare_dfw(self, dfw):
        dfw.df = analysis.defineAllP4(dfw.df)
        dfw.df = analysis.AddDeepHMEVariables(dfw.df)
        return dfw

    def run(self):
        pass
    