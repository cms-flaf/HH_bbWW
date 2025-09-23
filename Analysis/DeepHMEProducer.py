import numpy as np
import Analysis.hh_bbww as analysis
import awkward as ak
from DeepHME import DeepHME

class DeepHMEProducer:
    def __init__(self, cfg, payload_name):
        self.cfg = cfg
        self.payload_name = payload_name

        self.estimator = DeepHME(model_name=cfg['model_name'], 
                                 channel=cfg['channel'],
                                 return_errors=cfg['return_errors'])
    
    def prepare_dfw(self, dfw):
        dfw.df = analysis.defineAllP4(dfw.df)
        return dfw

    def run(self, dfw):
        branches_to_load = ['centralJet_pt', 'centralJet_eta', 'centralJet_phi', 'centralJet_mass',
                            'centralJet_btagPNetB', 'centralJet_btagPNetCvB', 'centralJet_btagPNetCvL', 'centralJet_btagPNetCvNotB',
                            'centralJet_btagPNetQvG', 'centralJet_PNetRegPtRawCorr', 'centralJet_PNetRegPtRawCorrNeutrino', 'centralJet_PNetRegPtRawRes',
                            'SelectedFatJet_pt', 'SelectedFatJet_eta', 'SelectedFatJet_phi', 'SelectedFatJet_mass',
                            'SelectedFatJet_particleNet_QCD', 'SelectedFatJet_particleNet_XbbVsQCD', 'SelectedFatJet_particleNetWithMass_QCD', 'SelectedFatJet_particleNetWithMass_HbbvsQCD', 'SelectedFatJet_particleNet_massCorr',
                            'lep1_pt', 'lep1_eta', 'lep1_phi', 'lep1_mass',
                            'lep2_pt', 'lep2_eta', 'lep2_phi', 'lep2_mass',
                            'met_pt', 'met_phi',
                            'event']
        branches = ak.from_rdataframe(dfw.df, branches_to_load)

        mass = None
        errors = None
        if self.cfg['return_errors']:
            mass, errors = self.estimator.predict(event_id=branches['event'],
                                                lep1_pt=branches['lep1_pt'], 
                                                lep1_eta=branches['lep1_eta'], 
                                                lep1_phi=branches['lep1_phi'], 
                                                lep1_mass=branches['lep1_mass'],
                                                lep2_pt=branches['lep2_pt'], 
                                                lep2_eta=branches['lep2_eta'], 
                                                lep2_phi=branches['lep2_phi'], 
                                                lep2_mass=branches['lep2_mass'],
                                                met_pt=branches['met_pt'], 
                                                met_phi=branches['met_phi'],
                                                jet_pt=branches['centralJet_pt'], 
                                                jet_eta=branches['centralJet_eta'], 
                                                jet_phi=branches['centralJet_phi'], 
                                                jet_mass=branches['centralJet_mass'], 
                                                jet_btagPNetB=branches['centralJet_btagPNetB'], 
                                                jet_btagPNetCvB=branches['centralJet_btagPNetCvB'], 
                                                jet_btagPNetCvL=branches['centralJet_btagPNetCvL'], 
                                                jet_btagPNetCvNotB=branches['centralJet_btagPNetCvNotB'], 
                                                jet_btagPNetQvG=branches['centralJet_btagPNetQvG'],
                                                jet_PNetRegPtRawCorr=branches['centralJet_PNetRegPtRawCorr'], 
                                                jet_PNetRegPtRawCorrNeutrino=branches['centralJet_PNetRegPtRawCorrNeutrino'], 
                                                jet_PNetRegPtRawRes=branches['centralJet_PNetRegPtRawRes'],
                                                fatjet_pt=branches['SelectedFatJet_pt'], 
                                                fatjet_eta=branches['SelectedFatJet_eta'], 
                                                fatjet_phi=branches['SelectedFatJet_phi'], 
                                                fatjet_mass=branches['SelectedFatJet_mass'],
                                                fatjet_particleNet_QCD=branches['SelectedFatJet_particleNet_QCD'], 
                                                fatjet_particleNet_XbbVsQCD=branches['SelectedFatJet_particleNet_XbbVsQCD'], 
                                                fatjet_particleNetWithMass_QCD=branches['SelectedFatJet_particleNetWithMass_QCD'], 
                                                fatjet_particleNetWithMass_HbbvsQCD=branches['SelectedFatJet_particleNetWithMass_HbbvsQCD'], 
                                                fatjet_particleNet_massCorr=branches['SelectedFatJet_particleNet_massCorr'],
                                                output_format='mass')
        else:
            mass = self.estimator.predict(event_id=branches['event'],
                                        lep1_pt=branches['lep1_pt'], 
                                        lep1_eta=branches['lep1_eta'], 
                                        lep1_phi=branches['lep1_phi'], 
                                        lep1_mass=branches['lep1_mass'],
                                        lep2_pt=branches['lep2_pt'], 
                                        lep2_eta=branches['lep2_eta'], 
                                        lep2_phi=branches['lep2_phi'], 
                                        lep2_mass=branches['lep2_mass'],
                                        met_pt=branches['met_pt'], 
                                        met_phi=branches['met_phi'],
                                        jet_pt=branches['centralJet_pt'], 
                                        jet_eta=branches['centralJet_eta'], 
                                        jet_phi=branches['centralJet_phi'], 
                                        jet_mass=branches['centralJet_mass'], 
                                        jet_btagPNetB=branches['centralJet_btagPNetB'], 
                                        jet_btagPNetCvB=branches['centralJet_btagPNetCvB'], 
                                        jet_btagPNetCvL=branches['centralJet_btagPNetCvL'], 
                                        jet_btagPNetCvNotB=branches['centralJet_btagPNetCvNotB'], 
                                        jet_btagPNetQvG=branches['centralJet_btagPNetQvG'],
                                        jet_PNetRegPtRawCorr=branches['centralJet_PNetRegPtRawCorr'], 
                                        jet_PNetRegPtRawCorrNeutrino=branches['centralJet_PNetRegPtRawCorrNeutrino'], 
                                        jet_PNetRegPtRawRes=branches['centralJet_PNetRegPtRawRes'],
                                        fatjet_pt=branches['SelectedFatJet_pt'], 
                                        fatjet_eta=branches['SelectedFatJet_eta'], 
                                        fatjet_phi=branches['SelectedFatJet_phi'], 
                                        fatjet_mass=branches['SelectedFatJet_mass'],
                                        fatjet_particleNet_QCD=branches['SelectedFatJet_particleNet_QCD'], 
                                        fatjet_particleNet_XbbVsQCD=branches['SelectedFatJet_particleNet_XbbVsQCD'], 
                                        fatjet_particleNetWithMass_QCD=branches['SelectedFatJet_particleNetWithMass_QCD'], 
                                        fatjet_particleNetWithMass_HbbvsQCD=branches['SelectedFatJet_particleNetWithMass_HbbvsQCD'], 
                                        fatjet_particleNet_massCorr=branches['SelectedFatJet_particleNet_massCorr'],
                                        output_format='mass')     
    
        dfw.DefineAndAppend('deepHME_mass', mass)
        if 'deepHME_mass_error' in self.cfg['columns']:
            dfw.DefineAndAppend('deepHME_mass', errors)
        return dfw