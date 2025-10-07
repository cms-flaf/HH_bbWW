import os, sys, gc, psutil
import uproot
import numpy as np
from datetime import datetime
import yaml
import awkward as ak
from tqdm import tqdm

import ROOT
import Analysis.hh_bbww as analysis

ROOT.gROOT.SetBatch(True)
ROOT.EnableThreadSafety()
# ROOT.EnableImplicitMT(4)

sys.path.append(os.environ['ANALYSIS_PATH'])
ana_path = os.environ['ANALYSIS_PATH']

header_path_AnalysisTools = "FLAF/include/AnalysisTools.h"
ROOT.gInterpreter.Declare(f'#include "{os.path.join(ana_path,header_path_AnalysisTools)}"')
header_path_AnalysisMath = "FLAF/include/AnalysisMath.h"
ROOT.gInterpreter.Declare(f'#include "{os.path.join(ana_path,header_path_AnalysisMath)}"')
header_path_MT2 = "FLAF/include/MT2.h"
ROOT.gInterpreter.Declare(f'#include "{os.path.join(ana_path,header_path_MT2)}"')
header_path_Lester_mt2_bisect = "FLAF/include/Lester_mt2_bisect.cpp"
ROOT.gInterpreter.Declare(f'#include "{os.path.join(ana_path,header_path_Lester_mt2_bisect)}"')
lep1_p4 = "ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(lep1_pt,lep1_eta,lep1_phi,lep1_mass)"
lep2_p4 = "ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(lep2_pt,lep2_eta,lep2_phi,lep2_mass)"
b1_p4 = "ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(centralJet_pt[0],centralJet_eta[0],centralJet_phi[0],centralJet_mass[0])"
b2_p4 = "ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(centralJet_pt[1],centralJet_eta[1],centralJet_phi[1],centralJet_mass[1])"
MET_p4 = "ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(met_pt, 0., met_phi, 0.)"



def create_signal_files(config_dict, output_folder, era):
    storage_folder = os.path.join(config_dict['storage_folder'], era)

    for signal_name in config_dict['signal']:
        signal_dict = config_dict['signal'][signal_name]
        mass_points = signal_dict['mass_points']
        dataset_name_format = signal_dict['dataset_name_format']
        use_combined = signal_dict['use_combined']

        if use_combined:
            combined_name = signal_dict['combined_name']

            out_file_folder = os.path.join(output_folder, combined_name)
            out_file_name = f"{os.path.join(out_file_folder, combined_name)}.root"

            if os.path.exists(out_file_name):
                print(f"Combined file {out_file_name} already exists, skip")
                continue
            out_file = uproot.recreate(out_file_name)

            new_array = {}
            nEvents = 0

            print(f"Starting to merge signal {signal_name} files")

            for mass_point in tqdm(mass_points):
                dataset_name = dataset_name_format.format(mass_point)
                extension_list = [ fn for fn in os.listdir(storage_folder) if fn.startswith(f"{dataset_name}_ext") ]

                for ext_name in ([dataset_name] + extension_list):
                    process_dir = os.path.join(storage_folder, ext_name)
                    for nano_file in [x for x in os.listdir(process_dir) if x.endswith(".root")]:
                        with uproot.open(f"{os.path.join(process_dir, nano_file)}:Events") as h:
                            tree = h.arrays()
                            nEvents += h.num_entries

                            keys = tree.fields
                            for key in keys:
                                if key not in new_array.keys():
                                    new_array[key] = tree[key]
                                else:
                                    new_array[key] = ak.concatenate([new_array[key], tree[key]])


            #Shuffle the signal data
            index = np.arange(nEvents)
            np.random.shuffle(index)
            for key in new_array.keys():
                new_array[key] = new_array[key][index]

            out_file['Events'] = new_array
            out_file.close()



def create_dict(config_dict, output_folder, era):
    batch_dict = config_dict['batch_dict']
    storage_folder = os.path.join(config_dict['storage_folder'], era)
    selection_branches = config_dict['selection_branches']
    selection_cut = config_dict['selection_cut']

    if 'batch_size' not in batch_dict.keys():
        batch_dict['batch_size'] = 0
    process_dict = {}
    for key in batch_dict.keys():
        if key == 'batch_size': continue
        batch_dict['batch_size'] += batch_dict[key]
        process_dict[key] = {}

    for nParity in range(config_dict['nParity']):
        empty_dict_example_file = ""
        nParity_Cut = config_dict['parity_func'].format(nParity = config_dict['nParity'], parity_scan = nParity)
        total_cut = f"{selection_cut} & {nParity_Cut}"

        out_yaml = f"batch_config_parity{nParity}.yaml"

        if os.path.exists(os.path.join(output_folder, out_yaml)):
            print(f"YAML file {out_yaml} already exists, skip")
            continue

        print("Looping over signals in config")
        for signal_name in tqdm(config_dict['signal']):
            signal_dict = config_dict['signal'][signal_name]
            class_value = signal_dict['class_value']
            spin_value = signal_dict['spin']
            mass_points = signal_dict['mass_points']
            dataset_name_format = signal_dict['dataset_name_format']
            use_combined = signal_dict['use_combined']


            #If a combined file exists, lets use that
            #if f"{signal_dict['combined_name']}.root" in os.listdir(output_folder):
            if use_combined:
                dataset_name = signal_dict['combined_name']

                process_dict[signal_name][dataset_name] = {
                    'total': 0,
                    'total_cut': 0,
                    'weight_cut': 0,
                    'nBatches': 0,
                    'batch_size': 0,
                    'batch_start': 0,
                    'class_value': class_value,
                    'spin': spin_value,
                    'mass': -1,
                    'all_extensions': [],
                    'storage_folder': os.path.join(os.getcwd(), output_folder)
                }
                    
                process_dict[signal_name][dataset_name]['all_extensions'] = [dataset_name]

                with uproot.open(f"{os.path.join(output_folder, dataset_name, dataset_name)}.root:Events") as h:
                    tree = h.arrays(selection_branches)
                    process_dict[signal_name][dataset_name]['total'] += int(h.num_entries)
                    process_dict[signal_name][dataset_name]['total_cut'] += int(np.sum(eval(total_cut)))
                    eval_string = f"float(np.sum(tree[{total_cut}].weight_MC_Lumi_pu))"
                    process_dict[signal_name][dataset_name]['weight_cut'] += eval(eval_string)



            for mass_point in mass_points:
                dataset_name = dataset_name_format.format(mass_point)

                process_dict[signal_name][dataset_name] = {
                    'total': 0,
                    'total_cut': 0,
                    'weight_cut': 0,
                    'nBatches': 0,
                    'batch_size': 0,
                    'batch_start': 0,
                    'class_value': class_value,
                    'spin': spin_value,
                    'mass': mass_point,
                    'all_extensions': [],
                    'storage_folder': storage_folder,
                }

                extension_list = [ fn for fn in os.listdir(storage_folder) if fn.startswith(f"{dataset_name}_ext") ]
                process_dict[signal_name][dataset_name]['all_extensions'] = [dataset_name] + extension_list

                for ext_name in process_dict[signal_name][dataset_name]['all_extensions']:
                    process_dir = os.path.join(storage_folder, ext_name)
                    for nano_file in [x for x in os.listdir(process_dir) if x.endswith(".root")]:
                        with uproot.open(f"{os.path.join(process_dir, nano_file)}:Events") as h:
                            tree = h.arrays(selection_branches)
                            process_dict[signal_name][dataset_name]['total'] += int(h.num_entries)
                            process_dict[signal_name][dataset_name]['total_cut'] += int(np.sum(eval(total_cut)))
                            eval_string = f"float(np.sum(tree[{total_cut}].weight_MC_Lumi_pu))"
                            process_dict[signal_name][dataset_name]['weight_cut'] += eval(eval_string)
                        empty_dict_example_file = os.path.join(process_dir, nano_file)




        print("Looping over backgrounds in config")
        for background_name in config_dict['background']:
            background_dict = config_dict['background'][background_name]
            class_value = background_dict['class_value']
            dataset_names = background_dict['background_datasets']

            if background_name not in process_dict.keys():
                print(f"Background {background_name} not in process_dict, skip")
                continue

            print(f"Looping background {background_name}")
            for dataset_name in tqdm(dataset_names):
                process_dict[background_name][dataset_name] = {
                    'total': 0,
                    'total_cut': 0,
                    'weight_cut': 0,
                    'nBatches': 0,
                    'batch_size': 0,
                    'batch_start': 0,
                    'class_value': class_value,
                    'all_extensions': [],
                    'storage_folder': storage_folder 
                }

                extension_list = [ fn for fn in os.listdir(storage_folder) if fn.startswith(f"{dataset_name}_ext") ]
                process_dict[background_name][dataset_name]['all_extensions'] = [dataset_name] + extension_list

                for ext_name in process_dict[background_name][dataset_name]['all_extensions']:
                    process_dir = os.path.join(storage_folder, ext_name)
                    for nano_file in [x for x in os.listdir(process_dir) if x.endswith(".root")]:
                        with uproot.open(f"{os.path.join(process_dir, nano_file)}:Events") as h:
                            tree = h.arrays(selection_branches)
                            process_dict[background_name][dataset_name]['total'] += int(h.num_entries)
                            process_dict[background_name][dataset_name]['total_cut'] += int(np.sum(eval(total_cut)))
                            eval_string = f"float(np.sum(tree[{total_cut}].weight_MC_Lumi_pu))"
                            process_dict[background_name][dataset_name]['weight_cut'] += eval(eval_string)

                print(f"Finished background {dataset_name}, how many total? {process_dict[background_name][dataset_name]['total_cut']}")






        #Add totals to start the spin/mass dist and remove the individual signal files
        signal_names = config_dict['signal'].keys()
        for process in process_dict:
            process_dict[process]['total'] = 0
            process_dict[process]['weight'] = 0
            use_combined = False
            if process in signal_names:
                use_combined = config_dict['signal'][process]['use_combined']
            for subprocess in process_dict[process].keys():
                if subprocess.startswith('total') or subprocess.startswith('weight'): continue
                if use_combined:
                    if subprocess == config_dict['signal'][process]['combined_name']: continue
                process_dict[process]['total'] += process_dict[process][subprocess]['total_cut']
                process_dict[process]['weight'] += process_dict[process][subprocess]['weight_cut']


        #Calculate the random spin/mass distribution for backgrounds to be assigned during parametric DNN
        spin_mass_dist = {}

        total_signal = 0
        for signal_name in config_dict['signal']:
            total_signal += process_dict[signal_name]['total']
        for signal_name in config_dict['signal']:
            keys_to_remove = [] #Keys we want to remove if the combined option is being used
            use_combined = config_dict['signal'][signal_name]['use_combined']
            for subprocess in process_dict[signal_name]:
                if subprocess.startswith('total') or subprocess.startswith('weight'): continue
                if use_combined:
                    if subprocess == config_dict['signal'][signal_name]['combined_name']: continue
                subprocess_dict = process_dict[signal_name][subprocess]
                if f"{subprocess_dict['spin']}" not in spin_mass_dist.keys():
                    spin_mass_dist[f"{subprocess_dict['spin']}"] = {}
                spin_mass_dist[f"{subprocess_dict['spin']}"][f"{subprocess_dict['mass']}"] = subprocess_dict['total_cut']/total_signal
                if use_combined:
                    keys_to_remove.append(subprocess)
            #Remove unneeded keys since we will use combined anyway
            for key in keys_to_remove:
                del process_dict[signal_name][key]




        for process in process_dict:
            batch_size_sum = 0
            for subprocess in process_dict[process]:
                if subprocess.startswith('total') or subprocess.startswith('weight'): continue
                process_dict[process][subprocess]['batch_size'] = int(batch_dict[process] * process_dict[process][subprocess]['weight_cut'] / process_dict[process]['weight'])
                print(f"Looking at subprocess {subprocess}")
                nBatches = 0
                if process_dict[process][subprocess]['batch_size'] != 0:
                    nBatches = int(process_dict[process][subprocess]['total_cut']/process_dict[process][subprocess]['batch_size'])
                process_dict[process][subprocess]['nBatches'] = nBatches
                batch_size_sum += process_dict[process][subprocess]['batch_size']

            print(f"Process {process} has batch size sum {batch_size_sum}")
            while batch_size_sum != batch_dict[process]:
                print(f"Warning this is bad batch size, size={batch_size_sum} where goal is {batch_dict[process]}")
                max_batches_subprocess = ""
                max_batches_val = 0
                for subprocess in process_dict[process].keys():
                    if subprocess.startswith('total') or subprocess.startswith('weight'): continue
                    if process_dict[process][subprocess]['nBatches'] > max_batches_val:
                        max_batches_val = process_dict[process][subprocess]['nBatches']
                        max_batches_subprocess = subprocess

                print(f"Trying to fix, incrementing {max_batches_subprocess} batch size {process_dict[process][max_batches_subprocess]['batch_size']} by 1")
                process_dict[process][max_batches_subprocess]['batch_size'] += 1
                print(f"nBatches went from {process_dict[process][max_batches_subprocess]['nBatches']}")
                process_dict[process][max_batches_subprocess]['nBatches'] = int(process_dict[process][max_batches_subprocess]['total_cut']/process_dict[process][max_batches_subprocess]['batch_size'])
                print(f"To {process_dict[process][max_batches_subprocess]['nBatches']}")
                batch_size_sum += 1

        current_index = 0
        for process in process_dict.keys():
            for subprocess in process_dict[process].keys():
                if subprocess.startswith('total') or subprocess.startswith('weight'): continue
                process_dict[process][subprocess]['batch_start'] = current_index
                current_index += process_dict[process][subprocess]['batch_size']


        nBatches = 1e100
        for process in process_dict.keys():
            for subprocess in process_dict[process].keys():
                if subprocess.startswith('total') or subprocess.startswith('weight'): continue
                if process_dict[process][subprocess]['nBatches'] < nBatches and (process_dict[process][subprocess]['nBatches'] != 0):
                    nBatches = process_dict[process][subprocess]['nBatches']




        print(f"Creating {nBatches} batches, according to distribution. ")
        print(process_dict)
        print(f"And total batch size is {batch_dict['batch_size']}")


        machine_yaml = {
            'meta_data': {},
            'processes': [],
        }

        machine_yaml['meta_data']['storage_folder'] = storage_folder
        machine_yaml['meta_data']['batch_dict'] = batch_dict
        machine_yaml['meta_data']['selection_branches'] = selection_branches
        machine_yaml['meta_data']['selection_cut'] = total_cut
        machine_yaml['meta_data']['iterate_cut'] = config_dict['iterate_cut'].format(nParity = config_dict['nParity'], parity_scan = nParity)
        machine_yaml['meta_data']['empty_dict_example'] = empty_dict_example_file #Example for empty dict structure
        machine_yaml['meta_data']['input_filename'] = f'batchfile{nParity}.root'
        machine_yaml['meta_data']['output_DNNname'] = f'ResHH_Classifier_parity{nParity}'

        machine_yaml['meta_data']['spin_mass_dist'] = spin_mass_dist #Dict of spin/mass distribution values for random choice parametric


        for process in process_dict:
            for subprocess in process_dict[process]:
                if subprocess.startswith('total') or subprocess.startswith('weight'): continue                
                print("Using subprocess ", subprocess)
                subprocess_dict = process_dict[process][subprocess]
                datasets_full_pathway = [ os.path.join(subprocess_dict['storage_folder'], fn) for fn in subprocess_dict['all_extensions'] ]
                tmp_process_dict = {
                    'datasets': datasets_full_pathway,
                    'class_value': subprocess_dict['class_value'],
                    'batch_start': subprocess_dict['batch_start'],
                    'batch_size': subprocess_dict['batch_size'],
                    'nBatches': subprocess_dict['nBatches'],
                }
                machine_yaml['processes'].append(tmp_process_dict)

        with open(os.path.join(output_folder, out_yaml), 'w') as outfile:
            yaml.dump(machine_yaml, outfile)



def create_file(config_dict, output_folder, out_filename):
    print(f"Starting create file. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}")
    nBatches = None
    print(config_dict.keys())
    for process in config_dict['processes']:
        if (nBatches is None) or ((process['nBatches'] <= nBatches) and (process['nBatches'] != 0)):
            nBatches = process['nBatches']

    print(f"Going to make {nBatches} batches")
    batch_size = config_dict['meta_data']['batch_dict']['batch_size']

    step_idx = 0

    #Get the name/type (And order!) of signal columns
    master_column_types = []
    master_column_names_vec = ROOT.std.vector("string")()
    #Assume master(signal) is saved first and use idx==0 entry to fill

    for process in config_dict['processes']:
        process_filelist = [ f"{x}/*.root" for x in process['datasets'] ]

        tmp_filename = os.path.join(output_folder, f'tmp{step_idx}.root')
        tmpnext_filename = os.path.join(output_folder, f'tmp{step_idx+1}.root')

        print(process_filelist)
        df_in = ROOT.RDataFrame('Events', process_filelist)

        #Filter for nLeps and Parity (iterate cut in config)
        df_in = df_in.Filter(config_dict['meta_data']['iterate_cut'])

        nEntriesPerBatch = process['batch_size']
        nBatchStart = process['batch_start']
        nBatchEnd = nBatchStart+nEntriesPerBatch

        if nEntriesPerBatch == 0:
            print(f"Process has batch size of 0, skip the save loop")
            continue


        #Load df_out, if first iter then load an empty, otherwise load the past file
        if step_idx == 0:
            df_out = ROOT.RDataFrame(nBatches*batch_size)
            df_out = df_out.Define("is_valid", 'false')
            #Fill master column nametype
            for name in df_in.GetColumnNames():
                if name.startswith('gen'): continue
                if name.startswith('weight_') and not name == "weight_MC_Lumi_pu": continue
                master_column_names_vec.push_back(name)
            master_column_types = [str(df_in.GetColumnType(str(c))) for c in master_column_names_vec]
        else:
            df_out = ROOT.RDataFrame('Events', tmp_filename)


        local_column_names_vec = ROOT.std.vector("string")()
        for name in df_in.GetColumnNames():
            if name.startswith('gen'): continue
            if name.startswith('weight_') and not name == "weight_MC_Lumi_pu": continue
            local_column_names_vec.push_back(name)
        local_column_types = [str(df_in.GetColumnType(str(c))) for c in local_column_names_vec]


        #Need a local_to_master_map so that local columns keep the same index as the master columns
        local_to_master_map = [list(master_column_names_vec).index(local_name) for local_name in local_column_names_vec]
        master_size = len(master_column_names_vec)

        queue_size = 10
        max_entries = nEntriesPerBatch*nBatches

        tuple_maker = ROOT.analysis.TupleMaker(*local_column_types)(queue_size, max_entries)

        df_out = tuple_maker.FillDF(ROOT.RDF.AsRNode(df_out), ROOT.RDF.AsRNode(df_in), local_to_master_map, master_size, local_column_names_vec, nBatchStart, nBatchEnd, batch_size)

        for column_idx, column_name in enumerate(master_column_names_vec):
            column_type = master_column_types[column_idx]

            if step_idx == 0:
                df_out = df_out.Define(str(column_name), f'_entry ? _entry->GetValue<{column_type}>({column_idx}) : {column_type}() ')
            else:
                if column_name not in local_column_names_vec: continue
                df_out = df_out.Redefine(str(column_name), f'_entry ? _entry->GetValue<{column_type}>({column_idx}) : {column_name} ')

        class_value = process['class_value']
        if step_idx == 0:
            df_out = df_out.Define('class_value', f'_entry ? int({class_value}) : int()')
        else:
            df_out = df_out.Redefine('class_value', f'_entry ? int({class_value}) : class_value')


        df_out = df_out.Redefine('is_valid', '(is_valid) || (_entry)')


        snapshotOptions = ROOT.RDF.RSnapshotOptions()
        #snapshotOptions.fOverwriteIfExists=False
        #snapshotOptions.fLazy=True
        snapshotOptions.fMode="RECREATE"
        snapshotOptions.fCompressionAlgorithm = getattr(ROOT.ROOT, 'k' + 'ZLIB')
        snapshotOptions.fCompressionLevel = 4
        ROOT.RDF.Experimental.AddProgressBar(df_out)
        print("Going to snapshot")
        save_column_names = ROOT.std.vector("string")(master_column_names_vec)
        save_column_names.push_back('is_valid')
        save_column_names.push_back('class_value')
        df_out.Snapshot('Events', tmpnext_filename, save_column_names, snapshotOptions)

        if step_idx != 0:
            os.system(f"rm {tmp_filename}")

        tuple_maker.join()

        step_idx += 1



    print("Finished create file loop, now we must add the DNN variables")
    # Increment the name indexes before I embarass myself again
    tmp_filename = os.path.join(output_folder, f'tmp{step_idx}.root')
    tmpnext_filename = os.path.join(output_folder, f'tmp{step_idx+1}.root')

    df_out = ROOT.RDataFrame('Events', tmp_filename)


    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    #snapshotOptions.fOverwriteIfExists=False
    #snapshotOptions.fLazy=True
    snapshotOptions.fMode="RECREATE"
    snapshotOptions.fCompressionAlgorithm = getattr(ROOT.ROOT, 'k' + 'ZLIB')
    snapshotOptions.fCompressionLevel = 4
    ROOT.RDF.Experimental.AddProgressBar(df_out)
    print("Going to snapshot")
    # Only need to save the prexisting columns plus the new DNN variables
    save_column_names = ROOT.std.vector("string")(df_out.GetColumnNames())
    df_out = analysis.defineAllP4(df_out)
    df_out = analysis.AddDNNVariables(df_out)
    highlevel_names = [
        'HT', 'dR_dilep', 'dR_dibjet', 
        'dR_dilep_dijet', 'dR_dilep_dibjet',
        'dPhi_lep1_lep2', 'dPhi_jet1_jet2',
        'dPhi_MET_dilep', 'dPhi_MET_dibjet',
        'min_dR_lep0_jets', 'min_dR_lep1_jets',
        'MT', 'MT2', 
        'MT2_ll', 'MT2_bb', 'MT2_blbl',
        'CosTheta_bb',
        'll_mass',
        'bb_mass', 'bb_mass_PNetRegPtRawCorr', 'bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino'
    ]
    for highlevel_name in highlevel_names:
        save_column_names.push_back(highlevel_name)
    df_out.Snapshot('Events', tmpnext_filename, save_column_names, snapshotOptions)


    print(f"Finished create file, will copy tmp file to final output {out_filename}")

    os.system(f"mv {tmpnext_filename} {out_filename}")
    os.system(f"rm {tmp_filename}")



def create_weight_file(inName, outName, bb_low=70, bb_high=150, bb_min=70, bb_max=300):
    print(f"On file {inName}")
    in_file = uproot.open(inName)
    out_file = uproot.recreate(outName)

    tree = in_file['Events']
    branches_to_load = ["class_value", "bb_mass", "bb_mass_PNetRegPtRawCorr", "bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino", "X_mass", "centralJet_hadronFlavour", "centralJet_pt", "SelectedFatJet_hadronFlavour", "weight_MC_Lumi_pu"]
    branches = tree.arrays(branches_to_load)

    class_value = branches["class_value"]
    bb_mass = branches["bb_mass"]
    bb_mass = branches["bb_mass_PNetRegPtRawCorr"]
    bb_mass = branches["bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino"]

    X_mass = branches['X_mass']

    hadronFlavour = ak.fill_none(ak.pad_none(branches["centralJet_hadronFlavour"], 2, axis=1), 0)
    ak8_hadronFlavour = ak.fill_none(ak.pad_none(branches["SelectedFatJet_hadronFlavour"], 1, axis=1), 0)

    # type_to_name = {'1': 'Signal', '2': 'Signal', '8': 'TT', '5': 'DY', '9': 'ST'} # 1 is Radion, 2 is Graviton
    # type_to_target = {'1': 0, '2': 0, '8': 1, '5': 2, '9': 3} # Multiclass type-to-target
    # type_to_target = {'1': 0, '2': 0, '8': 1, '5': 1, '9': 1} # Binary type-to-target

    value_to_name = {'0': 'Signal', '1': 'TT', '2': 'DY', '3': 'ST', '4': 'W'}
    # value_to_target = {'0': 0, '1': 1, '2': 1, '3': 1, '4': 1} # Binary type-to-target
    value_to_target = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4} # Multiclass type-to-target

    sample_name = np.array([value_to_name[str(value)] for value in class_value])
    class_targets = np.array([value_to_target[str(value)] for value in class_value])

    # Initialize the two branches, class weight and adv weight
    # Starting from their genWeight (includes XS and such)
    class_weight = branches['weight_MC_Lumi_pu']
    adv_weight = branches['weight_MC_Lumi_pu']

    # Lets just flatten the weight by bb_mass first for each sample
    for this_name in np.unique(sample_name):
        bb_mass_thissample = np.where(
            sample_name == this_name,
            bb_mass,
            -1.0
        )
        weights_thissample = np.where(
            sample_name == this_name,
            class_weight,
            0.0
        )

        # Create a histogram of bb_mass and change the weight of each event
        # to be 1 / (nEvents in that bin)
        # This will make the bb_mass distribution flat
        hist, bin_edges = np.histogram(bb_mass_thissample, bins=100, range=(70, 300), weights=weights_thissample)
        bin_indices = np.digitize(bb_mass_thissample, bin_edges)
        bin_indices = np.where(bin_indices == len(hist)+1, len(hist), bin_indices) # If we are in the overflow bin, set to last bin
        bin_indices = np.where(bin_indices == 0, 1, bin_indices) # If we are in the underflow bin, set to first bin
        bin_counts = hist[bin_indices-1] # Get the counts for each event
        class_weight = np.where(
            (sample_name == this_name) & (bin_counts > 0),
            class_weight / bin_counts,
            class_weight
        )
        adv_weight = np.where(
            (sample_name == this_name) & (bin_counts > 0),
            adv_weight / bin_counts,
            adv_weight
        )

    # First step, remove any sample types we want to
    # samples_to_remove = [ 'DY' ]
    samples_to_remove = []

    for sample_to_remove in samples_to_remove:
        class_weight = np.where(
            sample_name == sample_to_remove,
            0.0,
            class_weight
        )

        adv_weight = np.where(
            sample_name == sample_to_remove,
            0.0,
            adv_weight
        )


    # Next normalize between sample types (class)

    # First remove the signal that is not gen bb
    # class_weight = np.where(
    #     sample_name == 'Signal',
    #     np.where(
    #         ((hadronFlavour[:,0] == 5) & (hadronFlavour[:,1] == 5)) | (ak8_hadronFlavour[:,0] == 5), # & (X_mass == 800), # For now, only train on m450
    #         class_weight,
    #         0.0
    #     ),
    #     class_weight
    # )


    # Total_Signal == Total_DY + Total_TT (Equal weight of signal vs background in binary)
    total_signal = np.sum(np.where(sample_name == 'Signal', class_weight, 0.0))
    total_background = np.sum(np.where(sample_name != 'Signal', class_weight, 0.0))

    norm_factor = total_signal / total_background
    class_weight = np.where(
        sample_name == 'Signal',
        class_weight,
        class_weight * norm_factor
    )


    # Next normalize between m_bb regions (adversarial)
    # TT_Low == TT_Mid == TT_High
    # DY_Low == DY_Mid == DY_High

    # TT_Total / DY_Total == TT_yield / DY_yield
    adv_weight = np.where(
        sample_name == 'Signal',
        0.0,
        adv_weight
    )
    # bb_low = 70
    # bb_high = 150

    # Set adv targets
    adv_targets = np.where(
        bb_mass < bb_low,
        -1,
        np.where(
            bb_mass < bb_high,
            0,
            1
        )
    )

    #Option to set an lower and upper 
    # bb_min = 70
    # bb_max = 300
    adv_weight = np.where(
        bb_mass > bb_min,
        adv_weight,
        0.0
    )
    adv_weight = np.where(
        bb_mass < bb_max,
        adv_weight,
        0.0
    )



    for this_name in np.unique(sample_name):
        if this_name == 'Signal': continue
        print(f"On sample {this_name}")
        total_low = np.sum(
            np.where(
                (sample_name == this_name) & (bb_mass < bb_low),
                adv_weight,
                0.0
            )
        )
        total_mid = np.sum(
            np.where(
                (sample_name == this_name) & (bb_mass > bb_low) & (bb_mass < bb_high),
                adv_weight,
                0.0
            )
        )
        total_high = np.sum(
            np.where(
                (sample_name == this_name) & (bb_mass > bb_high),
                adv_weight,
                0.0
            )
        )
        # norm to mid
        adv_weight = np.where(
            (sample_name == this_name) & (bb_mass < bb_low),
            # total_mid * adv_weight / total_low,
            0.0, # For now, we will just ignore the down category
            adv_weight
        )
        adv_weight = np.where(
            (sample_name == this_name) & (bb_mass > bb_high),
            total_mid * adv_weight / total_high,
            adv_weight
        )



        total_scaled = np.sum(np.where(sample_name == this_name, adv_weight, 0.0))
        adv_weight = np.where(
            (sample_name == this_name),
            adv_weight / total_scaled,
            adv_weight
        )

    # Nan to num for any divide by 0 errors
    class_weight = np.nan_to_num(class_weight, 0.0)
    adv_weight = np.nan_to_num(adv_weight, 0.0)


    # Normalize both class weights and adv weights to nEvents
    print(f"Before normalization our class total {np.sum(class_weight)} and adv total {np.sum(adv_weight)}")
    nEvents = len(class_weight)
    class_weight = (nEvents / np.sum(class_weight)) * class_weight
    adv_weight = (nEvents / np.sum(adv_weight)) * adv_weight
    print(f"After normalization our class total {np.sum(class_weight)} and adv total {np.sum(adv_weight)}")


    out_dict = {
        "class_weight": class_weight,
        "adv_weight": adv_weight,
        "class_target": class_targets,
        "adv_target": adv_targets,
    }

    out_file["weight_tree"] = out_dict

    # m_bb weight validation plots
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.ROOT)
    fig, ax = plt.subplots(1,1)
    for this_name in np.unique(sample_name):
        mask = sample_name == this_name
        ax.hist(bb_mass[mask], weights=class_weight[mask], bins=100, range=(0,500), histtype='step', label=this_name)
    # ax.set_yscale('log')
    ax.set_xlabel(r'$m_{bb}$ [GeV]')
    ax.set_ylabel('Weighted Events')
    ax.legend()
    plt.savefig(outName.replace('.root', '_class_weight_bbmass.png'))
    plt.close()


    fig, ax = plt.subplots(1,1)
    for this_name in np.unique(sample_name):
        mask = sample_name == this_name
        ax.hist(bb_mass[mask], weights=adv_weight[mask], bins=100, range=(0,500), histtype='step', label=this_name)
    # ax.set_yscale('log')
    ax.set_xlabel(r'$m_{bb}$ [GeV]')
    ax.set_ylabel('Weighted Events')
    ax.legend()
    plt.savefig(outName.replace('.root', '_adv_weight_bbmass.png'))
    plt.close()





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create TrainTest Files for DNN.')
    parser.add_argument('--config', required=False, type=str, default='default_dataset.yaml', help="Config YAML")
    parser.add_argument('--output-folder', required=False, type=str, default='/eos/user/d/daebi/DNN_Training_Datasets', help="Output folder to store dataset")
    parser.add_argument('--era', required=False, type=str, default='Run3_2022', help="Era of data taking")

    args = parser.parse_args()

    config_file = args.config
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)

    path_exists = False
    
    # output_base = "DNN_Datasets"
    output_base = args.output_folder
    output_folder = os.path.join(output_base, f"Dataset_{args.era}")
    if os.path.exists(output_folder):
        print(f"Output folder {output_folder} exists!!!")
        path_exists = True
    os.makedirs(output_folder, exist_ok=True)
    os.system(f"cp {config_file} {output_folder}/.")

    print("Will create signal files")
    create_signal_files(config_dict, output_folder, args.era)
    print("Creating the batch dict")
    create_dict(config_dict, output_folder, args.era)

    gc.collect()

    print(f"We have finished making the dicts. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}")


    headers_dir = os.path.dirname(os.path.abspath(__file__))
    #headers = [ 'AnalysisTools.h', 'TupleMaker.h' ] #Order here matters since TupleMaker requires AnalysisTools
    headers = [ 'TupleMaker.h' ] #Order here matters since TupleMaker requires AnalysisTools
    for header in headers:
        header_path = os.path.join(headers_dir, header)
        if not ROOT.gInterpreter.Declare(f'#include "{header_path}"'):
            raise RuntimeError(f'Failed to load {header_path}')


    print(f"Starting the create file loop. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}")
    print(output_folder)
    print(os.listdir(output_folder))
    yaml_list = [fname for fname in os.listdir(output_folder) if ((".yaml" in fname) and ("batch_config_parity" in fname))]
    yaml_list.sort()
    for i, yamlname in enumerate(yaml_list):
        print(f"Starting batch {i} with yaml {yamlname}")
        config_dict = {}
        with open(os.path.join(output_folder, yamlname), 'r') as file:
            config_dict = yaml.safe_load(file)           
        if os.path.exists(os.path.join(output_folder, config_dict['meta_data']['input_filename'])):
            print("File exists, skipping")
            continue
        create_file(config_dict, output_folder, os.path.join(output_folder, config_dict['meta_data']['input_filename']))


    print("Finished making all the batch files, now we will make the weight files")
    inDir = output_folder
    batchfiles = [x for x in os.listdir(inDir) if 'batchfile' in x]

    bb_low = 70
    bb_high = 150

    bb_min = 70
    bb_max = 300

    for batchfile_name in batchfiles:
        weightfile_name = f'weightfile{batchfile_name[-6:]}'

        in_file = os.path.join(inDir, batchfile_name)
        out_file = os.path.join(inDir, weightfile_name)

        print(f"Starting infile {in_file} and making outfile {out_file}")
        if os.path.exists(out_file):
            print(f"Weight file {out_file} exists, skip")
            # continue
        create_weight_file(in_file, out_file, bb_low, bb_high, bb_min, bb_max)