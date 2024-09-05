import os
import uproot
import numpy as np
import psutil
from datetime import datetime
import yaml

def variable_dict(batch_size, process_subdict, process_iter, output_array):
    vardict = {}
    write_array = np.array(output_array)
    nBatchesThisChunk = len(write_array['lep1_pt'])//batch_size
    if len(write_array['lep1_pt'])%batch_size != 0: print("UH OH")
    if nBatchesThisChunk == 0: return vardict

    for nBatch in range(nBatchesThisChunk):
        process_array = next(process_iter)
        index_start = process_subdict['batch_start'] + (batch_size*nBatch)
        index_end = index_start+process_subdict['batch_size']

        write_array['lep1_pt'][index_start:index_end] = np.asarray(process_array['lep1_pt'])
        write_array['lep1_eta'][index_start:index_end] = np.asarray(process_array['lep1_eta'])
        write_array['lep1_phi'][index_start:index_end] = np.asarray(process_array['lep1_phi'])
        write_array['lep1_mass'][index_start:index_end] = np.asarray(process_array['lep1_mass'])

    vardict = {
        'lep1_pt': write_array['lep1_pt'],
        'lep1_eta': write_array['lep1_eta'],
        'lep1_phi': write_array['lep1_phi'],
        'lep1_mass': write_array['lep1_mass'],
    }
    return vardict


def init_empty_vardict(batch_size):
    empty_dict = {
        'lep1_pt': np.zeros(batch_size),
        'lep1_eta': np.zeros(batch_size),
        'lep1_phi': np.zeros(batch_size),
        'lep1_mass': np.zeros(batch_size),
    }
    return empty_dict




def iterate_uproot(fnames, batch_size, nBatchesPerChunk):
    for iter in uproot.iterate(fnames, step_size=nBatchesPerChunk*batch_size):
        p = 0
        while p < nBatchesPerChunk*batch_size:
            if p+batch_size > len(iter):
                print("Bad new bears, end of file")
                print(fnames)
                print(p+batch_size)
                print(len(iter))
                break
            else:
                out_array = iter[p:p+batch_size]
                p+= batch_size
                yield out_array




def create_traintest(storage_folder, output_file_pattern, batch_dict):
    if 'batch_size' not in batch_dict.keys():
        batch_dict['batch_size'] = 0
    process_dict = {}
    for key in batch_dict.keys():
        if key == 'batch_size': continue
        batch_dict['batch_size'] += batch_dict[key]
        process_dict[key] = {}

    out_filename = output_file_pattern.format(batch_dict['batch_size'])+f"_{datetime.today().strftime('%Y-%m-%d')}.root"

    samples = []

    for process in os.listdir(storage_folder):
        if process.startswith("GluGlutoRadion"):

            process_dict['signal'][process] = {
                'total': 0,
                'nBatches': 0,
                'batch_size': 0,
                'batch_start': 0,
            }

            samples.append(os.path.join(storage_folder, process))
            for nano_file in os.listdir(os.path.join(storage_folder, process)):
                with uproot.open(os.path.join(storage_folder, process, nano_file)+":Events") as h:
                    process_dict['signal'][process]['total'] += h.num_entries


        elif process.startswith("TT"):
            if not process == "TT": continue

            process_dict['TT'][process] = {
                'total': 0,
                'nBatches': 0,
                'batch_size': 0,
                'batch_start': 0,
            }

            samples.append(os.path.join(storage_folder, process))
            for nano_file in os.listdir(os.path.join(storage_folder, process)):
                with uproot.open(os.path.join(storage_folder, process, nano_file)+":Events") as h:
                    process_dict['TT'][process]['total'] += h.num_entries


    for process in process_dict.keys():
        process_dict[process]['total'] = 0
        for subprocess in process_dict[process].keys():
            if subprocess == 'total': continue
            process_dict[process]['total'] += process_dict[process][subprocess]['total']

        batch_size_sum = 0
        for subprocess in process_dict[process].keys():
            if subprocess == 'total': continue
            process_dict[process][subprocess]['batch_size'] = int(process_dict[process][subprocess]['total']/process_dict[process]['total'] * batch_dict[process])
            process_dict[process][subprocess]['nBatches'] = int(process_dict[process][subprocess]['total']/process_dict[process][subprocess]['batch_size'])
            batch_size_sum += process_dict[process][subprocess]['batch_size']


        print(f"Process {process} has batch size sum {batch_size_sum}")
        while batch_size_sum != batch_dict[process]:
            print(f"Warning this is bad batch size, size={batch_size_sum} where goal is {batch_dict[process]}")

            max_batches_subprocess = ""
            max_batches_val = 0
            for subprocess in process_dict[process].keys():
                if subprocess == 'total': continue
                if process_dict[process][subprocess]['nBatches'] > max_batches_val:
                    max_batches_val = process_dict[process][subprocess]['nBatches']
                    max_batches_subprocess = subprocess

            print(f"Trying to fix, incrementing {max_batches_subprocess} batch size {process_dict[process][max_batches_subprocess]['batch_size']} by 1")
            process_dict[process][max_batches_subprocess]['batch_size'] += 1
            print(f"nBatches went from {process_dict[process][max_batches_subprocess]['nBatches']}")
            process_dict[process][max_batches_subprocess]['nBatches'] = int(process_dict[process][max_batches_subprocess]['total']/process_dict[process][max_batches_subprocess]['batch_size'])
            print(f"To {process_dict[process][max_batches_subprocess]['nBatches']}")
            batch_size_sum += 1

    current_index = 0
    for process in process_dict.keys():
        for subprocess in process_dict[process].keys():
            if subprocess == 'total': continue
            process_dict[process][subprocess]['batch_start'] = current_index
            current_index += process_dict[process][subprocess]['batch_size']



    nBatches = 1e100
    for process in process_dict.keys():
        for subprocess in process_dict[process].keys():
            if subprocess == 'total': continue
            if process_dict[process][subprocess]['nBatches'] < nBatches:
                nBatches = process_dict[process][subprocess]['nBatches']




    print(f"Creating {nBatches} batches, according to distribution")
    print(process_dict)
    print(f"And total batch size is {batch_dict['batch_size']}")

    #Create the root file with all values set to 0, then we will fill by input next
    with uproot.recreate(out_filename) as file:
        for nBatch in range(nBatches):
            empty_dict = init_empty_vardict(batch_dict['batch_size'])
            if 'Events' not in '\t'.join(file.keys()):
                file['Events'] = empty_dict
            else:
                file['Events'].extend(empty_dict)

    print(samples)

    nBatchesPerChunk = int(nBatches/2) #This still is changed by hand, should be optimized for RAM usage
    for process in process_dict.keys():
        for subprocess in process_dict[process].keys():
            if subprocess == 'total': continue
            out_file = uproot.open(out_filename)
            tmp_file = uproot.recreate('tmp.root')



            process_iter = iterate_uproot(f"{os.path.join(storage_folder, subprocess)}/*.root:Events", process_dict[process][subprocess]['batch_size'], nBatchesPerChunk)
            output_iter = iterate_uproot(f"{out_filename}:Events", batch_dict['batch_size'], nBatchesPerChunk)

            chunk_counter = 0
            print(f"Looping output array {subprocess}. Memory usage in MB is ", psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20))
            for output_array in output_iter:
                new_write_dict = variable_dict(batch_dict['batch_size'], process_dict[process][subprocess], process_iter, output_array)

                if 'Events' not in '\t'.join(tmp_file.keys()):
                    tmp_file['Events'] = new_write_dict
                else:
                    tmp_file['Events'].extend(new_write_dict)

            out_file.close()
            tmp_file.close()
            os.system(f"rm {out_filename}")
            os.system(f"mv tmp.root {out_filename}")







if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create TrainTest Files for DNN.')
    parser.add_argument('--config', required=True, type=str, help="Config YAML")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config_dict = yaml.safe_load(file)
    print(config_dict)
    storage_folder = config_dict['storage_folder']
    batch_dict = config_dict['batch_dict']
    process_names = config_dict['process_names']
    output_file_pattern = config_dict['output_file_pattern']


    create_traintest(storage_folder, output_file_pattern, batch_dict)