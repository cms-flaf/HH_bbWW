import os
import uproot
import numpy as np
import psutil

storage_folder = "/eos/user/d/daebi/ANA_FOLDER/anaTuples/dev/Run3_2022/"

samples = []

batch_dict = {
    'signal': 1000,
    'TT': 4000,

    'batch_size': 0
}


for key in batch_dict.keys():
    if key == 'batch_size': continue
    batch_dict['batch_size'] += batch_dict[key]


process_dict = {
    'signal': {},
    'TT': {},
}




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

out_filename = "traintest.root"
#Create the root file with all values set to 0, then we will fill by input next
with uproot.recreate(out_filename) as file:
    for nBatch in range(nBatches):
        empty_dict = {
            'lep1_pt': np.zeros(batch_dict['batch_size']),
            'lep1_eta': np.zeros(batch_dict['batch_size']),
            'lep1_phi': np.zeros(batch_dict['batch_size']),
            'lep1_mass': np.zeros(batch_dict['batch_size']),
        }
        if 'Events' not in '\t'.join(file.keys()):
            file['Events'] = empty_dict
        else:
            file['Events'].extend(empty_dict)

print(samples)


for process in process_dict.keys():
    for subprocess in process_dict[process].keys():
        if subprocess == 'total': continue
        out_file = uproot.open(out_filename)
        tmp_file = uproot.recreate('tmp.root')


        process_arrays = uproot.iterate(f"{os.path.join(storage_folder, subprocess)}/*.root:Events", step_size=process_dict[process][subprocess]['batch_size'])
        write_arrays = uproot.iterate(f"{out_filename}:Events", step_size=batch_dict['batch_size'])

        a = [[process_array, write_array] for process_array in process_arrays for write_array in write_arrays]
        print("Created the arrays. Memory usage in MB is ", psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20))


        for nBatch in range(nBatches):
            index_start = process_dict[process][subprocess]['batch_start']
            index_end = index_start+process_dict[process][subprocess]['batch_size']
            print(f"For subprocess {subprocess} we are starting at {index_start} on batch number {nBatch}")


            process_array = a[nBatch][0]
            write_array = a[nBatch][1]

            new_write_array = np.array(write_array)

            new_write_array['lep1_pt'][index_start:index_end] = np.asarray(process_array['lep1_pt'])
            new_write_array['lep1_eta'][index_start:index_end] = np.asarray(process_array['lep1_eta'])
            new_write_array['lep1_phi'][index_start:index_end] = np.asarray(process_array['lep1_phi'])
            new_write_array['lep1_mass'][index_start:index_end] = np.asarray(process_array['lep1_mass'])

            new_write_dict = {
                'lep1_pt': new_write_array['lep1_pt'],
                'lep1_eta': new_write_array['lep1_eta'],
                'lep1_phi': new_write_array['lep1_phi'],
                'lep1_mass': new_write_array['lep1_mass'],
            }

            if 'Events' not in '\t'.join(tmp_file.keys()):
                tmp_file['Events'] = new_write_dict
            else:
                tmp_file['Events'].extend(new_write_dict)


        out_file.close()
        tmp_file.close()
        os.system(f"rm {out_filename}")
        os.system(f"mv tmp.root {out_filename}")









