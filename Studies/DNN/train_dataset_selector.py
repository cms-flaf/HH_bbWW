import os
import uproot
import numpy as np
import psutil
from datetime import datetime
import yaml

def variable_dict(batch_size, process_subdict, process_iter, output_array, mass_values):
    vardict = {}
    write_array = np.array(output_array)
    nBatchesThisChunk = len(write_array['lep1_pt'])//batch_size
    if len(write_array['lep1_pt'])%batch_size != 0: print("UH OH")
    if nBatchesThisChunk == 0: return vardict

    class_value = process_subdict['class_value']
    mass_value = process_subdict['mass_value']

    for nBatch in range(nBatchesThisChunk):
        process_array = next(process_iter)
        index_start = process_subdict['batch_start'] + (batch_size*nBatch)
        index_end = index_start+process_subdict['batch_size']

        write_array['class_value'][index_start:index_end] = np.full(index_end-index_start, class_value)
        if mass_value <= 0:
            write_array['mass_value'][index_start:index_end] = np.random.choice(mass_values, size=index_end-index_start).astype(float)
        else:
            write_array['mass_value'][index_start:index_end] = np.full(index_end-index_start, mass_value)


        write_array['lep1_pt'][index_start:index_end] =     np.asarray(process_array['lep1_pt'])
        write_array['lep1_eta'][index_start:index_end] =    np.asarray(process_array['lep1_eta'])
        write_array['lep1_phi'][index_start:index_end] =    np.asarray(process_array['lep1_phi'])
        write_array['lep1_mass'][index_start:index_end] =   np.asarray(process_array['lep1_mass'])

        write_array['lep2_pt'][index_start:index_end] =     np.asarray(process_array['lep2_pt'])
        write_array['lep2_eta'][index_start:index_end] =    np.asarray(process_array['lep2_eta'])
        write_array['lep2_phi'][index_start:index_end] =    np.asarray(process_array['lep2_phi'])
        write_array['lep2_mass'][index_start:index_end] =   np.asarray(process_array['lep2_mass'])

    vardict = {
        'class_value':  write_array['class_value'],
        'mass_value':   write_array['mass_value'],

        'lep1_pt':      write_array['lep1_pt'],
        'lep1_eta':     write_array['lep1_eta'],
        'lep1_phi':     write_array['lep1_phi'],
        'lep1_mass':    write_array['lep1_mass'],

        'lep2_pt':      write_array['lep2_pt'],
        'lep2_eta':     write_array['lep2_eta'],
        'lep2_phi':     write_array['lep2_phi'],
        'lep2_mass':    write_array['lep2_mass'],
    }
    return vardict


def init_empty_vardict(batch_size):
    empty_dict = {
        'class_value':  np.zeros(batch_size),
        'mass_value':   np.zeros(batch_size),

        'lep1_pt':      np.zeros(batch_size),
        'lep1_eta':     np.zeros(batch_size),
        'lep1_phi':     np.zeros(batch_size),
        'lep1_mass':    np.zeros(batch_size),

        'lep2_pt':      np.zeros(batch_size),
        'lep2_eta':     np.zeros(batch_size),
        'lep2_phi':     np.zeros(batch_size),
        'lep2_mass':    np.zeros(batch_size),
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



def create_traintest(storage_folder, output_file_pattern, batch_dict, mass_values, process_class_and_names):
    if 'batch_size' not in batch_dict.keys():
        batch_dict['batch_size'] = 0
    process_dict = {}
    for key in batch_dict.keys():
        if key == 'batch_size': continue
        batch_dict['batch_size'] += batch_dict[key]
        process_dict[key] = {}

    out_filename = output_file_pattern.format(batch_dict['batch_size'])+f"_{datetime.today().strftime('%Y-%m-%d')}.root"



    for process_class in process_class_and_names.keys():
        class_value = process_class_and_names[process_class]['class_value']
        process_names = process_class_and_names[process_class]['process_names'].keys()
        for process_name in process_names:
            mass_value = -1.0
            if 'mass_string' in process_class_and_names[process_class].keys():
                mass_value = float(eval(f"'{process_name}'{process_class_and_names[process_class]['mass_string']}"))
            process_dict[process_class][process_name] = {
                'total': 0,
                'nBatches': 0,
                'batch_size': 0,
                'batch_start': 0,
                'class_value': class_value,
                'mass_value': mass_value,
                'all_extensions': []
            }
            #Check for the extension files
            process_dict[process_class][process_name]['all_extensions'].append(process_name)
            if (process_class_and_names[process_class]['process_names'][process_name]) != None:
                for ext_name in (process_class_and_names[process_class]['process_names'][process_name]):
                    process_dict[process_class][process_name]['all_extensions'].append(ext_name)
            for ext_name in process_dict[process_class][process_name]['all_extensions']:
                process_directory = os.path.join(storage_folder, ext_name)
                for nano_file in os.listdir(process_directory):
                    with uproot.open(os.path.join(process_directory, nano_file)+":Events") as h:
                        process_dict[process_class][process_name]['total'] += h.num_entries


    for process in process_dict.keys():
        process_dict[process]['total'] = 0
        for subprocess in process_dict[process].keys():
            if subprocess == 'total': continue
            process_dict[process]['total'] += process_dict[process][subprocess]['total']

        batch_size_sum = 0
        for subprocess in process_dict[process].keys():
            if subprocess == 'total': continue
            process_dict[process][subprocess]['batch_size'] = max(int(process_dict[process][subprocess]['total']/process_dict[process]['total'] * batch_dict[process]), 1) #Hard require at least 1 per batch
            process_dict[process][subprocess]['nBatches'] = int(process_dict[process][subprocess]['total']/process_dict[process][subprocess]['batch_size']) #Requirement is due to divide here, but maybe we want some batches to not have certain low-rate backgrounds?
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


    nBatchesPerChunk = int(nBatches/2) #This still is changed by hand, should be optimized for RAM usage
    for process in process_dict.keys():
        for subprocess in process_dict[process].keys():
            if subprocess == 'total': continue

            out_file = uproot.open(out_filename)
            tmp_file = uproot.recreate('tmp.root')


            process_filelist = process_dict[process][subprocess]['all_extensions']
            formatted_filelist = [ f"{os.path.join(storage_folder, x)}/*.root:Events" for x in process_filelist ]

            process_iter = iterate_uproot(formatted_filelist, process_dict[process][subprocess]['batch_size'], nBatchesPerChunk)
            output_iter = iterate_uproot(f"{out_filename}:Events", batch_dict['batch_size'], nBatchesPerChunk)

            chunk_counter = 0
            print(f"Looping output array {subprocess}. Memory usage in MB is ", psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20))
            for output_array in output_iter:
                new_write_dict = variable_dict(batch_dict['batch_size'], process_dict[process][subprocess], process_iter, output_array, mass_values)

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
    storage_folder = config_dict['storage_folder']
    batch_dict = config_dict['batch_dict']
    output_file_pattern = config_dict['output_file_pattern']
    mass_values = config_dict['mass_values']
    process_class_and_names = config_dict['process_class_and_names']
    print(process_class_and_names)

    output_folder = f"DNN_dataset_{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(output_folder, exist_ok=True)
    os.system(f"cp {args.config} {output_folder}/.")

    create_traintest(storage_folder, os.path.join(output_folder, output_file_pattern), batch_dict, mass_values, process_class_and_names)