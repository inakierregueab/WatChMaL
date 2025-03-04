import h5py
import re
import numpy as np


def atoi(text):
    return int(text) if text.isdigit() else text
# Sort by only the basename of the file, with natural sorting of numbers in the filename
def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text.split('/')[-1]) ]


# define which files of events are included in test / validation / training split
# the same split is done for each particle type
test_files_start = 0
test_files_stop = test_files_start+50 # first 400 files are for test set
val_files_start = test_files_stop
val_files_stop = val_files_start+12 # next 100 files are for validation set
train_files_start = val_files_stop
train_files_stop = 500

# define which particle labels to include
labels = (0, 1)     # (photon, electron)

# load dataset
data_path = "/data/neutrinos/IWCD_Data/IWCD_mPMT_Short_emgp0_E0to1000MeV_digihits.h5"
f = h5py.File(data_path, "r")
event_labels = np.array(f['labels'])
root_files = np.array(f['root_files']).astype(str)

# get files and indices
files_in_labels = {l: sorted(set(root_files[event_labels==l]), key=natural_keys) for l in labels}
idxs_in_files = {f: range(i, i+c) for f,i,c in zip(*np.unique(root_files, return_index=True, return_counts=True))}

for l, f in files_in_labels.items():
    print("label", l,"has", len(f),"files and ", sum([len(idxs_in_files[i]) for i in f]), "indices")
# label 0 has 3000 files and  8868592 indices
# label 1 has 3000 files and  8833531 indices


# perform splitting
split_files = {"test_idxs":  [f for l in labels for f in files_in_labels[l][test_files_start:test_files_stop]],
               "val_idxs":   [f for l in labels for f in files_in_labels[l][val_files_start:val_files_stop]],
               "train_idxs": [f for l in labels for f in files_in_labels[l][train_files_start:train_files_stop]]}
split_idxs = {k: [i for f in v for i in idxs_in_files[f]] for k, v in split_files.items()}

for s in split_files.keys():
    print(s,"has", len(split_files[s]),"files and", len(split_idxs[s]),"indices")

# verify that all events are uniquely accounted for
all_indices = np.concatenate(list(split_idxs.values()))
print(len(event_labels))
print(len(all_indices))
print(len(set(all_indices)))

# export
np.savez('/home/ierregue/ssh_tunel/data/IWCD_mPMT_Short_2_class_idxs_xps.npz', **split_idxs)

splitting = np.load('IWCD_mPMT_Short_2_class_idxs_14_files.npz')

x=0
