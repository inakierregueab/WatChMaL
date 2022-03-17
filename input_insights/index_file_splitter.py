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
# TODO: fix ratio of splitting (70/10/20)
test_files_start = 0
test_files_stop = test_files_start + 10     # first 400 files are for test set
val_files_start = test_files_stop
val_files_stop = val_files_start + 10   # next 100 files are for validation set
train_files_start = val_files_stop
train_files_stop = None     # all remaining files are for training set

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

# perform splitting
split_files = {"test_idxs":  [f for l in labels for f in files_in_labels[l][0:10]],
               "val_idxs":   [f for l in labels for f in files_in_labels[l][10:20]],
               "train_idxs": [f for l in labels for f in files_in_labels[l][20:]]}
split_idxs = {k: [i for f in v for i in idxs_in_files[f]] for k, v in split_files.items()}

for s in split_files.keys():
    print(s,"has", len(split_files[s]),"files and", len(split_idxs[s]),"indices")

# verify that all events are uniquely accounted for
all_indices = np.concatenate(list(split_idxs.values()))
print(len(event_labels))
print(len(all_indices))
print(len(set(all_indices)))

# export
np.savez('IWCD_mPMT_Short_2_class_3M_emgp0_fixed_idxs.npz', **split_idxs)
