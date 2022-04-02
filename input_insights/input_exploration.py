import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import h5py

from watchmal.dataset.h5_dataset import H5Dataset


# LOAD INDEX SPLITTER:
splitting = np.load('/home/ierregue/ssh_tunel/data/IWCD_mPMT_Short_2_class_idxs_14_files.npz')
train_idxs = splitting['train_idxs']

#   DATASET
data = {
    'hit_charge': [],
    'hit_time': [],
    'hit_pmt': [],
    'event_label': [],
    'polar': [],
    'azimuth': [],
    'x': [],
    'y': [],
    'z': [],
    'energies': []
}

h5_path = '/data/neutrinos/IWCD_Data/IWCD_mPMT_Short_emgp0_E0to1000MeV_digihits.h5'
dataset = H5Dataset(h5_path, is_distributed=False)
raw_h5_file = h5py.File(h5_path, "r")

# Get event-hit mapping for selected events
original_mapping = dataset.event_hits_index[train_idxs]
data['event_hits_index'] = original_mapping - original_mapping[0]
hits_shifted = np.append(data['event_hits_index'][1:], len(original_mapping))
data['num_hits_per_event'] = hits_shifted - data['event_hits_index']

# Add hit info (pmt, charge and time) and labels
for idx in train_idxs:
    data['event_label'].append(dataset[idx]['labels'])
    data['hit_charge'] += list(dataset.event_hit_charges)
    data['hit_time'] += list(dataset.event_hit_times)
    data['hit_pmt'] += list(dataset.event_hit_pmts)

electrons = sum(data['event_label'])
num_events = len(data['event_label'])
num_hits = len(data['hit_pmt'])
gammas = num_events-electrons
print(f'Number of events: {num_events}')
print(f'Number of hits: {num_hits}')
print(f'Number of 1\'s (e): {electrons}, percentage: {(electrons/num_events)*100} %')
print(f'Number of 0\'s (gamma): {gammas}, percentage: {(gammas/num_events)*100} %')

# GET METADATA
h5_relevant_keys = ['angles', 'energies', 'positions']
for feature in h5_relevant_keys:
    for idx in train_idxs:
        if feature == 'angles':
            data['polar'].append(raw_h5_file.get(feature)[idx,0])
            data['azimuth'].append(raw_h5_file.get(feature)[idx,1])
        elif feature == 'positions':
            data['x'].append(raw_h5_file.get(feature)[idx,0,0])
            data['y'].append(raw_h5_file.get(feature)[idx,0,1])
            data['z'].append(raw_h5_file.get(feature)[idx,0,2])
        else:
            data[feature].append(raw_h5_file.get(feature)[idx,0])

# Cast to ndarray
for key in data.keys():
    data[key] = np.array(data[key])

# Add total charge sum and time interval
# TODO: re-do
data['charge_sum'] = np.empty(0)
data['time_intervals'] = np.empty(0)
initial_hit = 0
for final_hit in hits_shifted:
    data['charge_sum'] = np.append(data['charge_sum'], np.sum(data['hit_charge'][initial_hit:final_hit]))
    time_values = data['hit_time'][initial_hit:final_hit]
    if len(time_values) == 0:   #TODO: outlier?
        data['time_intervals'] = np.append(data['time_intervals'], 0)
    else:
        data['time_intervals'] = np.append(data['time_intervals'], np.max(time_values)-np.min(time_values))
    initial_hit = final_hit

dict_to_df = {}
for key in data.keys():
    if key not in ['hit_charge', 'hit_time', 'hit_pmt', 'event_hits_index']:
        dict_to_df[key] = data[key]

events_df = pd.DataFrame(dict_to_df, columns=dict_to_df.keys())

# Plot kernel densities
def kde_plotter(events_df):
    # KDE plot
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
    events_df.groupby('event_label')['polar'].plot.kde(ax=axes[0, 0])
    axes[0, 0].set_ylabel('')
    axes[0, 0].set_xlabel(r'$\theta$')
    axes[0, 0].legend()
    events_df.groupby('event_label')['azimuth'].plot.kde(ax=axes[0, 1])
    axes[0, 1].set_ylabel('')
    axes[0, 1].set_xlabel(r'$\varphi$')
    axes[0, 1].legend()
    events_df.groupby('event_label')['energies'].plot.kde(ax=axes[0, 2])
    axes[0, 2].set_ylabel('')
    axes[0, 2].set_xlabel(r'$E$')
    axes[0, 2].legend()
    events_df.groupby('event_label')['x'].plot.kde(ax=axes[1, 0])
    axes[1, 0].set_ylabel('')
    axes[1, 0].set_xlabel(r'$x$')
    axes[1, 0].legend()
    events_df.groupby('event_label')['y'].plot.kde(ax=axes[1, 1])
    axes[1, 1].set_ylabel('')
    axes[1, 1].set_xlabel(r'$y$')
    axes[1, 1].legend()
    events_df.groupby('event_label')['z'].plot.kde(ax=axes[1, 2])
    axes[1, 2].set_ylabel('')
    axes[1, 2].set_xlabel(r'$z$')
    axes[1, 2].legend()
    events_df.groupby('event_label')['num_hits_per_event'].plot.kde(ax=axes[2, 0])
    axes[2, 0].set_ylabel('')
    axes[2, 0].set_xlabel('num_hits')
    axes[2, 0].legend()
    events_df.groupby('event_label')['charge_sum'].plot.kde(ax=axes[2, 1])
    axes[2, 1].set_ylabel('')
    axes[2, 1].set_xlabel('charge_sum')
    axes[2, 1].legend()
    events_df.groupby('event_label')['time_intervals'].plot.kde(ax=axes[2, 2])
    axes[2, 2].set_ylabel('')
    axes[2, 2].set_xlabel('time_interval')
    axes[2, 2].legend()
    # plt.show(block=False)
    plt.show()

# TODO: error with charge sum and time intervals for electrons?
kde_plotter(events_df)

# Feature correlation
pd.plotting.scatter_matrix(events_df.drop(columns=['labels', 'event_hits_index']), alpha=0.2, figsize=(15, 15), diagonal='kde')
plt.show(block=False)
plt.savefig('./../../outputs/analysis/features_correlation _matrix.png')

