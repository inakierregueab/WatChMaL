import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import h5py

from watchmal.dataset.h5_dataset import H5Dataset


# LOAD INDEX SPLITTER:
splitting = np.load('/home/ierregue/ssh_tunel/data/IWCD_mPMT_Short_2_class_idxs_14_files.npz')
train_idxs = splitting['train_idxs']
#train_idxs = train_idxs[:100000]

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

# Get scaling constants
"""mu_q = np.mean(dataset.hdf5_hit_charge[:1000000000])
mu_t = np.mean(dataset.hdf5_hit_time[:1000000000])
std_q = np.std(dataset.hdf5_hit_charge[:1000000000])
std_t = np.std(dataset.hdf5_hit_time[:1000000000])
print('')
print(mu_q)
print(mu_t)
print(std_q)
print(std_t)"""

# Get event-hit mapping for selected events
data['event_hits_index'] = dataset.event_hits_index[train_idxs]

# Add hit info (pmt, charge and time) and labels
data['std_charge_per_event'] = []
data['mean_charge_per_event'] = []
data['charge_sum_per_event'] = []
data['std_time_per_event'] = []
data['mean_time_per_event'] = []
data['num_hits_per_event'] = []
for idx in train_idxs:
    data['event_label'].append(dataset[idx]['labels'])
    data['hit_charge'] += list(dataset.event_hit_charges)
    data['hit_time'] += list(dataset.event_hit_times)
    data['hit_pmt'] += list(dataset.event_hit_pmts)
    data['std_charge_per_event'].append(np.std(dataset.event_hit_charges))
    data['mean_charge_per_event'].append(np.mean(dataset.event_hit_charges))
    data['charge_sum_per_event'].append(np.sum(dataset.event_hit_charges))
    data['std_time_per_event'].append(np.std(dataset.event_hit_times))
    data['mean_time_per_event'].append(np.mean(dataset.event_hit_times))
    data['num_hits_per_event'].append(len(dataset.event_hit_pmts))

electrons = sum(data['event_label'])
num_events = len(data['event_label'])
num_hits = len(data['hit_pmt'])
gammas = num_events-electrons
print(f'\nNumber of events: {num_events}')
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

dict_to_df = {}
hits_to_df = {}
for key in data.keys():
    if key not in ['hit_charge', 'hit_time', 'hit_pmt', 'event_hits_index']:
        dict_to_df[key] = data[key]


events_df = pd.DataFrame(dict_to_df, columns=dict_to_df.keys())
events_df['event_label'] = events_df['event_label'].map({0: 'gamma', 1: 'electron'})

# Plot kernel densities
def kde_plotter_events(df):
    plt.rcParams.update({'font.size': 19})
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 16))

    df.groupby('event_label')['polar'].plot.kde(ax=axes[0, 0])
    axes[0, 0].set_ylabel('')
    axes[0, 0].set_xlabel(r'Angle $\theta$ [rad]')
    axes[0, 0].legend([r'$e$', r'$\gamma$'], loc=2)
    axes[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    df.groupby('event_label')['azimuth'].plot.kde(ax=axes[0, 1])
    axes[0, 1].set_ylabel('')
    axes[0, 1].set_xlabel(r'Angle $\varphi$ [rad]')
    axes[0, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    df.groupby('event_label')['energies'].plot.kde(ax=axes[0, 2])
    axes[0, 2].set_ylabel('')
    axes[0, 2].set_xlabel('Energy [MeV]')
    axes[0, 2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    df.groupby('event_label')['x'].plot.kde(ax=axes[0, 3])
    axes[0, 3].set_ylabel('')
    axes[0, 3].set_xlabel(r'Coordinate $x$')
    axes[0, 3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    df.groupby('event_label')['y'].plot.kde(ax=axes[1, 0])
    axes[1, 0].set_ylabel('')
    axes[1, 0].set_xlabel(r'Coordinate $y$')
    axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    df.groupby('event_label')['z'].plot.kde(ax=axes[1, 1])
    axes[1, 1].set_ylabel('')
    axes[1, 1].set_xlabel(r'Coordinate $z$')
    axes[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    df.groupby('event_label')['num_hits_per_event'].plot.kde(ax=axes[1, 2])
    axes[1, 2].set_ylabel('')
    axes[1, 2].set_xlabel(r'Number of hits per event')
    axes[1, 2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    df.groupby('event_label')['mean_charge_per_event'].plot.kde(ax=axes[2, 0])
    axes[2, 0].set_ylabel('')
    axes[2, 0].set_xlabel(r'Mean charge per event')
    axes[2, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    df.groupby('event_label')['mean_time_per_event'].plot.kde(ax=axes[2, 2])
    axes[2, 2].set_ylabel('')
    axes[2, 2].set_xlabel(r'Mean time per event [ns]')
    axes[2, 2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    df.groupby('event_label')['charge_sum_per_event'].plot.kde(ax=axes[1, 3])
    axes[1, 3].set_ylabel('')
    axes[1, 3].set_xlabel(r'Total charge per event')
    axes[1, 3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    df.groupby('event_label')['std_charge_per_event'].plot.kde(ax=axes[2, 1])
    axes[2, 1].set_ylabel('')
    axes[2, 1].set_xlabel(r'Std of charge per event')
    axes[2, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    df.groupby('event_label')['std_time_per_event'].plot.kde(ax=axes[2, 3])
    axes[2, 3].set_ylabel('')
    axes[2, 3].set_xlabel(r'Std of time per event [ns]')
    axes[2, 3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    plt.savefig('events_distrib.png', dpi=300, bbox_inches="tight")
    plt.show(block=False)
    plt.close()


def hist_plotter_hits(data):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    axes[0].hist(data['hit_charge'], histtype="stepfilled", bins=25, alpha=0.8, density=True)
    axes[1].hist(data['hit_time'], histtype="stepfilled", bins=25, alpha=0.8, density=True)
    plt.show(block=False)

kde_plotter_events(events_df)
#hist_plotter_hits(data)
