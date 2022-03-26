import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy
import h5py


# LOAD FILES:
# Event & Hit
#h5_file_path = '../data/IWCD_mPMT_Short_emgp0_E0to1000MeV_digihits_mini.h5'
#raw_h5_file = h5py.File(h5_file_path, 'r')
#num_events = len(raw_h5_file.get('event_ids'))
# mPMTS positions

#    POSITIONS OF MPMTS IN THE 2D IMAGE
positions = np.load('/data/neutrinos/IWCD_Data/IWCDshort_mPMT_image_positions.npz')
positions_df = pd.DataFrame(positions['mpmt_image_positions'])
plt.figure(figsize=(14, 8))
ax = positions_df.plot.scatter(x=1, y=0)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_ylim(top=30)
ax.set_title('mPMTs positions (unrolled cylinder)')
plt.show()


#       POSITIONS, ORIENTATION AND TUBE NUMBER FOR EACH PMT IN 3D
geo = np.load('/home/ierregue/ssh_tunel/data/geo_mPMTshort.npz')
geo_dict = {}
for item in geo.files:
    geo_dict[item] = geo[item]

positions = geo_dict['position']
# Creating figure
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")

# Creating plot
ax.scatter3D(positions[:,2], positions[:,0], positions[:,1], color="royalblue", s=1)
plt.title("PMTs positions", size=20)

plt.tight_layout()

# show plot
plt.show()



#       DATASET
dataset = {}
for feature in raw_h5_file.keys():
    dataset[feature] = np.array(raw_h5_file.get(feature))

# Add number of hits of every event
total_hits = [len(dataset['hit_pmt'])]
hits = dataset['event_hits_index']
hits_shifted = hits[1:]
hits_shifted = np.append(hits_shifted, total_hits)
dataset['num_hits'] = hits_shifted - hits

# Add total charge sum and time interval
charge_sum = []
time_intervals = []
initial_hit = 0
for final_hit in hits_shifted:
    charge_sum.append(np.sum(dataset['hit_charge'][initial_hit:final_hit]))
    time_values = dataset['hit_time'][initial_hit:final_hit]
    if len(time_values) == 0:   #TODO: outlier?
        time_intervals.append(0)
    else:
        time_intervals.append(np.max(time_values)-np.min(time_values))
    initial_hit = final_hit

dataset['charge_sum'] = np.array(charge_sum)
dataset['time_interval'] = np.array(time_intervals)

# Features:
event_dict = {}
undesirable_cols = ['hit_charge', 'hit_time', 'hit_pmt', 'event_ids', 'root_files']
for feature in dataset.keys():
    if feature not in undesirable_cols:
        if feature == 'angles':
            event_dict['polar'] = np.array(raw_h5_file.get(feature))[:, 0].reshape(num_events)
            event_dict['azimuth'] = np.array(raw_h5_file.get(feature))[:, 1].reshape(num_events)
        elif feature == 'positions':
            event_dict['x'] = np.array(raw_h5_file.get(feature))[:, :, 0].reshape(num_events)
            event_dict['y'] = np.array(raw_h5_file.get(feature))[:, :, 1].reshape(num_events)
            event_dict['z'] = np.array(raw_h5_file.get(feature))[:, :, 2].reshape(num_events)
        else:
            event_dict[feature] = dataset[feature].reshape(num_events)
events_df = pd.DataFrame(event_dict)

# KDE plot
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18,18))
events_df.groupby('labels')['polar'].plot.kde(ax=axes[0, 0])
axes[0,0].set_ylabel('')
axes[0,0].set_xlabel(r'$\theta$')
axes[0,0].legend()
events_df.groupby('labels')['azimuth'].plot.kde(ax=axes[0, 1])
axes[0,1].set_ylabel('')
axes[0,1].set_xlabel(r'$\varphi$')
axes[0,1].legend()
events_df.groupby('labels')['energies'].plot.kde(ax=axes[0, 2])
axes[0,2].set_ylabel('')
axes[0,2].set_xlabel(r'$E$')
axes[0,2].legend()

events_df.groupby('labels')['x'].plot.kde(ax=axes[1, 0])
axes[1,0].set_ylabel('')
axes[1,0].set_xlabel(r'$x$')
axes[1,0].legend()
events_df.groupby('labels')['y'].plot.kde(ax=axes[1, 1])
axes[1,1].set_ylabel('')
axes[1,1].set_xlabel(r'$y$')
axes[1,1].legend()
events_df.groupby('labels')['z'].plot.kde(ax=axes[1, 2])
axes[1,2].set_ylabel('')
axes[1,2].set_xlabel(r'$z$')
axes[1,2].legend()

events_df.groupby('labels')['num_hits'].plot.kde(ax=axes[2, 0])
axes[2,0].set_ylabel('')
axes[2,0].set_xlabel('num_hits')
axes[2,0].legend()
events_df.groupby('labels')['charge_sum'].plot.kde(ax=axes[2, 1])
axes[2,1].set_ylabel('')
axes[2,1].set_xlabel('charge_sum')
axes[2,1].legend()
events_df.groupby('labels')['time_interval'].plot.kde(ax=axes[2, 2])
axes[2,2].set_ylabel('')
axes[2,2].set_xlabel('time_interval')
axes[2,2].legend()

# plt.show(block=False)
plt.savefig('./../../outputs/analysis/kde_features_label')

# Feature correlation
pd.plotting.scatter_matrix(events_df.drop(columns=['veto', 'veto2', 'labels', 'event_hits_index']), alpha=0.2, figsize=(15, 15), diagonal='kde')
plt.show(block=False)
plt.savefig('./../../outputs/analysis/features_correlation _matrix.png')