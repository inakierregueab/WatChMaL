"""
Class implementing a mPMT dataset for CNNs in h5 format
"""

# torch imports
from torch import from_numpy
from torch import flip

# generic imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset
import watchmal.dataset.data_utils as du

barrel_map_array_idxs = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 15, 16, 17, 12, 13, 14, 18]
pmts_per_mpmt = 19


class CNNmPMTDataset(H5Dataset):
    def __init__(self, h5file, mpmt_positions_file, is_distributed, transforms=None, collapse_arrays=False, pad=False,
                 mode='both', scaling=False):
        """
        Args:
            h5_path             ... path to h5 dataset file
            is_distributed      ... whether running in multiprocessing mode
            transforms          ... transforms to apply
            collapse_arrays     ... whether to collapse arrays in return
        """
        super().__init__(h5file, is_distributed)

        self.mpmt_positions = np.load(mpmt_positions_file)['mpmt_image_positions']
        self.data_size = np.max(self.mpmt_positions, axis=0) + 1
        self.barrel_rows = [row for row in range(self.data_size[0]) if
                            np.count_nonzero(self.mpmt_positions[:, 0] == row) == self.data_size[1]]
        n_channels = pmts_per_mpmt
        self.data_size = np.insert(self.data_size, 0, n_channels)
        self.collapse_arrays = collapse_arrays
        self.transforms = du.get_transformations(self, transforms)
        self.pad = pad

        self.horizontal_flip_mpmt_map = [0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 17, 16, 15, 14, 13, 18]
        self.vertical_flip_mpmt_map = [6, 5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 15, 14, 13, 12, 17, 16, 18]

        self.mode = mode

        self.scaling = scaling
        self.mu_q = 2.4899564   #np.mean(self.hdf5_hit_charge[:1000000000])
        self.mu_t = 975.42676   #np.mean(self.hdf5_hit_time[:1000000000])
        self.std_q = 6.9522057  #np.std(self.hdf5_hit_charge[:1000000000])
        self.std_t = 47.54492   #np.std(self.hdf5_hit_time[:1000000000])
        ################

    def process_data(self, hit_pmts, hit_data, data_type):
        """
        Returns event data from dataset associated with a specific index
        Args:
            hit_pmts                ... array of ids of hit pmts
            hid_data                ... array of data associated with hits
            data_type               ... str specifying charge or time data
        
        Returns:
            data                    ... array of hits in cnn format
        """
        if self.scaling:
            hit_data = self.feature_scaling(hit_data, data_type)

        hit_mpmts = hit_pmts // pmts_per_mpmt   # Mapping each PMT hit to its correspondent mPMT
        hit_pmt_in_modules = hit_pmts % pmts_per_mpmt   # Mapping each PMT to a channel (position in its mPMT)

        hit_rows = self.mpmt_positions[hit_mpmts, 0]
        hit_cols = self.mpmt_positions[hit_mpmts, 1]

        data = np.zeros(self.data_size, dtype=np.float32)
        data[hit_pmt_in_modules, hit_rows, hit_cols] = hit_data

        # fix barrel array indexing to match endcaps in xyz ordering
        barrel_data = data[:, self.barrel_rows, :]
        data[:, self.barrel_rows, :] = barrel_data[barrel_map_array_idxs, :, :]

        return data

    def __getitem__(self, item):

        data_dict = super().__getitem__(item)

        # Fix same transformation for time and charge
        rand_choice = self.fix_transformation()

        # Build charge and time images
        charge_image = self.from_data_to_image(self.event_hit_charges, rand_choice, data_type='c')
        time_image = self.from_data_to_image(self.event_hit_times, rand_choice, data_type='t')

        # Arrange charge channels according to time (first PMT activated in mPMT is in the first channel)
        t_sorted, t_indices = torch.sort(time_image, dim=0)
        processed_image = torch.zeros_like(charge_image, dtype=charge_image.dtype).scatter_(dim=0, index=t_indices.sort(dim=0)[1], src=charge_image)

        data_dict["data"] = processed_image

        """
        To plot events: 

        time_image = np.array(time_image)
        charge_image = np.array(charge_image)
        self.event_plotter(charge_image, mode='heatmap', type='charge')
        self.event_plotter(time_image, mode='heatmap', type='time')
        time_image = from_numpy(time_image)
        charge_image = from_numpy(charge_image)
        """
        return data_dict

    def fix_transformation(self):
        if self.transforms is not None:
            rand_choice = random.randint(0, len(self.transforms))
        else:
            rand_choice = 0
        return rand_choice

    def from_data_to_image(self, hit_data, rand_choice, data_type='c'):

        # Build image with data
        hit_data = from_numpy(self.process_data(self.event_hit_pmts, hit_data, data_type))

        # TODO: Now only applying a single transformation
        # Perform transformation
        if self.transforms is not None:
            hit_data = du.apply_random_transformations(self.transforms, hit_data, rand_choice)

        # Add padding
        if self.pad:
            hit_data = self.mpmtPadding(hit_data)

        # TODO: modified to Q+Ts
        # collapse arrays if desired
        if self.collapse_arrays and data_type == 't':
            mean_channel = torch.mean(hit_data, 0, keepdim=True)
            std_channel = torch.std(hit_data, 0, keepdim=True)
            hit_data = torch.cat((mean_channel, std_channel), 0)
        return hit_data

    def horizontal_flip(self, data):
        return flip(data[self.horizontal_flip_mpmt_map, :, :], [2])

    def vertical_flip(self, data):
        return flip(data[self.vertical_flip_mpmt_map, :, :], [1])

    def front_back_reflection(self, data):
        """
        Returns an image with horizontal flip of the left and right halves of the barrels and
        vertical flip of the endcaps

        :param data : torch.tensor
        :returns transform_data: torch.tensor
        """
        w = data.shape[2]
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        l_endcap_index = w // 2 - 5
        r_endcap_index = w // 2 + 4

        transform_data = data.clone()

        # Take out the left and right halves of the barrel
        left_barrel = data[:, self.barrel_rows, :w // 2]
        right_barrel = data[:, self.barrel_rows, w // 2:]
        # Horizontal flip of the left and right halves of barrel
        transform_data[:, self.barrel_rows, :w // 2] = self.horizontal_flip(left_barrel)
        transform_data[:, self.barrel_rows, w // 2:] = self.horizontal_flip(right_barrel)

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index + 1]
        bottom_endcap = data[:, barrel_row_end + 1:, l_endcap_index:r_endcap_index + 1]
        # Vertical flip of the top and bottom endcaps
        transform_data[:, :barrel_row_start, l_endcap_index:r_endcap_index + 1] = self.vertical_flip(top_endcap)
        transform_data[:, barrel_row_end + 1:, l_endcap_index:r_endcap_index + 1] = self.vertical_flip(bottom_endcap)

        return transform_data

    def rotation180(self, data):
        """
        Returns an image with horizontal and vertical flip of the endcaps and 
        shifting of the barrel rows by half the width

        :param data : torch.tensor
        :returns transform_data: torch.tensor
        """
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]  # 10,18 respectively
        l_endcap_index = data.shape[2] // 2 - 5  # 15
        r_endcap_index = data.shape[2] // 2 + 4  # 24

        transform_data = data.clone()

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index + 1]
        bottom_endcap = data[:, barrel_row_end + 1:, l_endcap_index:r_endcap_index + 1]
        # Vertical and horizontal flips of the endcaps
        transform_data[:, :barrel_row_start, l_endcap_index:r_endcap_index + 1] = self.horizontal_flip(
            self.vertical_flip(top_endcap))
        transform_data[:, barrel_row_end + 1:, l_endcap_index:r_endcap_index + 1] = self.horizontal_flip(
            self.vertical_flip(bottom_endcap))

        # Swap the left and right halves of the barrel
        transform_data[:, self.barrel_rows, :] = torch.roll(transform_data[:, self.barrel_rows, :], 20, 2)

        return transform_data

    def mpmtPadding(self, data):
        half_len_index = int(data.shape[2] / 2)
        horiz_pad_data = torch.cat((data, torch.zeros_like(data[:, :, :half_len_index])), 2)
        horiz_pad_data[:, :, 2 * half_len_index:] = torch.tensor(0, dtype=torch.float64)
        horiz_pad_data[:, self.barrel_rows, 2 * half_len_index:] = data[:, self.barrel_rows, :half_len_index]

        l_index = data.shape[2] / 2 - 1
        r_index = data.shape[2] / 2

        l_endcap_ind = int(l_index - 4)
        r_endcap_ind = int(r_index + 4)

        top_end_cap = data[:, self.barrel_rows[-1] + 1:, l_endcap_ind:r_endcap_ind + 1]
        bot_end_cap = data[:, :self.barrel_rows[0], l_endcap_ind:r_endcap_ind + 1]

        vhflip_top = self.horizontal_flip(self.vertical_flip(top_end_cap))
        vhflip_bot = self.horizontal_flip(self.vertical_flip(bot_end_cap))

        horiz_pad_data[:, self.barrel_rows[-1] + 1:,
        l_endcap_ind + int(data.shape[2] / 2): r_endcap_ind + int(data.shape[2] / 2) + 1] = vhflip_top
        horiz_pad_data[:, :self.barrel_rows[0],
        l_endcap_ind + int(data.shape[2] / 2): r_endcap_ind + int(data.shape[2] / 2) + 1] = vhflip_bot

        return horiz_pad_data

    def retrieve_event_data(self, item):
        """
        Returns event data from dataset associated with a specific index
        Args:
            item                    ... index of event
        Returns:
            hit_pmts                ... array of ids of hit pmts
            pmt_charge_data         ... array of charge of hits
            pmt_time_data           ... array of times of hits
        """
        data_dict = super().__getitem__(item)

        # construct charge data with barrel array indexing to match endcaps in xyz ordering
        pmt_charge_data = self.process_data(self.event_hit_pmts, self.event_hit_charges).flatten()

        # construct time data with barrel array indexing to match endcaps in xyz ordering
        pmt_time_data = self.process_data(self.event_hit_pmts, self.event_hit_times).flatten()

        return self.event_hit_pmts, pmt_charge_data, pmt_time_data

    def event_plotter(self, data, mode, data_type):
        """
        Collapses all channels and plots the event data

        :param data: ndarray (n_channels, 29, 40)
        :param mode: str, scatter or heatmap
        :param data_type: str, charge or time
        :return: None
        """
        if data_type == 'charge':
            cmap = 'inferno'
            title = 'Event sum charge per mPMT (unrolled tank)'
            collapsed_data = np.sum(data, axis=0)
        elif data_type == 'time':
            cmap = 'cividis'
            title = 'Average time detection per mPMT (unrolled tank)'
            collapsed_data = np.mean(data, axis=0)

        plt.figure(figsize=(14, 8))

        if mode == 'scatter':
            coords = np.where(collapsed_data > 0.)   # Threshold
            mpmt_charge = collapsed_data[coords]
            x = coords[1]
            y = coords[0]

            plt.scatter(x, y, c=mpmt_charge, s=30, cmap=cmap, vmin=0)

        elif mode == 'heatmap':
            plt.pcolor(collapsed_data, cmap=cmap)
            plt.axhline(10, color='w', linewidth=0.5)
            plt.axhline(19, color='w', linewidth=0.5)
            plt.axvline(15, color='w', linewidth=0.5)
            plt.axvline(25, color='w', linewidth=0.5)

        plt.yticks(np.arange(0, 30, 5))
        plt.xticks(np.arange(0, 40, 5))
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def feature_scaling(self, hit_array, data_type):
        """
        When joining channels Q+T they must be in the same range [0,1]. The chosen method is standarization.
        """
        # Constants computed with 1 billion hits
        if data_type == 'c':
            mu = self.mu_q
            std = self.std_q

        elif data_type == 't':
            mu = self.mu_t
            std = self.std_t

        standarized_array = (hit_array - mu)/std
        return standarized_array



