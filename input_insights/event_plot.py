from analysis.event_analysis.EventPlotter import EventPlotter
import h5py

h5_file_path = '/data/neutrinos/IWCD_Data/IWCD_mPMT_Short_emgp0_E0to1000MeV_digihits.h5'
mpmt_positions_file_path = '/data/neutrinos/IWCD_Data/IWCDshort_mPMT_image_positions.npz'
geo_path = '/home/ierregue/ssh_tunel/data/geo_mPMTshort.npz'


# Get event index
raw_h5_file = h5py.File(h5_file_path, 'r')


datset = EventPlotter(h5_file_path, mpmt_positions_file_path, geo_path)
im = datset.display_event(index=3000)


# TODO: Modify plots in class
"""
- black background
- More spacing
- All saved to outputs (hydra)
- 3D version
"""