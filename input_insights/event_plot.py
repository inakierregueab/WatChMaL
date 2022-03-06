from analysis.event_analysis.EventPlotter import EventPlotter


h5_file_path = '../data/IWCD_mPMT_Short_emgp0_E0to1000MeV_digihits_mini.h5'
mpmt_positions_file_path = '../data/IWCDshort_mPMT_image_positions.npz'
geo_path = '../data/geo_mPMTshort.npz'

datset = EventPlotter(h5_file_path, mpmt_positions_file_path, geo_path)
im = datset.display_event(index=655) # max charge sum

# TODO: Modify plots in class
"""
- black background
- More spacing
- All saved to outputs (hydra)
- 3D version
"""