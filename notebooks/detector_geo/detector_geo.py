import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


#    POSITIONS, ORIENTATION AND TUBE NUMBER FOR EACH PMT IN 3D
geo = np.load('/Users/mariateresaalvarez-buhillapuig/Desktop/repositories/WatChMaL/data/geo_mPMTshort.npz')
geo_dict = {}
for item in geo.files:
    geo_dict[item] = geo[item]

positions = geo_dict['position']

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")
ax.scatter3D(positions[:,2], positions[:,0], positions[:,1], color="royalblue", s=1)
plt.title("PMTs positions", size=20)
plt.tight_layout()
plt.show(block=False)


#    UNFOLD 3D MPMTS IN A 2D IMAGE
# TODO: Review transformations, add list of indices and map, is necessary? (summary_of_knowledge)
x = positions[:,0]
y = positions[:,1]
z = positions[:,2]

top_height = max(y)
barrel_ids = np.where((y < max(y)-30) & (y > min(y)+30))[0]
x_barrel = x[barrel_ids]
y_barrel = y[barrel_ids]
z_barrel = z[barrel_ids]
phi = np.arctan2(z_barrel, x_barrel)
plt.figure(figsize=(22, 5))
plt.scatter(phi, y_barrel, s=0.5)
plt.show(block=False)

top_ids = np.where(y >= (max(y)-30))[0]
z_top = z[top_ids]
y_top = x[top_ids]
plt.scatter(z_top, y_top, s =0.5)
plt.show()

bot_ids = np.where(y <= (min(y)+30))[0]
z_bot = z[bot_ids]
y_bot = x[bot_ids]
plt.scatter(z_bot, y_bot, s =0.5)
plt.show(block=False)

y_new = np.concatenate((np.arctan2(x_barrel, z_barrel),y_bot,y_top))
x_new = np.concatenate((y_barrel,z_bot,z_top))
plt.scatter(x_new, y_new, s=0.5)
plt.show(block=False)