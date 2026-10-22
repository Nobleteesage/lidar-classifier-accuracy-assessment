import sys
import numpy as np
import laspy
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1]
cell = 2.0
las = laspy.read(path)
x = np.asarray(las.x)
y = np.asarray(las.y)
z = np.asarray(las.z)
cls = np.asarray(las.classification)
xmin, ymin, _ = las.header.mins
xmax, ymax, _ = las.header.maxs
nx = int(np.ceil((xmax - xmin) / cell))
ny = int(np.ceil((ymax - ymin) / cell))
valid = (cls != 7) & (cls != 18)
ix = np.clip(((x - xmin) / cell).astype(int), 0, nx - 1)
iy = np.clip(((y - ymin) / cell).astype(int), 0, ny - 1)
g = valid & (cls == 2)
dtm = np.full((ny, nx), np.inf)
np.minimum.at(dtm, (iy[g], ix[g]), z[g])
dsm = np.full((ny, nx), -np.inf)
np.maximum.at(dsm, (iy[valid], ix[valid]), z[valid])
empty = np.isinf(dtm)
if empty.any():
    idx = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
    dtm = dtm[tuple(idx)]
dsm[np.isinf(dsm)] = np.nan
chm = np.clip(dsm - dtm, 0, 40)
fig, ax = plt.subplots(figsize=(9, 9))
im = ax.imshow(chm, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
fig.colorbar(im, ax=ax, label="Height above ground (m)", shrink=0.8)
ticks = np.arange(0, 1.01, 0.1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.grid(True, color="white", alpha=0.4, linewidth=0.5)
ax.set_xlabel("fx (x position as a fraction of the tile)")
ax.set_ylabel("fy (y position as a fraction of the tile)")
ax.set_title("Height above ground, whole tile (2 m cells)")
plt.savefig("data/overview_chm.png", dpi=110, bbox_inches="tight")
print("Grid:", nx, "x", ny, "cells. Saved data/overview_chm.png")
