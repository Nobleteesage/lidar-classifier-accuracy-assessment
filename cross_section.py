import sys
import numpy as np
import laspy
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fy = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
WIDTH = 20.0
CELL = 2.0

ref = laspy.read("data/crop.laz")
pred = laspy.read("output/crop_classified.laz")
x = np.asarray(ref.x)
y = np.asarray(ref.y)
z = np.asarray(ref.z)
rc = np.asarray(ref.classification)
pc = np.asarray(pred.classification)

x0, y0 = x.min(), y.min()
nx = int(np.ceil((x.max() - x0) / CELL)) + 1
ny = int(np.ceil((y.max() - y0) / CELL)) + 1
ix = ((x - x0) / CELL).astype(int)
iy = ((y - y0) / CELL).astype(int)
g = np.full((ny, nx), np.inf)
rg = rc == 2
np.minimum.at(g, (iy[rg], ix[rg]), z[rg])
empty = np.isinf(g)
idx = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
g = g[tuple(idx)]
hag = z - g[iy, ix]

yc = y0 + fy * (y.max() - y0)
m = np.abs(y - yc) <= WIDTH / 2
xs = x[m] - x0
h = hag[m]
print("Strip centre fy =", fy, " points in strip:", int(m.sum()))

names = {1: "1 Unclassified", 2: "2 Ground", 5: "5 High vegetation", 6: "6 Building",
         14: "14 Wire conductor", 15: "15 Transmission tower"}
colors = {1: "#999999", 2: "#8c6d31", 5: "#2ca02c", 6: "#d62728", 14: "#1f77b4", 15: "#9467bd"}

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for c in sorted(set(pc[m].tolist())):
    k = pc[m] == c
    axes[0].scatter(xs[k], h[k], s=1.5, c=colors.get(c, "black"), label=names.get(c, str(c)))
axes[0].legend(markerscale=6, loc="upper right", fontsize=8, ncol=3)
axes[0].set_title("Model classes, side view of a 20 m strip")
axes[0].set_ylabel("Height above USGS ground (m)")

kg = rc[m] == 2
axes[1].scatter(xs[~kg], h[~kg], s=1.5, c="#2ca02c", label="USGS not ground")
axes[1].scatter(xs[kg], h[kg], s=1.5, c="#8c6d31", label="USGS ground")
axes[1].legend(markerscale=6, loc="upper right", fontsize=8)
axes[1].set_title("USGS labels, same strip")
axes[1].set_ylabel("Height above USGS ground (m)")
axes[1].set_xlabel("Distance along strip (m)")
for a in axes:
    a.set_ylim(-3, 35)
plt.tight_layout()
out = "data/cross_section_fy" + str(fy).replace(".", "") + ".png"
plt.savefig(out, dpi=100)
print("Saved", out)
