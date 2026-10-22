import sys
import numpy as np
import laspy

src, dst = sys.argv[1], sys.argv[2]
side, fx, fy = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
las = laspy.read(src)
xmin, ymin, _ = las.header.mins
xmax, ymax, _ = las.header.maxs
cx = xmin + fx * (xmax - xmin)
cy = ymin + fy * (ymax - ymin)
half = side / 2
x = np.asarray(las.x)
y = np.asarray(las.y)
cls = np.asarray(las.classification)
mask = ((x >= cx - half) & (x <= cx + half) & (y >= cy - half) & (y <= cy + half)
        & (cls != 7) & (cls != 18))
las.points = las.points[mask]
las.write(dst)
print("Kept points:", int(mask.sum()))
vals, counts = np.unique(np.asarray(las.classification), return_counts=True)
for v, c in zip(vals, counts):
    print(f"  class {int(v)}: {c}")
