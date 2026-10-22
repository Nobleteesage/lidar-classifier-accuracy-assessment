import sys
import numpy as np
import laspy

path = sys.argv[1]
las = laspy.read(path)
h = las.header
n = h.point_count
xmin, ymin, zmin = h.mins
xmax, ymax, zmax = h.maxs
print("File:", path)
print("Points:", n)
print("X range:", xmin, xmax, "width:", round(xmax - xmin, 1))
print("Y range:", ymin, ymax, "height:", round(ymax - ymin, 1))
print("Z range:", zmin, zmax)
area = (xmax - xmin) * (ymax - ymin)
print("Approx density (points per square unit):", round(n / area, 2))
try:
    crs = h.parse_crs()
    print("CRS:", crs.name)
    print("Axis units:", [a.unit_name for a in crs.axis_info])
except Exception as e:
    print("Could not read CRS:", e)
vals, counts = np.unique(las.classification, return_counts=True)
names = {0: "never classified", 1: "unclassified", 2: "ground", 3: "low vegetation",
         4: "medium vegetation", 5: "high vegetation", 6: "building",
         7: "low point (noise)", 9: "water", 17: "bridge deck", 18: "high noise"}
print("Classification counts:")
for v, c in zip(vals, counts):
    print(f"  {int(v):>3}  {names.get(int(v), 'other'):<20} {c:>12}  {100 * c / n:5.1f}%")
