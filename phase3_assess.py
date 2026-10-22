import csv
import numpy as np
import laspy
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF = "data/crop.laz"
PRED = "output/crop_classified.laz"
names = {1: "Unclassified", 2: "Ground", 5: "High vegetation", 6: "Building",
         14: "Wire conductor (ASPRS 14)", 15: "Transmission tower (ASPRS 15)"}

ref = laspy.read(REF)
pred = laspy.read(PRED)


def keys(las):
    return np.stack([
        np.round(np.asarray(las.x) * 100).astype(np.int64),
        np.round(np.asarray(las.y) * 100).astype(np.int64),
        np.round(np.asarray(las.z) * 100).astype(np.int64),
    ], axis=1)


kr, kp = keys(ref), keys(pred)
if len(kr) != len(kp):
    raise SystemExit("Point counts differ: " + str(len(kr)) + " vs " + str(len(kp)))
if np.array_equal(kr, kp):
    order_r = np.arange(len(kr))
    order_p = order_r
    print("Points are in the same order in both files.")
else:
    order_r = np.lexsort((kr[:, 2], kr[:, 1], kr[:, 0]))
    order_p = np.lexsort((kp[:, 2], kp[:, 1], kp[:, 0]))
    if not np.array_equal(kr[order_r], kp[order_p]):
        raise SystemExit("The two files do not contain the same points, stopping.")
    print("Points matched after sorting.")

ref_cls = np.asarray(ref.classification)[order_r]
pred_cls = np.asarray(pred.classification)[order_p]
x = np.asarray(ref.x)[order_r]
y = np.asarray(ref.y)[order_r]
z = np.asarray(ref.z)[order_r]
n = len(z)

r = ref_cls == 2
p = pred_cls == 2
tp = int((r & p).sum())
fp = int((~r & p).sum())
fn = int((r & ~p).sum())
tn = int((~r & ~p).sum())
acc = (tp + tn) / n
prec = tp / (tp + fp)
rec = tp / (tp + fn)
f1 = 2 * prec * rec / (prec + rec)
iou_g = tp / (tp + fp + fn)
iou_n = tn / (tn + fp + fn)
miou = (iou_g + iou_n) / 2
pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
kappa = (acc - pe) / (1 - pe)

print("\nGROUND VS NON GROUND (reference = USGS labels)")
print("  Points compared:", n)
print("  Ground found correctly (TP):", tp)
print("  Non ground called ground (FP):", fp)
print("  Ground the model missed (FN):", fn)
print("  Non ground called non ground (TN):", tn)
print(f"  Overall accuracy: {acc:.4f}")
print(f"  Ground precision: {prec:.4f}")
print(f"  Ground recall:    {rec:.4f}")
print(f"  Ground F1:        {f1:.4f}")
print(f"  IoU ground:       {iou_g:.4f}")
print(f"  IoU non ground:   {iou_n:.4f}")
print(f"  Mean IoU:         {miou:.4f}")
print(f"  Cohen kappa:      {kappa:.4f}")

pcls = sorted(set(pred_cls.tolist()))
print("\nWHAT THE MODEL CALLED EACH REFERENCE CLASS (rows = USGS, columns = model class)")
print(f"{'':<20}" + "".join(f"{c:>10}" for c in pcls))
for code, label in [(1, "USGS unclassified"), (2, "USGS ground")]:
    m = ref_cls == code
    print(f"{label:<20}" + "".join(f"{int((m & (pred_cls == c)).sum()):>10}" for c in pcls))

cell = 2.0
x0, y0 = x.min(), y.min()
nx = int(np.ceil((x.max() - x0) / cell)) + 1
ny = int(np.ceil((y.max() - y0) / cell)) + 1
ix = ((x - x0) / cell).astype(int)
iy = ((y - y0) / cell).astype(int)
gm = np.full((ny, nx), np.inf)
np.minimum.at(gm, (iy[r], ix[r]), z[r])
empty = np.isinf(gm)
idx_near = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
gm = gm[tuple(idx_near)]
hag = z - gm[iy, ix]

print("\nHEIGHT ABOVE LOCAL REFERENCE GROUND BY MODEL CLASS (metres, approximate)")
print(f"{'class':<34}{'count':>9}{'p10':>8}{'median':>8}{'p90':>8}")
for c in pcls:
    h = hag[pred_cls == c]
    print(f"{str(c) + ' ' + names.get(c, 'other'):<34}{len(h):>9}"
          f"{np.percentile(h, 10):>8.1f}{np.median(h):>8.1f}{np.percentile(h, 90):>8.1f}")

with open("output/phase3_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    for k, v in [("points", n), ("TP", tp), ("FP", fp), ("FN", fn), ("TN", tn),
                 ("accuracy", acc), ("precision_ground", prec), ("recall_ground", rec),
                 ("f1_ground", f1), ("iou_ground", iou_g), ("iou_nonground", iou_n),
                 ("mean_iou", miou), ("kappa", kappa)]:
        w.writerow([k, v])

rng = np.random.default_rng(0)
sel = rng.choice(n, size=min(250000, n), replace=False)
colors = {1: "#999999", 2: "#8c6d31", 5: "#2ca02c", 6: "#d62728", 14: "#1f77b4", 15: "#9467bd"}
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
ax = axes[0]
for c in pcls:
    m = sel[pred_cls[sel] == c]
    ax.scatter(x[m], y[m], s=0.3, c=colors.get(c, "black"), label=str(c) + " " + names.get(c, "other"))
ax.legend(markerscale=15, loc="lower left", fontsize=8)
ax.set_title("Model classes")
ax2 = axes[1]
ok = sel[r[sel] == p[sel]]
fpi = sel[(~r[sel]) & p[sel]]
fni = sel[r[sel] & (~p[sel])]
ax2.scatter(x[ok], y[ok], s=0.3, c="#cccccc", label="Agrees with USGS")
ax2.scatter(x[fpi], y[fpi], s=0.6, c="#d62728", label="Model ground, USGS not ground")
ax2.scatter(x[fni], y[fni], s=0.6, c="#1f77b4", label="USGS ground, model missed")
ax2.legend(markerscale=12, loc="lower left", fontsize=8)
ax2.set_title("Ground agreement")
for a in axes:
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.savefig("data/phase3_figure.png", dpi=110)
print("\nSaved data/phase3_figure.png and output/phase3_metrics.csv")
