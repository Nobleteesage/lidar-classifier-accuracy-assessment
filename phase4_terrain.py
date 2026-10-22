import csv
import numpy as np
import laspy
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF = "data/crop.laz"
PRED = "output/crop_classified.laz"
CELL = 2.0
BLOCK = 20.0

ref = laspy.read(REF)
pred = laspy.read(PRED)
x = np.asarray(ref.x)
y = np.asarray(ref.y)
z = np.asarray(ref.z)
if len(x) != len(pred.x) or not np.allclose(x, np.asarray(pred.x)) or not np.allclose(y, np.asarray(pred.y)):
    raise SystemExit("Files are not in the same point order, stopping.")
rc = np.asarray(ref.classification)
pc = np.asarray(pred.classification)
rg = rc == 2
pg = pc == 2

x0, y0 = x.min(), y.min()
nx = int(np.ceil((x.max() - x0) / CELL)) + 1
ny = int(np.ceil((y.max() - y0) / CELL)) + 1
ix = ((x - x0) / CELL).astype(int)
iy = ((y - y0) / CELL).astype(int)


def min_grid(mask):
    g = np.full((ny, nx), np.inf)
    np.minimum.at(g, (iy[mask], ix[mask]), z[mask])
    g[np.isinf(g)] = np.nan
    return g


g_ref = min_grid(rg)
g_pred = min_grid(pg)
obs_ref = ~np.isnan(g_ref)
obs_pred = ~np.isnan(g_pred)
total = nx * ny
print("TERRAIN CELLS (2 m cells), grid:", nx, "x", ny)
print(f"  Cells with USGS ground points:  {obs_ref.sum()} ({100 * obs_ref.sum() / total:.1f}%)")
print(f"  Cells with model ground points: {obs_pred.sum()} ({100 * obs_pred.sum() / total:.1f}%)")
print(f"  Cells USGS has but model lacks: {(obs_ref & ~obs_pred).sum()}")


def stats(d):
    d = d[~np.isnan(d)]
    return {"n": int(len(d)), "mean": float(d.mean()), "mae": float(np.abs(d).mean()),
            "rmse": float(np.sqrt((d ** 2).mean())), "p90abs": float(np.percentile(np.abs(d), 90)),
            "gt1": float((np.abs(d) > 1).mean() * 100), "gt2": float((np.abs(d) > 2).mean() * 100)}


def show(title, s):
    print(title)
    print(f"  cells: {s['n']}  mean error: {s['mean']:.2f} m  MAE: {s['mae']:.2f} m  RMSE: {s['rmse']:.2f} m")
    print(f"  90th percentile abs error: {s['p90abs']:.2f} m  cells over 1 m: {s['gt1']:.1f}%  over 2 m: {s['gt2']:.1f}%")


both = obs_ref & obs_pred
s_both = stats(g_pred[both] - g_ref[both])
show("\nCELLS WHERE BOTH HAVE GROUND POINTS (no interpolation, model minus USGS)", s_both)

gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))


def fill(g):
    m = ~np.isnan(g)
    pts = np.column_stack([gx[m], gy[m]])
    lin = griddata(pts, g[m], (gx, gy), method="linear")
    near = griddata(pts, g[m], (gx, gy), method="nearest")
    return np.where(np.isnan(lin), near, lin)


dtm_ref = fill(g_ref)
dtm_pred = fill(g_pred)
diff = dtm_pred - dtm_ref
s_all = stats(diff.ravel())
show("\nFULL TERRAIN MODELS AFTER FILLING GAPS (model minus USGS)", s_all)
holes = obs_ref & ~obs_pred
s_holes = stats(diff[holes])
show("\nONLY CELLS WHERE THE MODEL HAD NO GROUND (filled by interpolation)", s_holes)

hag = z - dtm_ref[iy, ix]
print("\nPREDICTED CLASSES 14 AND 15 SPLIT BY WHAT USGS SAYS")
for c in (14, 15):
    m = pc == c
    ing = int((m & rg).sum())
    ong = int((m & ~rg).sum())
    tot = ing + ong
    print(f"  Class {c}: {tot} points, {ing} are USGS ground ({100 * ing / tot:.1f}%), {ong} are not")
    h = hag[m & ~rg]
    if len(h):
        print(f"    height of the non ground ones: p10 {np.percentile(h, 10):.1f}  median {np.median(h):.1f}  p90 {np.percentile(h, 90):.1f} m")

bx = ((x - x0) / BLOCK).astype(int)
by = ((y - y0) / BLOCK).astype(int)
nbx, nby = bx.max() + 1, by.max() + 1
ref_cnt = np.zeros((nby, nbx))
tp_cnt = np.zeros((nby, nbx))
np.add.at(ref_cnt, (by[rg], bx[rg]), 1)
np.add.at(tp_cnt, (by[rg & pg], bx[rg & pg]), 1)
valid = ref_cnt >= 50
recall = np.where(valid, tp_cnt / np.maximum(ref_cnt, 1), np.nan)
rv = recall[valid]
print("\nHOW BLOCK LIKE ARE THE MISSES (20 m blocks with at least 50 USGS ground points)")
print(f"  Blocks: {len(rv)}")
print(f"  Ground recall under 0.2 (almost all missed): {100 * (rv < 0.2).mean():.1f}% of blocks")
print(f"  Ground recall between 0.2 and 0.8:           {100 * ((rv >= 0.2) & (rv <= 0.8)).mean():.1f}% of blocks")
print(f"  Ground recall over 0.8 (mostly found):       {100 * (rv > 0.8).mean():.1f}% of blocks")

with open("output/phase4_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["measure", "n", "mean_m", "mae_m", "rmse_m", "p90abs_m", "pct_over_1m", "pct_over_2m"])
    for name, s in [("both_observed", s_both), ("filled_full", s_all), ("model_gap_cells", s_holes)]:
        w.writerow([name, s["n"], s["mean"], s["mae"], s["rmse"], s["p90abs"], s["gt1"], s["gt2"]])

vmin = np.nanmin(dtm_ref)
vmax = np.nanmax(dtm_ref)
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
panels = [
    (axes[0, 0], dtm_ref, "terrain", vmin, vmax, "Terrain from USGS ground", "Elevation (m)"),
    (axes[0, 1], dtm_pred, "terrain", vmin, vmax, "Terrain from model ground", "Elevation (m)"),
    (axes[1, 0], diff, "RdBu_r", -5, 5, "Model minus USGS terrain", "Difference (m)"),
    (axes[1, 1], recall, "viridis", 0, 1, "Ground recall per 20 m block", "Recall"),
]
for ax, arr, cmap, lo, hi, title, label in panels:
    im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label=label, shrink=0.8)
plt.tight_layout()
plt.savefig("data/phase4_figure.png", dpi=100)
print("\nSaved data/phase4_figure.png and output/phase4_metrics.csv")
