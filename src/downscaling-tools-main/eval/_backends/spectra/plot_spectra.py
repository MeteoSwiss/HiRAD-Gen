import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cmcrameri.cm as cmc
import warnings

from eval.paths import reference_spectra_dir


OUTPUT_DIR = "/home/ecm5702/plots/spectra"

# DATE_START = "2024-02-01 00:00:00"
# DATE_END = "2024-02-01 00:00:00"
DATE_START = "2023-08-01 00:00:00"
DATE_END = "2023-08-10 00:00:00"
STEPS_TO_PLOT = [144]
ENSEMBLE_MEMBERS = [1, 2]
DATE_FREQ = "1D"

PARAM_CONFIGS = [
    {"param": "2t", "level": "sfc", "dir_name": "2t_sfc"},
    {"param": "10u", "level": "sfc", "dir_name": "10u_sfc"},
    {"param": "10v", "level": "sfc", "dir_name": "10v_sfc"},
    {"param": "sp", "level": "sfc", "dir_name": "sp_sfc"},
    {"param": "t", "level": "850", "dir_name": "t_850"},
    {"param": "z", "level": "500", "dir_name": "z_500"},
]


def add_expver_config(n, exp_type, label=""):
    return {
        "name": n,
        "type": exp_type,
        "base_path": f"/home/ecm5702/perm/ai_spectra/{n}/spectra",
        "label": label if label else n,
    }


EXPVER_CONFIGS = []
# EXPVER_CONFIGS.append(add_expver_config("iytd", "ai", "5k steps (iytd)"))
# EXPVER_CONFIGS.append(add_expver_config("iysd", "ai", "13k steps (iysd)"))
# EXPVER_CONFIGS.append(add_expver_config("iytc", "ai", "27k steps (iytc)"))
# EXPVER_CONFIGS.append(add_expver_config("iz2p", "ai", "50k steps (iz2p)"))
# EXPVER_CONFIGS.append(add_expver_config("iz2o", "ai", "100k steps (iz2o)"))
EXPVER_CONFIGS.append(add_expver_config("j0ys", "ai", "j0ys"))
EXPVER_CONFIGS.append(add_expver_config("j1li", "ai", "j1lh"))
# EXPVER_CONFIGS.append(add_expver_config("j1li", "ai", "j1lj"))
# EXPVER_CONFIGS.append(add_expver_config("j10e", "ai", "j10e"))
# EXPVER_CONFIGS.append(add_expver_config("j10d", "ai", "j10d"))
# EXPVER_CONFIGS.append(add_expver_config("j107", "ai", "j107"))
# EXPVER_CONFIGS.append(add_expver_config("j109", "ai", "j109"))
# EXPVER_CONFIGS.append(add_expver_config("j10b", "ai", "j10b"))

# EXPVER_CONFIGS.append(add_expver_config("j0ys", "ai", "j0ys"))

# n = "j0ys"

"""
n = "ip6y"
EXPVER_CONFIGS += [
    {
        "name": n,
        "type": "ai",
        "base_path": f"/home/ecm5702/perm/ai_spectra/{n}",
        "label": f"downscaling->O320 ({n}) / low noise+finetuning",
    },
]
n = "ioj2"
EXPVER_CONFIGS += [
    {
        "name": n,
        "type": "ai",
        "base_path": f"/home/ecm5702/perm/ai_spectra/{n}",
        "label": f"downscaling->O320 ({n}) / low noise+finetuning",
    },
]
"""
n = "eefo_o96"
EXPVER_CONFIGS += [
    {
        "name": n,
        "type": "hpc",
        "base_path": str(reference_spectra_dir(n)),
        "label": "eefo O96",
    },
]

n = "enfo_o320"
EXPVER_CONFIGS += [
    {
        "name": n,
        "type": "hpc",
        "base_path": str(reference_spectra_dir(n)),
        "label": "enfo O320",
    },
]

print("Experiment versions to plot:")
for c in EXPVER_CONFIGS:
    print(f" - {c['name']} ({c['type']}), base: {c['base_path']}")
TYPE_LINESTYLE = {"ai": "-", "hpc": "--"}


r = 6.371 * 1000
k = 2 * np.pi * r


def wn2ls(x):
    x = np.array(x, float)
    m = np.isclose(x, 0)
    x[m] = np.inf
    x[~m] = k / x[~m]
    return x


def ls2wn(x):
    return np.array(x, float) / k


FIG_HEI = 3.7
FIG_FAC = 1.718
GRID = {"color": "grey", "linestyle": "--", "linewidth": 0.22}
LEG_LOC = "lower left"
LEG_FS = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)
date_list = pd.date_range(pd.Timestamp(DATE_START), pd.Timestamp(DATE_END), freq=DATE_FREQ)
expvers = EXPVER_CONFIGS
colors = [cmc.batlow(i / (len(expvers) - 1)) for i in range(len(expvers))] if expvers else []


def get_paths(conf, dir_name, date_in, step, param, level, number):
    b = conf["base_path"]
    if conf["type"] == "ai":
        w = f"{b}/{dir_name}/wvn_{date_in}_{step}_{param}_{level}_{conf['name']}_n{number}.npy"
        a = f"{b}/{dir_name}/ampl_{date_in}_{step}_{param}_{level}_{conf['name']}_n{number}.npy"
    else:
        w = f"{b}/{dir_name}/wvn_{date_in}_{step}_{param}_{level}_1_n{number}.npy"
        a = f"{b}/{dir_name}/ampl_{date_in}_{step}_{param}_{level}_1_n{number}.npy"
    return w, a


def mean_curve(curves):
    if not curves:
        raise ValueError("Cannot average an empty list of curves.")
    min_len = min(len(curve) for curve in curves)
    trimmed = [curve[:min_len] for curve in curves]
    return np.mean(trimmed, axis=0)


for cfg in PARAM_CONFIGS:
    param, level, dir_name = cfg["param"], cfg["level"], cfg["dir_name"]
    print(f"{param} {level}")
    fig, ax = plt.subplots(figsize=(FIG_HEI * FIG_FAC, FIG_HEI))
    ax2 = ax.secondary_xaxis("top", functions=(wn2ls, wn2ls))
    ax2.set_xlabel("Approximate scale [km]")
    ax2.set_xscale("log")
    any_data = False
    for ie, conf in enumerate(expvers):
        missing_counter = 0
        found_counter = 0
        step_w_curves, step_a_curves = [], []
        for step in STEPS_TO_PLOT:
            mw, mA = [], []
            for number in ENSEMBLE_MEMBERS:
                W, A = [], []
                for d in date_list:
                    date_in = d.year * 10000 + d.month * 100 + d.day
                    wfp, afp = get_paths(conf, dir_name, date_in, step, param, level, number)
                    try:
                        w = np.load(wfp)
                        a = np.load(afp)
                        W.append(w)
                        A.append(a)
                        found_counter += 1
                    except FileNotFoundError:
                        missing_counter += 1

                if W and A:
                    mw.append(np.stack(W, axis=1))
                    mA.append(np.stack(A, axis=1))
            if missing_counter > 0:
                warnings.warn(
                    f"Missing {missing_counter}/{missing_counter + found_counter} files for "
                    f"{conf['name']} (param={param}, level={level}, step={step})"
                )
            if mw and mA:
                avg_w = np.mean([arr.mean(axis=1) for arr in mw], axis=0)
                avg_a = np.mean([arr.mean(axis=1) for arr in mA], axis=0)
                step_w_curves.append(avg_w)
                step_a_curves.append(avg_a)
        if step_w_curves and step_a_curves:
            avg_w = mean_curve(step_w_curves)
            avg_a = mean_curve(step_a_curves)
            iok = range(3, len(avg_w))
            x = avg_w[iok]
            y = avg_a[iok]
            ax.plot(
                x,
                y,
                color=colors[ie],
                linestyle=TYPE_LINESTYLE.get(conf["type"], "-"),
                label=conf.get("label", conf["name"]),
            )
            any_data = True
    ax.set_yscale("log")
    ax.set_ylabel("Mean power")
    ax.set_xscale("log")
    ax.set_xlabel("Zonal wavenumber")
    if any_data:
        ax.set_xlim([x.min(), x.max() * 1.1])
    ax.grid(**GRID)
    ax.legend(loc=LEG_LOC, frameon=False, fontsize=LEG_FS)
    ax.set_title(f"{param} at level {level}")
    plt.tight_layout()
    out = f"{OUTPUT_DIR}/spectra_{param}_{level}.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()
