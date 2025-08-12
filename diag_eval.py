from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from matplotlib.patches import Patch

# --------- config ---------
test = pd.read_parquet('test.parquet')
model_paths = {
    24: "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/d24",
    48: "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/d48",
    72: "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/d72",
    96: "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/d96",
    120: "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/d120",
}
ordered_diags = ['CN', 'MCI', 'AD', 'Overall']  # bar order
diag_colors = {'CN': 'darkgreen', 'MCI': 'gold', 'AD': 'firebrick', 'Overall': 'slategray'}

# Figure style: sized for a Google Doc page
FIGSIZE = (6.5, 4.5)  # inches
DPI = 200

def class_f1_ovr(y_true, y_pred, label):
    """One-vs-rest F1 for a single class."""
    yt = (y_true == label).astype(int)
    yp = (y_pred == label).astype(int)
    if yt.sum() == 0 and yp.sum() == 0:
        return np.nan
    return f1_score(yt, yp)

for cp, path in model_paths.items():
    pred_model = TabularPredictor.load(path)

    y_true = test[f'd{cp}']
    y_pred = pred_model.predict(test)
    is_extra = (test[f'bd{cp}'].astype(float) == 1)

    f1_vals = {}
    p_extra = {}

    # Per-class
    for diag in ['CN', 'MCI', 'AD']:
        f1_vals[diag] = class_f1_ovr(y_true, y_pred, diag)
        mask_diag = (y_true == diag)
        denom = mask_diag.sum()
        p_extra[diag] = (is_extra & mask_diag).sum() / denom if denom > 0 else np.nan

    # Overall
    try:
        f1_overall = f1_score(y_true, y_pred, average='macro')
    except ValueError:
        f1_overall = np.nan
    f1_vals['Overall'] = f1_overall
    p_extra['Overall'] = is_extra.mean() if len(is_extra) else np.nan

    # Plot
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    x = np.arange(len(ordered_diags))
    width = 0.6

    for i, diag in enumerate(ordered_diags):
        f1h = f1_vals.get(diag, np.nan)
        frac_extra = p_extra.get(diag, np.nan)
        if np.isnan(f1h) or np.isnan(frac_extra):
            interp_h = np.nan
            extra_h = np.nan
        else:
            interp_h = f1h * max(0.0, 1.0 - frac_extra)
            extra_h  = f1h * max(0.0, frac_extra)

        ax.bar(
            x[i], interp_h, width=width,
            color=diag_colors[diag], edgecolor='black', linewidth=0.6
        )
        ax.bar(
            x[i], extra_h, width=width,
            bottom=interp_h if not np.isnan(interp_h) else None,
            color=diag_colors[diag], alpha=0.45,
            hatch='//', edgecolor='black', linewidth=0.6
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_diags)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('F1 Score')
    ax.set_title(f'Diagnosis F1 — {cp} Months')

    # Legend only if not 120 months
    if cp != 120:
        class_handles = [Patch(facecolor=diag_colors[d], edgecolor='black', label=d) for d in ['CN', 'MCI', 'AD']]
        overall_handle = Patch(facecolor=diag_colors['Overall'], edgecolor='black', label='Overall')
        extra_handle = Patch(facecolor='white', edgecolor='black', hatch='//', label='Extrapolated share')
        ax.legend(
            handles=class_handles + [overall_handle, extra_handle],
            loc='best', frameon=False, title='Legend'
        )

    plt.tight_layout()
    plt.show()
