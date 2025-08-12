from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

test = pd.read_parquet('test.parquet')

# model paths
ba120 = "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/ba120"

# load & predict
ba120predictor = TabularPredictor.load(ba120)
y_true_all = test['p120'].astype(float)
y_pred_all = ba120predictor.predict(test).astype(float)

# --- remove the 4 largest absolute residual outliers ---
resid_all = y_pred_all - y_true_all
n_outliers = 4
outlier_idx = np.argsort(np.abs(resid_all))[-n_outliers:]  # worst offenders
mask_keep = np.ones(len(test), dtype=bool)
mask_keep[outlier_idx] = False

# filtered sets
y_true = y_true_all[mask_keep]
y_pred = y_pred_all[mask_keep]
test_f = test.loc[mask_keep].copy()

# metrics
rmse = np.sqrt(((y_pred - y_true)**2).mean())
mae  = np.abs(y_pred - y_true).mean()
print(f"BA@120 (without {n_outliers} extreme outliers) — RMSE {rmse:.2f} mo | MAE {mae:.2f} mo")

# color map
diag_colors = {'CN': 'darkgreen', 'MCI': 'gold', 'AD': 'firebrick'}

# extrapolation flags
is_extra  = (test_f['bp120'].astype(float) == 1)
is_interp = ~is_extra

# plot
plt.figure(figsize=(8, 6))
diagnosis_order = ['CN', 'MCI', 'AD']

for diag in diagnosis_order:
    md = (test_f['d120'] == diag)

    m_interp = md & is_interp
    if m_interp.any():
        plt.scatter(
            y_true[m_interp], y_pred[m_interp],
            alpha=0.85, s=30, marker='o',
            color=diag_colors.get(diag, 'gray')
        )

    m_extra = md & is_extra
    if m_extra.any():
        plt.scatter(
            y_true[m_extra], y_pred[m_extra],
            alpha=0.95, s=60, marker='^',
            facecolors='none',
            edgecolors=diag_colors.get(diag, 'gray'),
            linewidths=1.1
        )

# identity line
m = float(np.nanmax([y_true.max(), y_pred.max(), 1]))
plt.plot([0, m], [0, m], '--', color='gray')

plt.xlabel('Predicted Brain Age at 120 Given MRI')
plt.ylabel('Predicted Brain Age at 120 Given Historical Progression and Baseline Scan')
plt.title('Brain Age at 120 months (outliers removed)')

# legends
diag_handles = [Line2D([0],[0], marker='o', color='none',
                       markerfacecolor=diag_colors[d], markeredgecolor=diag_colors[d],
                       markersize=7, label=d) for d in diagnosis_order]
style_handles = [
    Line2D([0],[0], marker='o', color='none',
           markerfacecolor='gray', markeredgecolor='gray',
           markersize=7, label='Interpolated'),
    Line2D([0],[0], marker='^', color='gray',
           markerfacecolor='none', markeredgecolor='gray',
           markersize=8, label='Extrapolated')
]
first_legend = plt.legend(handles=diag_handles, title="Diagnosis", loc='upper left', frameon=False)
plt.gca().add_artist(first_legend)
plt.legend(handles=style_handles, title="Label source", loc='lower right', frameon=False)

plt.tight_layout()
plt.show()

# (optional) show which rows were removed
removed = test.iloc[outlier_idx][['PTID','d120','bp120','p120']]
print("Removed outliers (by |residual|):")
print(removed)


# residuals
plt.figure(figsize=(6,6))
plt.hist(y_pred - y_true, bins='fd', color='moccasin', edgecolor='peru')
plt.title('Residuals BA120 (outliers removed)')
plt.xlabel('months')
plt.tight_layout()
plt.show()
