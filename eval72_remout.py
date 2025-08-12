from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

test = pd.read_parquet('test.parquet')

# model paths
ba72 = "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/ba72"
d72  = "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/d72"

# 72 months — brain age
ba72predictor = TabularPredictor.load(ba72)

# raw eval
_ = ba72predictor.evaluate(test)  # AG's internal
y_true = test['p72'].astype(float)
y_pred = ba72predictor.predict(test).astype(float)
rmse  = np.sqrt(((y_pred - y_true)**2).mean())
mae   = np.abs(y_pred - y_true).mean()
print(f"[WITH outlier] BA@72 — RMSE {rmse:.2f} mo | MAE {mae:.2f} mo")

# --- remove worst outlier by absolute residual ---
resid   = (y_pred.values - y_true.values)
out_idx = int(np.argmax(np.abs(resid)))     # index (in current DataFrame order)
keep    = np.ones(len(test), dtype=bool)
keep[out_idx] = False

# report the removed point
try:
    removed_row = test.iloc[out_idx]
    print(f"Removing worst point: PTID={removed_row.get('PTID', 'NA')}, "
          f"true p72={y_true.iloc[out_idx]:.2f}, pred={y_pred.iloc[out_idx]:.2f}, "
          f"residual={resid[out_idx]:.2f}")
except Exception:
    pass

# filtered data / metrics
test_f   = test.iloc[keep]
y_true_f = y_true.iloc[keep]
y_pred_f = y_pred.iloc[keep]

rmse_f = np.sqrt(((y_pred_f - y_true_f)**2).mean())
mae_f  = np.abs(y_pred_f - y_true_f).mean()
print(f"[NO outlier]  BA@72 — RMSE {rmse_f:.2f} mo | MAE {mae_f:.2f} mo")

# ----- plotting (filtered) -----
diag_colors = {'CN': 'darkgreen', 'MCI': 'gold', 'AD': 'firebrick'}

is_extra_f  = (test_f['bp72'].astype(float) == 1)
is_interp_f = ~is_extra_f

plt.figure(figsize=(8, 6))
diagnoses = ['CN','MCI','AD']  # fixed legend order; plot even if some are absent

for diag in diagnoses:
    mask_diag = (test_f['d72'] == diag)

    # interpolated (filled circles)
    m_interp = mask_diag & is_interp_f
    if m_interp.any():
        plt.scatter(
            y_true_f[m_interp], y_pred_f[m_interp],
            alpha=0.85, s=30, marker='o',
            color=diag_colors.get(diag, 'gray')
        )

    # extrapolated (hollow triangles)
    m_extra = mask_diag & is_extra_f
    if m_extra.any():
        plt.scatter(
            y_true_f[m_extra], y_pred_f[m_extra],
            alpha=0.95, s=60, marker='^',
            facecolors='none',
            edgecolors=diag_colors.get(diag, 'gray'),
            linewidths=1.1
        )

# identity line
m = float(np.nanmax([y_true_f.max(), y_pred_f.max(), 1]))
plt.plot([0, m], [0, m], '--', color='gray')

plt.xlabel('Predicted Brain Age at 72 Given MRI')
plt.ylabel('Predicted Brain Age at 72 Given Historical Progression and Baseline Scan')
plt.title('Brain Age at 72 months (outlier removed)')

# legends (fixed order)
diag_handles = [Line2D([0],[0], marker='o', color='none',
                       markerfacecolor=diag_colors[d], markeredgecolor=diag_colors[d],
                       markersize=7, label=d) for d in diagnoses]

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

# residuals (filtered)
plt.figure(figsize=(6,6))
plt.hist(y_pred_f - y_true_f, bins='fd', color='moccasin', edgecolor='peru')
plt.title('Residuals BA72 (outlier removed)')
plt.xlabel('months')
plt.tight_layout()
plt.show()
