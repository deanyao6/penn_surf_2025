from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

test = pd.read_parquet('test.parquet')

# model paths
ba120 = "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/ba120"

# load model + predict
ba120predictor = TabularPredictor.load(ba120)

y_true = test['p120'].astype(float)
y_pred = ba120predictor.predict(test).astype(float)

rmse = np.sqrt(((y_pred - y_true)**2).mean())
mae  = np.abs(y_pred - y_true).mean()
print(f"BA@120 distillation — RMSE {rmse:.2f} mo | MAE {mae:.2f} mo")

# Fixed diagnosis color map
diag_colors = {'CN': 'darkgreen', 'MCI': 'gold', 'AD': 'firebrick'}

# extrapolation flag at 120: 1 = extrapolated, 0 = interpolated
is_extra  = (test['bp120'].astype(float) == 1)
is_interp = ~is_extra

# scatter: diagnosis colors + marker by label source
plt.figure(figsize=(8, 6))
diagnosis_order = ['CN', 'MCI', 'AD']

for diag in diagnosis_order:
    mask_diag = (test['d120'] == diag)

    # interpolated (filled circle)
    m_interp = mask_diag & is_interp
    if m_interp.any():
        plt.scatter(
            y_true[m_interp], y_pred[m_interp],
            alpha=0.85, s=30, marker='o',
            color=diag_colors.get(diag, 'gray')
        )

    # extrapolated (hollow triangle)
    m_extra = mask_diag & is_extra
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
plt.title('Brain Age at 120 months')

# legends (fixed order)
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

# residuals
plt.figure(figsize=(6,6))
plt.hist(y_pred - y_true, bins='fd', color='moccasin', edgecolor='peru')
plt.title('Residuals BA120')
plt.xlabel('months')
plt.tight_layout()
plt.show()
