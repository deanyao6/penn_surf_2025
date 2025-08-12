from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

test = pd.read_parquet('test.parquet')

# model paths
ba96 = "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/ba96"

# load model
ba96predictor = TabularPredictor.load(ba96)

# predictions
y_true = test['p96'].astype(float)
y_pred = ba96predictor.predict(test).astype(float)

# remove outlier based on residuals
residuals = y_pred - y_true
rmse_full = np.sqrt(((y_pred - y_true)**2).mean())
mask_no_outlier = np.abs(residuals) < 3 * rmse_full  # keep within ±3×RMSE

y_true = y_true[mask_no_outlier]
y_pred = y_pred[mask_no_outlier]
test   = test[mask_no_outlier]

# recalc metrics without outlier
rmse = np.sqrt(((y_pred - y_true)**2).mean())
mae  = np.abs(y_pred - y_true).mean()
print(f"BA@96 (no outlier) — RMSE {rmse:.2f} mo | MAE {mae:.2f} mo")

# fixed diagnosis colors
diag_colors = {'CN': 'darkgreen', 'MCI': 'gold', 'AD': 'firebrick'}

# extrapolation flags
is_extra  = (test['bp96'].astype(float) == 1)
is_interp = ~is_extra

# plot
plt.figure(figsize=(8, 6))
diagnoses = ['CN', 'MCI', 'AD']

for diag in diagnoses:
    mask_diag = (test['d96'] == diag)
    
    # interpolated (filled circle)
    m_interp = mask_diag & is_interp
    plt.scatter(
        y_true[m_interp], y_pred[m_interp],
        alpha=0.85, s=30, marker='o',
        color=diag_colors.get(diag, 'gray')
    )
    
    # extrapolated (hollow triangle)
    m_extra = mask_diag & is_extra
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

plt.xlabel('Predicted Brain Age at 96 Given MRI')
plt.ylabel('Predicted Brain Age at 96 Given Historical Progression and Baseline Scan')
plt.title('Brain Age at 96 months (outlier removed)')

# legends
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


# residuals
plt.figure(figsize=(6,6))
plt.hist(y_pred - y_true, bins='fd', color='moccasin', edgecolor='peru')
plt.title('Residuals BA96 (outlier removed)')
plt.xlabel('months')
plt.tight_layout()
plt.show()

