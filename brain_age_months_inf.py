from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
import matplotlib.pyplot as plt
from brain_age_months import load_and_split
# from bam_nonconv import load_and_split

# data path
path = '/Users/deanyao/Desktop/penn surf/mri_data/istaging_data.csv'
train_val, test = load_and_split(path)
print("balls", len(train_val['PTID'].unique()))
print(len(test['PTID'].unique()))

# load continuous model
model_folder = "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/brain_age_months_nestedcv"
# model_folder = "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/brain_age_months_trying"


# load model
predictor = TabularPredictor.load(model_folder)

eval = predictor.evaluate(test)
rmse = -eval['root_mean_squared_error']

print(eval)
print(rmse)

y_true = test['calendar_time']
y_pred = predictor.predict(test)

accuracy1 = (np.abs(y_pred - y_true) <= rmse).mean()
accuracy2 = (np.abs(y_pred - y_true) <= 2 * rmse).mean()

plt.figure(figsize=(6,6))

for pid, grp in test.groupby("PTID"):  
    # sort that patient's visits by true time
    grp = grp.sort_values("calendar_time")
    x = grp["calendar_time"]
    y = y_pred[grp.index]  # pick out predicted values for this patient
    plt.plot(x, y, '-', alpha=0.3, linewidth=0.8)


# Scatter plot colored by diagnosis
for diag in test['Diagnosis'].unique():
    mask = test['Diagnosis'] == diag
    plt.scatter(y_true[mask], y_pred[mask], alpha=0.6, label=diag)

# Add identity line
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], linestyle='--', color='gray', label='y = x')

plt.plot([0, max_val], [min_val + rmse, max_val + rmse], linestyle='--', color='red')
plt.plot([0, max_val], [min_val - rmse, max_val - rmse], linestyle='--', color='red')
plt.ylim(0, y_pred.max() + 12)

# Labels, title, legend
plt.xlabel('True Calendar Time (months)')
plt.ylabel('Predicted Calendar Time (months)')
plt.title('True vs Predicted Calendar Time Since Baseline (ADNI)')
plt.legend(title='Diagnosis', bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()

# plt.hist(train_val['calendar_time'], bins=30, alpha=0.6, label='train_val')
# plt.hist(test['calendar_time'],    bins=30, alpha=0.6, label='test')
# plt.legend(); plt.title("Calendar Time Distribution"); plt.show()

