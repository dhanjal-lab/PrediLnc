from src import *

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

# Assuming output1 contains ROC curve and Precision-Recall curve data for each model and fold

colors = ['r', 'g', 'b', 'c', 'm']  # List of colors for different models
model_names = list(output1.keys())  # List of model names
num_folds = 5  # Number of folds

# Plot ROC curves
for i, model_name in enumerate(model_names):
    for j,fold in enumerate(output1[model_name]):
        y_test, y_pred_proba= fold
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'Model {model_name}, Fold {j+1} (AUC = {roc_auc:.2f})', color=colors[j])

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot Precision-Recall curves
for i, model_name in enumerate(model_names):
    for j,fold in enumerate(output1[model_name]):
        y_test, y_pred_proba= fold
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        aupr = average_precision_score(y_test, y_pred_proba)
        plt.plot(recall, precision, label=f'Model {model_name}, Fold {j+1} (AUPR = {aupr:.2f})', color=colors[j])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()



# Create input features for the meta-model
meta_X = np.column_stack([predictions[model_name] for model_name in models1])

# Train the meta-model
meta_model1.fit(meta_X, y)

# Evaluate the meta-model
y_meta_pred_proba = meta_model1.predict_proba(meta_X)[:, 1]
y_meta_pred = (y_meta_pred_proba >= 0.5).astype(int)

# Display evaluation metrics for the meta-model
roc_auc_meta = roc_auc_score(y, y_meta_pred_proba)
accuracy_meta = accuracy_score(y, y_meta_pred)
precision_meta = precision_score(y, y_meta_pred)
recall_meta = recall_score(y, y_meta_pred)
f1_meta = f1_score(y, y_meta_pred)
mcc_meta = matthews_corrcoef(y, y_meta_pred)
aupr_meta = average_precision_score(y, y_meta_pred_proba)

print("\nMeta-Model Metrics:")
print(f"ROC AUC: {roc_auc_meta:.4f}")
print(f"Accuracy: {accuracy_meta:.4f}")
print(f"Precision: {precision_meta:.4f}")
print(f"Recall: {recall_meta:.4f}")
print(f"F1-score: {f1_meta:.4f}")
print(f"MCC: {mcc_meta:.4f}")
print(f"AUPR: {aupr_meta:.4f}")

# Plot ROC curve for the meta-model
fpr_meta, tpr_meta, thresholds_meta = roc_curve(y, y_meta_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr_meta, tpr_meta, label=f'Meta-Model (AUC = {roc_auc_meta:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Meta-Model')
plt.legend()
plt.show()
