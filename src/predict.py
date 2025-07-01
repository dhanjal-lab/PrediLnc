
from src import *
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import time
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim


lnc1=topk(l1,result_lnc,10)
lnc2=topk(l3,result_lnc,10)
dis1=topk(d1,result_dis,10)
dis2=topk(d2,result_dis,10)
gg=topk(genes_func_sim,genes_gaussian_sim,10)

gl=genes_matrix[:,0:4458]
gd=genes_matrix[:,4458:]

features=np.vstack((encoded_lnc,encoded_dis,np.zeros((13356, 512))))

adj=adj_matrix(lda,lnc1,dis1,gl,gd,gg)
print(adj.shape)

model = GCN(nfeat=512, nhid=512, dropout=0.4)
reconstruction_criterion = nn.MSELoss()  # Reconstruction loss criterion

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)




# Calculate Laplacian matrix
adj_tensor = torch.Tensor(adj)

# Training loop
t_total = time.time()
features_tensor = torch.Tensor(features)
adj=adj_norm(adj_tensor)

for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    output1 = model(features_tensor, adj)
    
    # Reconstruction Loss
    reconstruction_loss = reconstruction_criterion(output1, features_tensor)
    
    # Total Loss
    loss = reconstruction_loss
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    print("Epoch: ", epoch, " Reconstruction Loss: ", reconstruction_loss.item())

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


model=torch.load('GCN_node1.pth')
model.eval() 
node_output1 = model(features_tensor, adj_tensor).detach().numpy()


# features from lncRNA sequence information and semantic information is taken which is generated using GCN

lncRNA_feature=node_output1[0:4458]
disease_feature=node_output1[4458:]
print(lncRNA_feature.shape)
print(disease_feature.shape)

# Identify indices where value is 1 in lda
indices_1 = np.argwhere(lda == 1)

# Identify indices where value is 0 in lda
indices_0 = np.argwhere(lda == 0)
# Initialize arrays to store extracted rows and labels
X=[]
y = []
label=[]
# Extract rows for indices where value is 1
for idx in indices_1:
    i, j = idx
    a=lncRNA_feature[i]
    b=disease_feature[j]
    X.append(np.concatenate((a,b)))
    y.append(1)
    label.append((i,j))

# Extract rows for randomly selected indices where value is 0
for idx in np.random.choice(len(indices_0), len(indices_1), replace=False):
    i, j = indices_0[idx]
    a=lncRNA_feature[i]
    b=disease_feature[j]
    X.append(np.concatenate((a,b)))
    y.append(0)
    label.append((i,j))


X = np.array(X)
y = np.array(y)


from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Assuming X and y are your feature matrix and labels
num_samples = X.shape[0]  # Assuming all datasets have the same number of samples
indices = np.arange(num_samples)
np.random.shuffle(indices)


# This is the prediction using the nodes where bias is towards lncRNA sequence and disease semantic information



# Shuffling all datasets using the same shuffled indices
X_shuffled = X[indices]
y1 = y[indices]
X1 = X_shuffled
y = y1

# Models for binary classification
models1 = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'MLP': MLPClassifier(max_iter=300)  # Adjust parameters as needed
}

# Meta-model
meta_model1 = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features='sqrt')

# Dictionary to store evaluation metrics
evaluation_metrics = {}

# Leave-One-Out Cross-Validation with shuffling
skf = StratifiedKFold(n_splits=5)  # StratifiedKFold for shuffling

predictions = {model_name: [] for model_name in models1}

scaler = StandardScaler()
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
output1={
     'Logistic Regression': [],
    'SVM': [],
    'Random Forest': [],
    'Gradient Boosting': [],
    'MLP': []  # Adjust parameters as needed
}
for train_index, test_index in skf.split(X1, y):
    X_train1, X_test1 = X1[train_index], X1[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Standardize features
    X_train1_std = scaler.fit_transform(X_train1)
    X_test1_std = scaler.transform(X_test1)

    # Train and predict with each base model
    for model_name, model in models1.items():
        print(model_name)
        model.fit(X_train1_std, y_train)
        y_pred_proba = model.predict_proba(X_test1_std)[:, 1]
        predictions[model_name].extend(y_pred_proba)
        output1[model_name].append((y_test,y_pred_proba))


