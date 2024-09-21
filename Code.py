# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12waKp9VgedzEBSqHk2Ukeni4H0fEUNZy
"""


import os
import torch
import urllib.request
import zipfile
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split

# Step 1: Download the Facebook Ego dataset
url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
dataset_path = "facebook_combined.txt.gz"

if not os.path.exists(dataset_path):
    urllib.request.urlretrieve(url, dataset_path)
    print("Facebook dataset downloaded.")

# Step 2: Preprocess the data - Create edge list
import gzip

edges = []
with gzip.open(dataset_path, 'rt') as f:
    for line in f:
        src, dst = map(int, line.strip().split())
        edges.append([src, dst])

edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Number of nodes
num_nodes = edges.max().item() + 1

# Generate some dummy node features (for simplicity, use identity matrix)
x = torch.eye(num_nodes)

# Step 3: Create PyTorch Geometric Data object
data = Data(x=x, edge_index=edges)

# Step 4: Define a simple GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Step 5: Create a simple task - Node Classification (Dummy labels)
# For simplicity, let's create some random labels to train the model.
# In a real-world case, you would have actual node labels.

labels = torch.randint(0, 2, (num_nodes,))
train_mask, test_mask = train_test_split(torch.arange(num_nodes), test_size=0.2)

# Step 6: Initialize the model and optimizer
model = GCN(in_channels=x.size(1), hidden_channels=16, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 7: Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Step 8: Test function
def test():
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        correct = (pred[test_mask] == labels[test_mask]).sum()
        acc = int(correct) / test_mask.size(0)
        return acc

# Training the model
for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        acc = test()
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')