# -*- coding: utf-8 -*-
"""Code.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12waKp9VgedzEBSqHk2Ukeni4H0fEUNZy
"""



"""# Install torch and download the dataset"""

import os
import torch
import urllib.request

# Step 1: Download the Facebook Ego dataset
url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
dataset_path = "facebook_combined.txt.gz"

if not os.path.exists(dataset_path):
    urllib.request.urlretrieve(url, dataset_path)
    print("Facebook dataset downloaded.")

import os
import torch
import urllib.request

# Step 1: Download the Facebook Ego dataset
full_url = "https://snap.stanford.edu/data/facebook.tar.gz"
full_dataset_path = "facebook.tar.gz"

if not os.path.exists(full_dataset_path):
    urllib.request.urlretrieve(full_url, full_dataset_path)
    print("Facebook dataset downloaded.")

import zipfile
import gzip
import pandas as pd

lines = []
with gzip.open(full_dataset_path, 'rt') as f:
    for line in f:
        line = line.strip(' ')
        lines.append(line)
pd.DataFrame(lines).head()

"""
# Data analysis"""

import networkx as nx
import matplotlib.pylab as plt

# Load the original graph from the edge list
graph = nx.read_edgelist('facebook_combined.txt.gz', delimiter=' ', create_using=nx.Graph(), nodetype=int)

# Get the first 50 nodes
subgraph = list(graph.nodes())[25:75]

# Create a subgraph with only those 50 nodes
subgraph = graph.subgraph(subgraph)

# Generate the layout for the subgraph
pos = pos = nx.circular_layout(subgraph)
# Draw the subgraph
nx.draw(subgraph, pos, node_color='#A0CBE2', edge_color='#808080', width=1, edge_cmap=plt.cm.Blues, with_labels=True, arrows=False)

# Show the plot
plt.show()

print("The number of unique persons",len(graph.nodes()))

degree_dist = list(dict(graph.degree()).values())
degree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(degree_dist)
plt.xlabel('Index No')
plt.ylabel('No Of Followers')
plt.show()

# let's zoom in
degree_dist = list(dict(graph.degree()).values())
degree_dist.sort()
plt.figure(figsize=(10,6))
plt.plot(degree_dist[0:2000])
plt.xlabel('Index No')
plt.ylabel('No Of Followers')
plt.show()

plt.boxplot(degree_dist)
plt.ylabel('No of followers')
plt.show()

import numpy as np
# 0 and 90-100 percentile
print(0, "percentile value is", np.percentile(degree_dist,0))
for i in range(0,11):
  print(90+i, "percentile value is", np.percentile(degree_dist,90+i))

# Commented out IPython magic to ensure Python compatibility.
import seaborn as sns
# %matplotlib inline
sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.histplot(degree_dist, color='#16A085')
plt.xlabel('PDF of Degree')
sns.despine()
plt.show()

"""# Data preprocessing"""

import zipfile
# Step 2: Preprocess the data - Create edge list
import gzip

edges = []
with gzip.open(dataset_path, 'rt') as f:
    for line in f:
        src, dst = map(int, line.strip().split())
        edges.append([src, dst])

edges_list = edges
print(len(edges_list))
edges_list = [edge for edge in edges_list if edge[0] != edge[1]]
edges_list = list(set([tuple(sorted(edge)) for edge in edges_list]))
#remove nodes (and their edges) that have a degree below 2
from collections import Counter
degrees = Counter([edge[0] for edge in edges_list] + [edge[1] for edge in edges_list])
edges_list = [edge for edge in edges_list if degrees[edge[0]] > 1 and degrees[edge[1]] > 1]
print(len(edges_list))
high_degree_threshold = 0.99  # remove top 1% of nodes with highest degrees
threshold = np.percentile(list(degrees.values()), 99)
edges_list = [edge for edge in edges_list if degrees[edge[0]] < threshold and degrees[edge[1]] < threshold]
print(len(edges_list))

"""# Data preprocessing, Creating and training the model"""

import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool  # Global pooling
from sklearn.model_selection import train_test_split
from torch_geometric.nn import BatchNorm
from node2vec import Node2Vec

edges_copy = edges_list
torch_edges = torch.tensor(edges_copy, dtype=torch.long).t().contiguous()
# Number of nodes
num_nodes = torch_edges.max().item() + 1

# Create node features based on node degree
node_degrees = torch.tensor([degrees[node] for node in range(num_nodes)], dtype=torch.float)

# Reshape node features to have them as a feature vector (degree as a single feature)
x = node_degrees.view(-1, 1)

# Calculate the average clustering coefficient
avg_clustering_coeff = nx.average_clustering(graph)

# Add the average clustering coefficient as a feature for every node
avg_clustering_tensor = torch.full((num_nodes, 1), avg_clustering_coeff, dtype=torch.float)

# Append the feature to the existing node features
x = torch.cat([x, avg_clustering_tensor], dim=1)

# Add the diameter as a feature for each node
graph_diameter = 8  # based on the dataset information
diameter_tensor = torch.full((num_nodes, 1), graph_diameter, dtype=torch.float)

# Add the diameter as a feature to each node
x = torch.cat([x, diameter_tensor], dim=1)


# Initialize the Node2Vec model
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200)

# Fit the model to generate embeddings
model = node2vec.fit(window=5, min_count=1)
print(len(model.wv))  # Should match the number of nodes in the graph
print(num_nodes)      # Should also match the number of nodes in the graph
# Get embeddings for all nodes
embeddings = [model.wv[str(node)] for node in range(num_nodes)]

# Convert embeddings to a tensor
embedding_tensor = torch.tensor(embeddings, dtype=torch.float)

# Concatenate the embeddings to the existing feature matrix
x = torch.cat([x, embedding_tensor], dim=1)

# Update the Data object with the new feature matrix
data = Data(x=x, edge_index=torch_edges)

print(f"Node features based on degree: {data.x[:5]}")

# Step 4: Define a simple GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN, self).__init__()
        # First layer: from in_channels to hidden_channels
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Second layer: from hidden_channels to hidden_channels
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # Third layer: from hidden_channels to out_channels (final output)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.batch_norm2 = BatchNorm(hidden_channels)
        # Fully connected layers
        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm1(x)  # Batch normalization
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm2(x)  # Batch normalization
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Third (final) layer
        x = self.conv3(x, edge_index)

        # Global pooling (for graph-level representation)
        #x = global_mean_pool(x, batch)  # Pooling over all nodes in the graph

        # Pass through the fully connected layer
        x = self.fc(x)

        return F.log_softmax(x, dim=1)  # Final classification

# Step 5: Create a simple task - Node Classification (Dummy labels)
# For simplicity, let's create some random labels to train the model.
# In a real-world case, you would have actual node labels.

labels = torch.randint(0, 2, (num_nodes,))
train_mask, test_mask = train_test_split(torch.arange(num_nodes), test_size=0.2)

# Step 6: Initialize the model and optimizer
model = GCN(in_channels=x.size(1), hidden_channels=16, out_channels=2, dropout=0.5)
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
for epoch in range(1, 1201):
    loss = train()
    if epoch % 20 == 0:
        acc = test()
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')