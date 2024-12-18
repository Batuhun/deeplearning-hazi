# -*- coding: utf-8 -*-
"""Code.ipynb

Original file is located at
    https://colab.research.google.com/drive/12waKp9VgedzEBSqHk2Ukeni4H0fEUNZy

"""

# Standard library imports
import os
import gzip
import zipfile
import urllib.request
from collections import Counter

# Numerical and scientific computing
import numpy as np

# PyTorch and PyTorch Geometric
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, BatchNorm, global_mean_pool

# Machine Learning and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Graph-specific tools
import networkx as nx
from node2vec import Node2Vec

# Optimization
import optuna

# Experiment tracking
import wandb


"""# Download the `facebook_combined.txt.gz` dataset
## From: https://snap.stanford.edu/data/ego-Facebook.html
"""
# Initialize W&B
os.environ["WANDB_MODE"] = "offline"

wandb.init(project="gcn-edge-prediction")

# Step 1: Download the Facebook Ego dataset
url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
dataset_path = "facebook_combined.txt.gz"

if not os.path.exists(dataset_path):
    urllib.request.urlretrieve(url, dataset_path)
    print("Facebook dataset downloaded.")

"""
# Data analysis"""

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

# Load the original graph from the edge list
graph = nx.read_edgelist('facebook_combined.txt.gz', delimiter=' ', create_using=nx.Graph(), nodetype=int)

# Generate a spring layout for the graph (force-directed)
pos = nx.spring_layout(graph, seed=42)  # Seed for consistent layout

# Draw the graph with smaller nodes and edges
nx.draw(
    graph,
    pos,
    node_size=10,       # Smaller nodes
    edge_color='#808080',
    width=0.2,          # Thinner edges
    edge_cmap=plt.cm.Blues,
    with_labels=False   # No labels
)

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

# 0 and 90-100 percentile
print(0, "percentile value is", np.percentile(degree_dist,0))
for i in range(0,11):
  print(90+i, "percentile value is", np.percentile(degree_dist,90+i))

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
sns.set_style('ticks')
fig, ax = plt.subplots()
sns.histplot(degree_dist, color='#16A085')
plt.xlabel('PDF of Degree')
sns.despine()
plt.show()

"""# Data preprocessing"""

# Preprocess the data - Create edge list

edges = []
with gzip.open(dataset_path, 'rt') as f:
    for line in f:
        src, dst = map(int, line.strip().split())
        edges.append([src, dst])
edges_list = edges

degrees = Counter([edge[0] for edge in edges_list] + [edge[1] for edge in edges_list])

edges_list = edges
print(len(edges_list))
edges_list = [edge for edge in edges_list if edge[0] != edge[1]]
edges_list = list(set([tuple(sorted(edge)) for edge in edges_list]))
# remove nodes (and their edges) that have a degree below 2
edges_list = [edge for edge in edges_list if degrees[edge[0]] > 1 and degrees[edge[1]] > 1]
print(len(edges_list))
# remove top 1% of nodes with highest degrees
# high_degree_threshold = 0.99
# threshold = np.percentile(list(degrees.values()), 99)
# edges_list = [edge for edge in edges_list if degrees[edge[0]] < threshold and degrees[edge[1]] < threshold]
# print(len(edges_list))


graph = nx.Graph()
# Load the original graph from the edge list
graph.add_edges_from(edges_list)

# Generate a spring layout for the graph (force-directed)
pos = nx.spring_layout(graph, seed=42)  # Seed for consistent layout

# Draw the graph with smaller nodes and edges
nx.draw(
    graph,
    pos,
    node_size=10,       # Smaller nodes
    edge_color='#808080',
    width=0.2,          # Thinner edges
    edge_cmap=plt.cm.Blues,
    with_labels=False   # No labels
)

# Show the plot
plt.show()

"""#Baseline"""



# Step 1: Build the Graph from the Edge List
def build_graph(edge_list):
    G = nx.Graph()
    G.add_edges_from(edge_list)
    return G

# Load preprocessed edge list and build the graph
G = build_graph(edges_list)

# Step 2: Split Edges into Train and Test Sets
np.random.seed(42)
edges = np.array(edges_list)
np.random.shuffle(edges)

train_size = int(0.8 * len(edges))  # 80% train, 20% test
train_edges = edges[:train_size]
test_edges = edges[train_size:]

# Non-existent edges to use as negative samples in the test set
all_possible_edges = set(nx.non_edges(G))
negative_samples = np.array(list(all_possible_edges))[:len(test_edges)]

# Step 3: Calculate Jaccard Similarity Scores
def calculate_jaccard_scores(edges, G):
    scores = []
    for src, dst in edges:
        if G.has_node(src) and G.has_node(dst):
            jaccard_score = list(nx.jaccard_coefficient(G, [(src, dst)]))
            if jaccard_score:
                score = jaccard_score[0][2]
            else:
                score = 0
        else:
            score = 0
        scores.append(score)
    return np.array(scores)

# Calculate similarity scores for positive (test edges) and negative samples
positive_scores = calculate_jaccard_scores(test_edges, G)
negative_scores = calculate_jaccard_scores(negative_samples, G)

# Step 4: Evaluate the Baseline Model
# Concatenate scores and create labels for evaluation
all_scores = np.concatenate([positive_scores, negative_scores])
labels = np.concatenate([np.ones(len(positive_scores)), np.zeros(len(negative_scores))])

# Set a threshold to classify edges as "connected" or "not connected"
threshold = 0.5
predictions = (all_scores >= threshold).astype(int)

# Calculate evaluation metrics
accuracy = accuracy_score(labels, predictions)
auc = roc_auc_score(labels, all_scores)

# Print results
print(f"Baseline Model - Jaccard Similarity")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")


"""# Adding extra features to the nodes"""

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=1)

# Fit the model to generate embeddings
model = node2vec.fit(window=5, min_count=1)

# Get embeddings for all nodes as a numpy array
embeddings = []
for node in range(num_nodes):
    key = str(node)  # Ensure consistent string conversion
    if key in model.wv:
        embeddings.append(model.wv[key])
    else:
        #print(f"Node {node} missing from embeddings. Initializing with zeros.")
        embeddings.append(np.zeros(model.vector_size))  # Placeholder
embeddings = np.array(embeddings)

# Convert embeddings to a tensor
embedding_tensor = torch.tensor(embeddings, dtype=torch.float)

# Concatenate the embeddings to the existing feature matrix
x = torch.cat([x, embedding_tensor], dim=1)

# Update the Data object with the new feature matrix
data = Data(x=x, edge_index=torch_edges)

# Save the data object
torch.save(data, 'data_object.pth')

#print(f"Node features based on degree: {data.x[:5]}")

"""#Generating negative edges and splitting up data

"""

# Create labels for friend recommendation
# Positive samples (existing edges)
positive_edges = torch.tensor(edges_copy, dtype=torch.long).t().contiguous()

# Generate negative samples
num_nodes = torch_edges.max().item() + 1
negative_edges = []
while len(negative_edges) < len(positive_edges[0]) // 2:  # Half the size of positive edges
    src = torch.randint(0, num_nodes, (1,)).item()
    dst = torch.randint(0, num_nodes, (1,)).item()
    if src != dst and (src, dst) not in edges_list and (dst, src) not in edges_list:
        negative_edges.append([src, dst])

# Split into train and combined test/validation set
all_edges = torch.cat([positive_edges, torch.tensor(negative_edges, dtype=torch.long).t()], dim=1)
all_labels = torch.cat([torch.ones(positive_edges.size(1)), torch.zeros(len(negative_edges))])  # Labels for edges

train_edges, test_val_edges, train_labels, test_val_labels = train_test_split(
    all_edges.t(), all_labels, test_size=0.2, random_state=42
)

# Create Data object for the training set
train_data = Data(x=x, edge_index=train_edges.t().contiguous(), y=train_labels)

# Split the combined test/validation set into test and validation sets
val_edges, test_edges, val_labels, test_labels = train_test_split(
    test_val_edges, test_val_labels, test_size=0.5, random_state=42
)

# Create Data objects for validation and test sets
val_data = Data(x=x, edge_index=val_edges.t().contiguous(), y=val_labels)
test_data = Data(x=x, edge_index=test_edges.t().contiguous(), y=test_labels)

"""#Training and testing"""

class GCNEdgePrediction(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.25):
        super(GCNEdgePrediction, self).__init__()

        # Feature transformation layer
        self.pre_transform = Linear(in_channels, hidden_channels)

        # Define GCN layers with a mix of GCNConv, GATConv, and SAGEConv
        self.layers = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # First layer: GCNConv
        self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.bns.append(BatchNorm1d(hidden_channels))

        # Intermediate layers: Mix of GCNConv, GATConv, and SAGEConv
        for i in range(1, num_layers - 1):
            if i % 3 == 0:
                self.layers.append(GCNConv(hidden_channels, hidden_channels))
            elif i % 3 == 1:
                self.layers.append(GATConv(hidden_channels, hidden_channels, heads=4, concat=False))
            else:
                self.layers.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))

        # Last layer: GCNConv
        self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.bns.append(BatchNorm1d(hidden_channels))

        # Fully connected layers
        self.fc1 = Linear(2 * hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, hidden_channels // 2)
        self.fc3 = Linear(hidden_channels // 2, hidden_channels // 4)
        self.fc4 = Linear(hidden_channels // 4, 1)

        # Dropout rate
        self.dropout = dropout

    def forward(self, x, edge_index, edge_label_index):
        # Pre-transform input features
        x = self.pre_transform(x)

        # GCN Layers with residual connections
        for layer, bn in zip(self.layers, self.bns):
            x_residual = x
            x = layer(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Add residual connection if shapes match
            if x.shape == x_residual.shape:
                x += x_residual

        # Edge embeddings: combine features for each pair of nodes
        src, dst = edge_label_index
        edge_embeddings = torch.cat([x[src], x[dst]], dim=1)

        # Fully connected layers for final prediction
        x = F.relu(self.fc1(edge_embeddings))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        edge_scores = self.fc4(x)  # Output score for each edge

        return edge_scores


# Initialize model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNEdgePrediction(in_channels=data.x.size(1), hidden_channels=128, dropout=0.1097076435322618).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00011840161483188985)

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

# Training and testing functions
def train():
    model.train()
    optimizer.zero_grad()
    out = model(train_data.x, train_data.edge_index, train_data.edge_index)
    loss = F.binary_cross_entropy_with_logits(out, train_data.y.float().unsqueeze(1))
    loss.backward()
    optimizer.step()
    return loss.item()

def test(data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_index)
        probs = torch.sigmoid(out).squeeze()
        pred_labels = (probs > 0.5).float()
        acc = accuracy_score(data.y.cpu(), pred_labels.cpu())
        auc = roc_auc_score(data.y.cpu(), probs.cpu())
        return acc, auc

def validate():
    val_acc, val_auc = test(val_data)
    return val_acc, val_auc


# Set up W&B configuration and logging
config = wandb.config
config.patience = 150
config.epochs = 2000

patience = config.patience
best_val_auc = 0.0
epochs_without_improvement = 0

for epoch in range(1, config.epochs + 1):
    train_loss = train()
    val_acc, val_auc = validate()

    # Log metrics to W&B
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_accuracy": val_acc,
        "val_auc": val_auc,
    })

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch}, best validation AUC: {best_val_auc:.4f}")
        break

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

# Final test evaluation
model.load_state_dict(torch.load('best_model.pth'))
test_acc, test_auc = test(test_data)
print(f"Final Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

# Log final test metrics to W&B
wandb.log({"test_accuracy": test_acc, "test_auc": test_auc})
wandb.finish()

"""#Hyper parameter optimization"""

# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters to tune
    hidden_channels = trial.suggest_int('hidden_channels', 64, 256)
    num_layers = trial.suggest_int('num_layers', 1, 16)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Initialize the model and optimizer with trial-suggested parameters
    model = GCNEdgePrediction(
        in_channels=data.x.size(1),
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(50):  # A small number of epochs to save time
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index, train_data.edge_index)
        loss = F.binary_cross_entropy_with_logits(out, train_data.y.float().unsqueeze(1).to(device))
        loss.backward()
        optimizer.step()

    # Validation step
    val_acc, val_auc = test(val_data)  # Assume `test` function works with validation data
    return val_auc  # Return validation AUC as the metric to maximize

# Running Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=250)

# Best trial results
print("Best trial:")
trial = study.best_trial
print(f"  AUC: {trial.value}")
print("  Best hyperparameters:", trial.params)

best_trial = study.best_trial
best_params = best_trial.params
model = GCNEdgePrediction(
    in_channels=data.x.size(1),
    hidden_channels=best_params['hidden_channels'],
    num_layers=best_params['num_layers'],
    dropout=best_params['dropout']
).to(device)
# Initialize model and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

# Initialize W&B
wandb.init(project="gcn-edge-prediction")
# Set up W&B configuration and logging
config = wandb.config
config.patience = 150
config.epochs = 2000

patience = config.patience
best_val_auc = 0.0
epochs_without_improvement = 0

for epoch in range(1, config.epochs + 1):
    train_loss = train()
    val_acc, val_auc = validate()

    # Log metrics to W&B
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_accuracy": val_acc,
        "val_auc": val_auc,
    })

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch}, best validation AUC: {best_val_auc:.4f}")
        break

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

# Final test evaluation
model.load_state_dict(torch.load('best_model.pth'))
test_acc, test_auc = test(test_data)
print(f"Final Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

# Log final test metrics to W&B
wandb.log({"test_accuracy": test_acc, "test_auc": test_auc})
wandb.finish()

