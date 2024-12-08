# -*- coding: utf-8 -*-
"""Code.ipynb

Original file is located at
    https://colab.research.google.com/drive/12waKp9VgedzEBSqHk2Ukeni4H0fEUNZy

"""


# Numerical and scientific computing
import numpy as np

# PyTorch and PyTorch Geometric
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

import gradio as gr


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

# Define the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = torch.load('data_object.pth')
data.x = data.x.to(device)
data.edge_index = data.edge_index.to(device)
model = GCNEdgePrediction(
    in_channels=data.x.size(1),
    hidden_channels=195,
    num_layers=1,
    dropout=0.2741000810650823
).to(device)
# Load model
model.load_state_dict(torch.load('best_model.pth', map_location=device))

# Move model and data to the selected device
model.to(device)


# Step 1: Generate a larger list of real human names
name_list = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hannah", "Ivy",
    "Jack", "Kathy", "Liam", "Mia", "Noah", "Olivia", "Paul", "Quinn", "Riley",
    "Sophia", "Tom", "Uma", "Victor", "Willow", "Xander", "Yara", "Zane",
    "Emma", "James", "Lucas", "Ava", "Isabella", "Alexander", "Benjamin",
    "Charlotte", "Daniel", "Elijah", "Ethan", "Henry", "Jacob", "Logan",
    "Madison", "Mason", "Natalie", "Samuel", "Sebastian", "Scarlett", "Zoe",
    "Oliver", "Lily", "Chloe", "Ella", "Emily", "Ryan", "Jason", "Aaron",
    "Nathan", "Matthew", "Sophia", "David", "Sarah", "Anna", "Elizabeth"
]

# Ensure we have enough names for all nodes by repeating names if necessary
num_nodes = data.x.size(0)
while len(name_list) < num_nodes:
    name_list.extend(name_list)  # Duplicate the name list if not enough names

# Shuffle the names randomly
np.random.shuffle(name_list)

# Create a mapping of node index to name
node_names = {i: name_list[i] for i in range(num_nodes)}


# Step 2: Modify the friend recommendation function to use names
def recommend_friend_with_names(node):
    model.eval()

    # Validate node index
    if node < 0 or node >= data.x.size(0):
        return f"Invalid node. Node should be between 0 and {data.x.size(0) - 1}.", None

    # Identify potential friends (nodes not directly connected to the given node)
    existing_edges = set(map(tuple, data.edge_index.t().tolist()))
    potential_friends = [
        i for i in range(data.x.size(0))
        if (node, i) not in existing_edges and (i, node) not in existing_edges and node != i
    ]

    if not potential_friends:
        return f"No potential friends for {node_names[node]}.", None

    # Prepare edges for prediction
    edges_to_predict = torch.tensor(
        [[node, friend] for friend in potential_friends], dtype=torch.long
    ).t().to(device)

    # Predict edge probabilities
    with torch.no_grad():
        preds = model(data.x, data.edge_index, edges_to_predict)
        probs = torch.sigmoid(preds).squeeze().cpu().tolist()

    # Pair potential friends with probabilities
    recommendations = list(zip(potential_friends, probs))

    # Sort recommendations by probability in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Best recommendation
    best_friend, best_score = recommendations[0]
    recommendation_text = (
        f"Recommended friend for {node_names[node]}: {node_names[best_friend]} "
        f"with likelihood {best_score:.4f}"
    )

    # Display top 5 recommendations
    top_5_recommendations = [
        f"{node_names[friend]}: {score:.4f}" for friend, score in recommendations[:5]
    ]

    return recommendation_text, top_5_recommendations

# Step 3: Gradio interface with names
interface = gr.Interface(
    fn=recommend_friend_with_names,
    inputs=gr.Number(label="Enter Node Index"),
    outputs=[
        gr.Textbox(label="Friend Recommendation"),
        gr.Textbox(label="Top 5 Recommendations"),
    ],
    title="GCN Friend Recommendation System with Names",
    description="Enter a node index to get a recommended friend based on the trained GCN model."
)

# Launch Gradio app
interface.launch(share=True)
