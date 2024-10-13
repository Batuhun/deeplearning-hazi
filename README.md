# deeplearning-hazi
## Team name: Programozók (not 100% sure)

## Team members' names and Neptun codes:
Tóth Fábián Tamás (D4ZXEU), 
Galacz Barnabás (D133RO)
## Project description
### Friend recommendation with graph neural networks
The goal of this project is to develop a personalized friend recommendation system by using Graph Neural Networks (GNNs). You have to analyze data from Facebook, Google+, or Twitter to suggest meaningful connections based on user profiles and interactions. This project offers a hands-on opportunity to deepen your deep learning and network analysis skills. 
## Functions of the files in the repository
`deeplearning-hazi/`     Repository

├── `Code.ipynb`         Interactive Jupyter notebook where the project’s code is written and executed.

├── `Code.py`            Python script containing the same code as the notebook but written for command-line execution.

├── `Dockerfile`         Instructions to build a Docker image for containerizing the project.

├── `README.md`          Contains details about the project and how to run it instructions for users.


## Related works (papers, GitHub repositories, blog posts, etc)
- https://github.com/Swetadas-1718/Facebook_Friend_Recommendation_Using_Graph_Mining
- https://github.com/miladfa7/Social-Network-Analysis-in-Python
- https://github.com/emreokcular/social-circle
- https://swetapadma449.medium.com/facebook-friend-recommendation-using-graph-mining-8d8d62153d14
- https://rendazhang.medium.com/graph-neural-network-series-2-convolution-on-graphs-delving-into-graph-convolutional-networks-79b42b042f53
- https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html

## How to run it (building and running the container, running your solution within the container)
-Put dockerfile and Code.py in the same folder
-Build docker image with: docker build -t my_gnn_project:latest .
-Run docker image with: docker run --gpus all -it my_gnn_project
(it will build the enviroment and run the code automatically)
