# Topological and semantic contrastive graph clustering by Ricci curvature augmentation and hypergraph fusion

We propose a Topological and Semantic Contrastive Graph Clustering (TSCGC) model consisting of three learning components. The representation learning component augments original graph using Ricci curvature to preserve the cluster structure, and introduces hypergraph view to capture high-order relationships. Graph and hypergraph convolutional networks are used to encode the triple-view embeddings. Meanwhile, we develop a dual contrastive learning component to extract the topological and semantic information. It employs the pseudo clustering labels to guide positive sample selection and reduces the false negative samples. The self-supervised learning component is leveraged to align the three graph views.

![image](https://github.com/ZPGuiGroupWhu/TSCGC/blob/main/Graphical%20Abstract.png)
