from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from process_html import BlockNode

def get_leaf_nodes(node) -> list:
    """Recursively collect all leaf nodes with embeddings."""
    if not node.children and node.embedding is not None:
        return [node]
    leaves = []
    for child in node.children:
        leaves.extend(get_leaf_nodes(child))
    return leaves

def visualize_block_clusters(
    root_node,
    method="umap",               # "umap" or "tsne"
    title="Block Cluster Visualization"
):
    # Step 1: Collect embedded leaf nodes
    leaf_nodes = get_leaf_nodes(root_node)
    if not leaf_nodes:
        print("No leaf nodes with embeddings found.")
        return

    embeddings = np.array([node.embedding for node in leaf_nodes])
    labels = np.array([
        getattr(node, "cluster_label", -1)
        for node in leaf_nodes
    ])

    # Step 2: Dimensionality reduction
    if method == "umap":
        reducer = UMAP(n_neighbors=10, min_dist=0.3, metric="cosine", random_state=42)
    # elif method == "tsne":
    #     reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    else:
        raise ValueError("Unsupported method. Use 'umap' or 'tsne'.")

    reduced = reducer.fit_transform(embeddings)

    # Step 3: Visualization
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=reduced[:, 0],
        y=reduced[:, 1],
        hue=labels,
        palette="tab10",
        s=70,
        legend="full"
    )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
