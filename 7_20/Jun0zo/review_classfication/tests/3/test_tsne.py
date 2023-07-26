import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create a t-SNE model and apply it to the data
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
print(X_tsne[:, 0])
print(X_tsne[:, 0].shape)

# Create a scatter plot of the t-SNE results
plt.figure(figsize=(8, 6))
for i, target_name in zip(range(len(iris.target_names)), iris.target_names):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=target_name)

plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization of Iris Dataset')
plt.legend()
plt.show()
