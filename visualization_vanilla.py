import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


features_dir = 'results/iu_xray/'

# load high-dimensional features
features = np.load(features_dir + 'features.npy', allow_pickle=True)

print(f"Loaded features of shape {features.shape}")
print(features)

# # use t-SNE to reduce the dimensionality of the features
# tsne = TSNE(n_components=2, perplexity=100, random_state=42)
# features_2d = tsne.fit_transform(features)

# # plot the 2D features
# plt.figure(figsize=(10, 10))
# plt.scatter(features_2d[:, 0], features_2d[:, 1], s=5, cmap='viridis')
# plt.title('t-SNE Visualization of High-Dimensional Data')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
# # plt.colorbar()


# # save the plot to a file
# output_path = features_dir + 'tsne_visualization.png'
# plt.savefig(output_path, dpi=300)
# print(f"t-SNE visualization saved to {output_path}")

# # plt.show()
