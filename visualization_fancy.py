import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json

features_dir = 'results/iu_xray/'
annotation_path = 'data/iu_xray/updated_annotation.json'

# Load high-dimensional features
features = np.load(features_dir + 'features.npy', allow_pickle=True).item()
print('features: ', features)

# Load annotation data
with open(annotation_path, 'r') as f:
    annotations = json.load(f)

# Extract labels
labels = {}
for split in ['test']:
    for item in annotations[split]:
        image_id = item['id']
        labels[image_id] = 0  # Default to no disease
        for key, value in item['labels'].items():
            if key not in ['Support Devices', 'No Finding'] and value == '1.0':
                labels[image_id] = 1  # Mark as having disease
                break

# Prepare data for t-SNE
feature_list = []
color_list = []
for key, feature in features.items():
    image_id = key.split('_view')[0]
    feature_list.append(feature)
    color_list.append(labels[image_id])

feature_array = np.array(feature_list)

# Use t-SNE to reduce the dimensionality of the features
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_2d = tsne.fit_transform(feature_array)

# Plot the 2D features
plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=color_list, s=5, cmap='viridis')
plt.title('t-SNE Visualization of High-Dimensional Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(scatter, ticks=[0, 1], label='Disease Status')
plt.clim(-0.5, 1.5)

# Save the plot to a file
output_path = features_dir + 'tsne_visualization.png'
plt.savefig(output_path, dpi=300)
print(f"t-SNE visualization saved to {output_path}")

# # Show the plot
# plt.show()
