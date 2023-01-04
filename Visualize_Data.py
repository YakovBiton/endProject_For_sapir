from DataSetRead import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_data(features, labels):
   if isinstance(features, list):
     features = np.array(features)
   flat_features = features.reshape(-1, 1)
   #pca =PCA(n_components=1)
  # pca.fit(flat_features)
   #reduced_features = pca.transform(flat_features)
   X_embedded = TSNE(n_components=2,perplexity=10).fit_transform(flat_features)
   plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels)
   plt.show()