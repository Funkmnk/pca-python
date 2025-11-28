import pickle

# Verificar K-Means
with open('../models/kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)
    print(f"Clusters no modelo: {kmeans.n_clusters}")
    print(f"Centroides shape: {kmeans.cluster_centers_.shape}")