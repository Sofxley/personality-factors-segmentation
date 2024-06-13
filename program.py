import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import sys

def train_and_run_model(data_path, ):
    data = pd.read_csv(data_path,delimiter="\t")
    columns_to_drop = ['age', 'gender', 'accuracy', 'country', 'source', 'elapsed']
    data = data.drop(columns=columns_to_drop)
    null_rows = data[data['country'].isnull()]
    data = data.drop(null_rows.index,axis=0)

    # Encoding the categorical features
    label_encoder = LabelEncoder()
    data['country_encoded'] = label_encoder.fit_transform(data['country'].astype(str))
    data = data.drop(['country'], axis=1)

    # Standard scaling all features
    std = StandardScaler()
    data_std = pd.DataFrame(std.fit_transform(data), columns=data.columns)

    # Reducing dimensions to 3.
    pca = PCA(n_components=3, random_state=12)
    data_reduced = pd.DataFrame(pca.fit_transform(data_std),)

    # Train model
    kmeans = KMeans(n_clusters=5, random_state=12, n_init='auto')
    data_copy['kmeans_cluster'] = kmeans.fit_predict(data_reduced) # Fit the model and predict clusters
    data_copy['kmeans_cluster'] = data_copy['kmeans_cluster'].astype('category')

    # Export data & model
    data_copy.to_csv('data_with_groups.csv', index=False)
    with open('kmeans_model.pkl','wb') as f:
        pickle.dump(kmeans, f)

    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python program.py <data_path>")
        sys.exit(1)
    data_path = sys.argv[1]
    train_and_run_model(data_path)