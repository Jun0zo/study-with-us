import os
import pandas as pd
import torch
import pickle
from transformers import BertTokenizer, BertModel
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util

# Step 1: Read the CSV and extract the first column (review data)
def read_review_data_csv(csv_file):
    df = pd.read_csv(csv_file)
    review_data = df.iloc[:, 0].tolist()
    return review_data

def calculate_bert_embeddings(review_data):
    # Load BERT model and tokenizer

    # model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens') 
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    # Initialize lists to store embeddings and attention masks
    all_embeddings = model.encode(review_data)
    print('emb shape', all_embeddings.shape)

    with open('bert_kor_embeddings.pkl', 'wb') as f:
        pickle.dump(all_embeddings, f)

    return all_embeddings

# Step 3: Cluster the embeddings using EM Clustering
def em_clustering(embeddings, n_clusters=-1):
    print("embeddings", embeddings.shape)
    embeddings = np.reshape(embeddings, (embeddings.shape[0], -1))
    print("embeddings after reshape", embeddings.shape)

    pca = PCA(n_components=3)
    embeddings_2d = pca.fit_transform(embeddings)

    # Try different number of clusters and calculate AIC and BIC

    if n_clusters == -1:
        n_clusters_range = range(2, 11)
        aic_values = []
        bic_values = []

        for n_clusters in n_clusters_range:
            gmm = GaussianMixture(n_components=n_clusters)
            gmm.fit(embeddings_2d)

            # Calculate AIC and BIC values
            aic_values.append(gmm.aic(embeddings_2d))
            bic_values.append(gmm.bic(embeddings_2d))

        # Plot AIC and BIC values
        # plt.plot(n_clusters_range, aic_values, label='AIC')
        # plt.plot(n_clusters_range, bic_values, label='BIC')
        # plt.xlabel('Number of Clusters')
        # plt.ylabel('AIC/BIC Value')
        # plt.title('AIC and BIC for EM Clustering')
        # plt.legend()
        # plt.savefig('AIC_BIC.png')

        # Find the optimal number of clusters using AIC and BIC
        optimal_clusters_aic = np.argmin(aic_values) + 2  # +2 is due to n_clusters_range starting from 2
        optimal_clusters_bic = np.argmin(bic_values) + 2

    else:
        optimal_clusters_aic = n_clusters
    print('optimal_clusters_aic', optimal_clusters_aic)


    # Perform EM Clustering with the optimal number of clusters (chosen based on AIC)
    gmm_optimal = GaussianMixture(n_components=optimal_clusters_aic)
    gmm_optimal.fit(embeddings_2d)
    cluster_labels = gmm_optimal.predict(embeddings_2d)

    return cluster_labels, embeddings_2d

def visualize_clusters(embeddings_3d, cluster_labels):
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D points
    ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=cluster_labels, cmap='viridis')

    # show other views 
    ax.view_init(30, 30)
    plt.savefig('clusters_3d_1.png')
    ax.view_init(30, 60)
    plt.savefig('clusters_3d_2.png')
    ax.view_init(30, 90)
    plt.savefig('clusters_3d_3.png')
    ax.view_init(30, 120)
    plt.savefig('clusters_3d_4.png')
    ax.view_init(30, 150)
    plt.savefig('clusters_3d_5.png')
    ax.view_init(30, 180)
    plt.savefig('clusters_3d_6.png')
    


    # Add labels to each point
    #for i, txt in enumerate(cluster_labels):
    #   ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], str(txt))

    # Set axis labels and title
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('EM Clustering in 3D')

    # Save the plot to a file
    plt.savefig('clusters_3d.png')



if __name__ == "__main__":
    csv_file_path = "./train_data.csv"  # Replace with the actual path to your CSV file
    n_clusters = 5  # Specify the number of clusters for EM Clustering

    # Step 1
    review_data = read_review_data_csv(csv_file_path)
    print(type(review_data))

    # Step 2
    if not os.path.exists('models/bert_kor_embeddings.pkl'):
        print('Calculating BERT embeddings...')
        embeddings = calculate_bert_embeddings(review_data)
    else:
        print('Loading BERT embeddings...')
        with open('models/bert_embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)

    

    # Step 3
    print('EM Clustering...')
    cluster_labels, embeddings_2d = em_clustering(embeddings, n_clusters = -1)
    print(len(cluster_labels))


    # Visualize the clusters
    visualize_clusters(embeddings_2d, cluster_labels)

    # Print the cluster labels for each review
    print('visualize_clusters')
    print(cluster_labels)

    # Step 4 (Optional): Save the cluster labels to a CSV file
    # i want to save the cluster labels to a csv file (read original file and add a column and save to new file)
    df = pd.read_csv(csv_file_path)
    df['cluster_labels'] = cluster_labels
    df.to_csv('train_data_with_clusters.csv', index=False)
    print('Done!')


