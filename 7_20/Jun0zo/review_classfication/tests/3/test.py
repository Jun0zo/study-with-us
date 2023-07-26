import os
import pandas as pd
import torch
import pickle
from transformers import BertTokenizer, BertModel
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

def tsne_embedding(embeddings):
    tsne = TSNE(n_components=3, learning_rate=1000, perplexity=50, n_iter=2000)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d

def visualize_clusters(embeddings_2d):
    # Create a scatter plot
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], cmap='viridis')

    # Set axis labels and title
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Clustering')

    # Save the plot to a file
    plt.savefig('clusters_tsne.png')

def visualize_clusters_3d(embeddings_3d):
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D points without cluster labels
    ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], cmap='viridis')

    # Set axis labels and title
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    ax.set_title('t-SNE Embeddings in 3D')

    # Save the plot to a file
    plt.savefig('embeddings_tsne_3d.png')

# ... (rest of the code remains the same)

if __name__ == "__main__":
    csv_file_path = "./train_data.csv"  # Replace with the actual path to your CSV file

    # Step 1
    review_data = read_review_data_csv(csv_file_path)
    print(type(review_data))

    # Step 2
    if not os.path.exists('bert_kor_embeddings.pkl'):
        print('Calculating BERT embeddings...')
        embeddings = calculate_bert_embeddings(review_data)
    else:
        print('Loading BERT embeddings...')
        with open('bert_kor_embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)


    # Step 3
    embeddings_2d = tsne_embedding(embeddings)


    # Visualize the clusters
    visualize_clusters(embeddings_2d)
    visualize_clusters_3d(embeddings_2d)

    # Print the cluster labels for each review
    print('visualize_clusters')
    '''print(cluster_labels)

    # Step 4 (Optional): Save the cluster labels to a CSV file
    df = pd.read_csv(csv_file_path)
    df['cluster_labels'] = cluster_labels
    df.to_csv('train_data_with_clusters.csv', index=False)
    print('Done!')'''


