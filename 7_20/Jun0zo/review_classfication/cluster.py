import pandas as pd
import torch
import pickle
from transformers import BertTokenizer, BertModel
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Read the CSV and extract the first column (review data)
def read_review_data_csv(csv_file):
    df = pd.read_csv(csv_file)
    review_data = df.iloc[:, 0].tolist()
    return review_data

def calculate_bert_embeddings(review_data):
    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Initialize lists to store embeddings and attention masks
    all_embeddings = []

    for review in review_data:
        # Tokenize the review text and convert to tensor
        encoded_review = tokenizer.encode_plus(review, add_special_tokens=True, max_length=512, padding='max_length',
                                               truncation=True, return_tensors='pt')

        input_ids = encoded_review['input_ids']
        attention_mask = encoded_review['attention_mask']

        # Generate the BERT embeddings for the review
        with torch.no_grad():
            model_output = model(input_ids, attention_mask=attention_mask)

        all_hidden_states = model_output.last_hidden_state
        all_token_embeddings = all_hidden_states[:, 1:-1, :].numpy()  # Exclude [CLS] and [SEP] tokens
        all_embeddings.extend(all_token_embeddings)

    # all_embeddings = np.concatenate(all_embeddings, axis=0)  # Concatenate along the first axis (samples)
    all_embeddings = np.array(all_embeddings)
    print(all_embeddings.shape)
    # Save the embeddings to a file
    with open('bert_embeddings.pkl', 'wb') as f:
        pickle.dump(all_embeddings, f)

    return all_embeddings

# Step 3: Cluster the embeddings using EM Clustering
def em_clustering(embeddings, n_clusters=-1):
    print("embeddings", embeddings.shape)
    embeddings = np.reshape(embeddings, (embeddings.shape[0], -1))
    print("embeddings after reshape", embeddings.shape)

    pca = PCA(n_components=2)
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

def visualize_clusters(embeddings_2d, cluster_labels):
    # plot the 2D points
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='viridis')
    plt.savefig('clusters.png')



if __name__ == "__main__":
    csv_file_path = "./train_data.csv"  # Replace with the actual path to your CSV file
    n_clusters = 5  # Specify the number of clusters for EM Clustering

    # Step 1
    review_data = read_review_data_csv(csv_file_path)

    # Step 2
    # embeddings = calculate_bert_embeddings(review_data)

    with open('bert_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    # Step 3
    print('EM Clustering...')
    cluster_labels, embeddings_2d = em_clustering(embeddings, n_clusters = 5)
    print(len(cluster_labels))


    # Visualize the clusters
    visualize_clusters(embeddings_2d, cluster_labels)

    # Print the cluster labels for each review
    print('visualize_clusters')
    print(cluster_labels)
