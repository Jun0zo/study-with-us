import pandas as pd
import torch
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

# Step 2: Calculate BERT embeddings for each review
def calculate_bert_embeddings(review_data):
    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Initialize lists to store embeddings and attention masks
    all_embeddings = []

    for review in review_data:
        # Tokenize the review text and convert to tensor
        print('review :', review)
        encoded_review = tokenizer.encode_plus(review, add_special_tokens=True, max_length=512, padding='max_length',
                                               truncation=True, return_tensors='pt')

        input_ids = encoded_review['input_ids']
        attention_mask = encoded_review['attention_mask']

        # Generate the BERT embeddings for the review
        with torch.no_grad():
            model_output = model(input_ids, attention_mask=attention_mask)

        all_hidden_states = model_output.last_hidden_state
        print('shape :', all_hidden_states.shape)
        # all_token_embeddings = torch.stack(all_hidden_states, dim=2)
        all_token_embeddings = all_hidden_states
        # concatenated_embedding = all_token_embeddings[:, 0, :].numpy()
        concatenated_embedding = all_token_embeddings.view(-1, all_token_embeddings.size(-1)).numpy()
        print('concat shape :', concatenated_embedding.shape)

        all_embeddings.append(concatenated_embedding)

    all_embeddings = np.vstack(all_embeddings)

    return all_embeddings

# Step 3: Cluster the embeddings using EM Clustering
def em_clustering(embeddings, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(embeddings)
    cluster_labels = gmm.predict(embeddings)
    return cluster_labels

def visualize_clusters(embeddings, cluster_labels):
    # Reduce the dimensionality of embeddings to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Create a DataFrame for easier plotting
    df = pd.DataFrame(embeddings_2d, columns=['PC1', 'PC2'])
    df['Cluster'] = cluster_labels

    # Plot the clusters
    plt.figure(figsize=(10, 8))
    for cluster in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster]
        plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}', alpha=0.7)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('EM Clustering of BERT Embeddings')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    csv_file_path = "./train_data.csv"  # Replace with the actual path to your CSV file
    n_clusters = 5  # Specify the number of clusters for EM Clustering

    # Step 1
    review_data = read_review_data_csv(csv_file_path)

    # Step 2
    embeddings, _ = calculate_bert_embeddings(review_data)

    # Step 3
    cluster_labels = em_clustering(embeddings, n_clusters)

    # Visualize the clusters
    visualize_clusters(embeddings, cluster_labels)

    # Print the cluster labels for each review
    print(cluster_labels)
