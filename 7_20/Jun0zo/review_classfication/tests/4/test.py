import os
import pandas as pd
import torch
import pickle
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Step 1: Read the CSV and extract the first column (review data)
def read_review_data_csv(csv_file):
    df = pd.read_csv(csv_file)
    review_data = df.iloc[:, 0].tolist()
    label_data = df.iloc[:, 1].tolist()
    # convert label_data (korean to number)
    for i in range(len(label_data)):
        if label_data[i] == '높음':
            label_data[i] = 2
        elif label_data[i] == '보통':
            label_data[i] = 1
        elif label_data[i] == '낮음':
            label_data[i] = 0
    # convert label data to numpy array
    label_data = np.array(label_data)
    return review_data, label_data

def calculate_bert_embeddings(review_data):
    # Load BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
    model = AutoModel.from_pretrained("skt/kobert-base-v1")

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
    with open('bert_kor_embeddings.pkl', 'wb') as f:
        pickle.dump(all_embeddings, f)

    return all_embeddings

# Step 3: Cluster the embeddings using EM Clustering
def em_clustering(embeddings, n_clusters=-1, depth=3):
    print("embeddings", embeddings.shape)
    embeddings = np.reshape(embeddings, (embeddings.shape[0], -1))
    print("embeddings after reshape", embeddings.shape)

    # pca = PCA(n_components=2)
    # embeddings_2d = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=depth, learning_rate=500, perplexity=50, n_iter=2000)
    embeddings = tsne.fit_transform(embeddings)
    print("embeddings after tsne", embeddings.shape)

    # Try different number of clusters and calculate AIC and BIC

    if n_clusters == -1:
        n_clusters_range = range(2, 11)
        aic_values = []
        bic_values = []

        for n_clusters in n_clusters_range:
            gmm = GaussianMixture(n_components=n_clusters)
            gmm.fit(embeddings)

            # Calculate AIC and BIC values
            aic_values.append(gmm.aic(embeddings))
            bic_values.append(gmm.bic(embeddings))

        # Find the optimal number of clusters using AIC and BIC
        optimal_clusters_aic = np.argmin(aic_values) + 2  # +2 is due to n_clusters_range starting from 2
        optimal_clusters_bic = np.argmin(bic_values) + 2

    else:
        optimal_clusters_aic = n_clusters
    print('optimal_clusters_aic', optimal_clusters_aic)


    # Perform EM Clustering with the optimal number of clusters (chosen based on AIC)
    gmm_optimal = GaussianMixture(n_components=optimal_clusters_aic)
    gmm_optimal.fit(embeddings)
    cluster_labels = gmm_optimal.predict(embeddings)

    return cluster_labels, embeddings

def visualize_clusters(embeddings, cluster_labels, depth=3, tag='inference'):
    print("=====================================")
    print("visualize_clusters")
    print("embeddings", embeddings.shape)
    print("cluster_labels", cluster_labels.shape)
    print(cluster_labels[:20])
    if depth == 2:
        # Plot the 2D points without cluster labels
        print(embeddings[:, 0].shape, embeddings[:, 1].shape)
        plt.scatter(embeddings[:, 0], embeddings[:, 1])

        # Set axis labels and title
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('t-SNE Embeddings in 2D')

        # Save the plot to a file
        plt.savefig(f'embeddings_tsne_2d_{tag}.png')

    if depth == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the 3D points without cluster labels
        # add cluster labels
        ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=cluster_labels, cmap='viridis')

        # Set axis labels and title
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_zlabel('t-SNE Dimension 3')
        ax.set_title('t-SNE Embeddings in 3D')

        # Save the plot to a file
        plt.savefig(f'embeddings_tsne_3d_{tag}.png')

        # save plots rotated with for loop and save images
        '''for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.001)
            plt.savefig("movie/movie%d.png" % angle)'''


if __name__ == "__main__":
    csv_file_path = "./train_data.csv"  # Replace with the actual path to your CSV file
    n_clusters = 5  # Specify the number of clusters for EM Clustering

    # Step 1
    review_data, label_data = read_review_data_csv(csv_file_path)
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
    print('EM Clustering...')
    cluster_labels_3d, embeddings_3d = em_clustering(embeddings, depth=3, n_clusters = -1)
    cluster_labels_2d, embeddings_2d = em_clustering(embeddings, depth=2, n_clusters = -1)


    # Visualize the clusters
    visualize_clusters(embeddings_3d, cluster_labels_3d, depth=3, tag='inference')
    visualize_clusters(embeddings_3d, label_data, depth=3, tag='answer')

    # visualize_clusters(embeddings_2d, cluster_labels_2d, depth=2, tag='inference')
    visualize_clusters(embeddings_2d, label_data, depth=2, tag='answer')

    # Print the cluster labels for each review
    # print('visualize_clusters')
    # print(cluster_labels_3d)

    # Step 4 (Optional): Save the cluster labels to a CSV file
    # i want to save the cluster labels to a csv file (read original file and add a column and save to new file)
    df = pd.read_csv(csv_file_path)
    df['cluster_labels'] = cluster_labels_3d
    df.to_csv('train_data_with_clusters.csv', index=False)
    print('Done!')
