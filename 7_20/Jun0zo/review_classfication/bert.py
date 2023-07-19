import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel

def encode_texts_with_bert_cls(texts):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    all_embeddings = []

    for text in texts:
        tokens = tokenizer.tokenize(text)
        total_tokens = len(tokens)

        
        print('tockens', total_tokens)
        encoded_text = tokenizer.encode_plus(tokens, add_special_tokens=True, max_length=512, return_tensors='pt', truncation=True)

        input_ids = encoded_text['input_ids']
        attention_mask = encoded_text['attention_mask']
        
        # add padding
        # if input_ids.shape[1] < max_length:
        #     padding = torch.zeros((input_ids.shape[0], max_length - input_ids.shape[1]), dtype=torch.long)
        #     input_ids = torch.cat((input_ids, padding), dim=1)
        #     attention_mask = torch.cat((attention_mask, padding), dim=1)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # Get the output embeddings (last layer hidden states)

        cls_embedding = outputs.last_hidden_state[0]
        print('emb', cls_embedding.shape, outputs.last_hidden_state.shape)
        # print('emb', embeddings.shape)
        all_embeddings.append(cls_embedding)

    # embeddings = torch.cat(all_embeddings, dim=0)
    embeddings = torch.stack(all_embeddings)
    return embeddings

    
def encode_texts_with_bert_sliding_window(texts, max_length=512, window_size=256):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    all_embeddings = []

    for text in texts:
        # Tokenize the text
        tokens = tokenizer.tokenize(text)
        total_tokens = len(tokens)

        text_embeddings = []
        

        for i in range(0, total_tokens, window_size):
            window = tokens[i:i+window_size]
            window_text = " ".join(window)

            # Encode the window text and add special tokens [CLS] and [SEP]
            encoded_text = tokenizer.encode_plus(window_text, add_special_tokens=True, max_length=max_length, return_tensors='pt', truncation=True)

            input_ids = encoded_text['input_ids']
            attention_mask = encoded_text['attention_mask']

            # Make sure input_ids and attention_mask have the same length (pad if needed)
            if input_ids.shape[1] < max_length:
                padding = torch.zeros((input_ids.shape[0], max_length - input_ids.shape[1]), dtype=torch.long)
                input_ids = torch.cat((input_ids, padding), dim=1)
                attention_mask = torch.cat((attention_mask, padding), dim=1)

            # Encode the window using the BERT model
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)

            # Get the output embeddings (last layer hidden states)
            embeddings = outputs.last_hidden_state
            print('emb', embeddings.shape)
            text_embeddings.append(embeddings)

        # Merge embeddings for all windows in the text
        text_embeddings = torch.cat(text_embeddings, dim=1)
        print('texe', text_embeddings.shape)
        if text_embeddings.shape[1] > max_length:
            # text_embeddings = text_embeddings[:, :max_length, :]
            continue
        all_embeddings.append(text_embeddings)

    # Concatenate embeddings for all texts
    embeddings = torch.cat(all_embeddings, dim=0)

    return embeddings

def encode_cls_texts_with_bert_sliding_window(texts, max_length=512, window_size=256):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    all_embeddings = []

    for text in texts:
        # Tokenize the text
        tokens = tokenizer.tokenize(text)
        total_tokens = len(tokens)

        text_embeddings = []
        

        for i in range(0, total_tokens, window_size):
            window = tokens[i:i+window_size]
            window_text = " ".join(window)

            # Encode the window text and add special tokens [CLS] and [SEP]
            encoded_text = tokenizer.encode_plus(window_text, add_special_tokens=True, max_length=max_length, return_tensors='pt', truncation=True)

            input_ids = encoded_text['input_ids']
            attention_mask = encoded_text['attention_mask']

            # Make sure input_ids and attention_mask have the same length (pad if needed)
            if input_ids.shape[1] < max_length:
                padding = torch.zeros((input_ids.shape[0], max_length - input_ids.shape[1]), dtype=torch.long)
                input_ids = torch.cat((input_ids, padding), dim=1)
                attention_mask = torch.cat((attention_mask, padding), dim=1)

            # Encode the window using the BERT model
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)

            # Get the output embeddings (last layer hidden states)
            embeddings = outputs.last_hidden_state[:, 0]
            # print('emb', embeddings.shape)
            text_embeddings.append(embeddings)

        # Merge embeddings for all windows in the text
        text_embeddings = torch.cat(text_embeddings, dim=1)
        # print('texe', text_embeddings.shape)
        all_embeddings.append(text_embeddings)

    # Concatenate embeddings for all texts
    embeddings = torch.cat(all_embeddings, dim=0)

    return embeddings


# Step 2: Read the CSV file containing review data
# Replace this with the path to your CSV file
csv_file_path = "./train_data.csv"
df = pd.read_csv(csv_file_path)

# Step 3: Get the reviews from the first column
reviews = df.iloc[:, 0].tolist()

# token_embeddings = encode_texts_with_bert_sliding_window(reviews)
token_embeddings = encode_texts_with_bert_cls(reviews)

# The embeddings tensor contains the encoded representation of the input reviews
print(token_embeddings.shape)

flattened_embeddings = token_embeddings.view(-1, token_embeddings.size(-1))
print(flattened_embeddings.shape)

# Step 7: Process and analyze the token embeddings for each review


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Step 1: Perform EM Clustering
num_clusters = 5  # Specify the desired number of clusters
gmm = GaussianMixture(n_components=num_clusters)

# Step 2: Fit the Clustering Model
gmm.fit(token_embeddings.numpy())

# Step 3: Assign Cluster Labels
cluster_labels = gmm.predict(token_embeddings.numpy())

# Step 4: Visualize the Clusters
# Reduce the dimensionality of the embeddings for visualization (e.g., using PCA or t-SNE)
# Here, we'll use PCA for simplicity
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
token_embeddings_2d = pca.fit_transform(token_embeddings.numpy())

# Create a scatter plot of the token embeddings, colored by cluster labels
plt.scatter(token_embeddings_2d[:, 0], token_embeddings_2d[:, 1], c=cluster_labels)
plt.title("EM Clustering of Token Embeddings")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
