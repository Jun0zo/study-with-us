import torch
import pandas as pd
from transformers import BertTokenizer, BertModel

# Step 1: Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Step 2: Read the CSV file containing review data
# Replace this with the path to your CSV file
csv_file_path = "./train_data.csv"
df = pd.read_csv(csv_file_path)

# Step 3: Get the reviews from the first column
reviews = df.iloc[:, 0].tolist()

# Step 4: Tokenization and Encoding
encoded_texts = tokenizer(reviews, padding=True,
                          truncation=True, return_tensors="pt")
input_ids = encoded_texts["input_ids"]
attention_mask = encoded_texts["attention_mask"]

# Step 5: Inference
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)

# Step 6: Get embedded token representations
# (batch_size, sequence_length, hidden_size)
token_embeddings = outputs.last_hidden_state
print(type(token_embeddings))

# Step 7: Process and analyze the token embeddings for each review
for i, review in enumerate(reviews):
    print("Review", i + 1, ":", review)
    review_tokens = tokenizer.tokenize(review)
    for token_idx, token in enumerate(review_tokens):
        # Skip the [CLS] token
        token_embedding = token_embeddings[i, token_idx + 1]
        # Perform analysis or use the embeddings for your specific task here
        print("Token:", token, "Embedding shape:", token_embedding.shape)
        # You can process the token_embedding vector or use it for your analysis

# Note: The indexing for token_embeddings is [i, token_idx + 1] to skip the [CLS] token, which is at position 0.
# If you want to include the [CLS] token as well, you can use [i, token_idx].
