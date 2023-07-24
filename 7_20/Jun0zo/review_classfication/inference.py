import torch
from transformers import BertTokenizer, BertModel

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Set the number of Monte Carlo Dropout samples (forward passes)
num_mc_samples = 10

def mc_dropout_inference(model, input_ids, attention_mask, num_samples):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        all_outputs = []
        for _ in range(num_samples):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_outputs.append(outputs[0])  # Output representations from the model

        # Stack all the outputs along a new dimension (for averaging)
        stacked_outputs = torch.stack(all_outputs, dim=0)

        # Average the output representations across the Monte Carlo samples
        averaged_output = torch.mean(stacked_outputs, dim=0)

    return averaged_output

# Example usage:
input_text = "Example text for illustration purposes."
input_ids = torch.tensor([tokenizer.encode(input_text, add_special_tokens=True)])
attention_mask = torch.ones_like(input_ids)

averaged_representation = mc_dropout_inference(model, input_ids, attention_mask, num_mc_samples)

# Now, 'averaged_representation' contains the final representation of the input text,
# with Monte Carlo Dropout applied for uncertainty estimation.
