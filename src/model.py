import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Function to load PhoBERT model and tokenizer
def load_model(model_name="vinai/phobert-base"):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded and moved to device: {device}")
    return model, tokenizer, device

# Initialize the model and tokenizer when needed
if __name__ == "__main__":
    model, tokenizer, device = load_model()