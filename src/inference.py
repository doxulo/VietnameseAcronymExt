from src.model import load_model

# Function to test if model is working by tokenizing a sample acronym
def test_model_loading():
    model, tokenizer, device = load_model()
    
    # Test with a sample acronym input (like "aye")
    sample_acronym = "aye"
    inputs = tokenizer(sample_acronym, return_tensors="pt").to(device)
    outputs = model(**inputs)
    
    print("Model test successful! Output:", outputs)

# Run the test
if __name__ == "__main__":
    test_model_loading()