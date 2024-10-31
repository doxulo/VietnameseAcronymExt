import pandas as pd
from transformers import AutoTokenizer

# Load the tokenizer for PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

def load_data(file_path):
    """
    Load the acronym dataset from a CSV file.
    Args:
        file_path (str): Path to the CSV file with acronyms and expansions.
    Returns:
        DataFrame: Loaded data as a pandas DataFrame.
    """
    data = pd.read_csv(file_path)
    return data

def tokenize_data(data):
    """
    Tokenize acronyms and expansions for PhoBERT.
    Args:
        data (DataFrame): DataFrame with 'Acronym' and 'Expansion' columns.
    Returns:
        List[Dict]: List of tokenized data dictionaries.
    """
    tokenized_data = []
    for _, row in data.iterrows():
        # Tokenize acronym and expansion
        input_encoding = tokenizer(row['acronym'], truncation=True, padding="max_length", max_length=12)
        target_encoding = tokenizer(row['expansion'], truncation=True, padding="max_length", max_length=20)
        
        # Create a dictionary for each example
        tokenized_data.append({
            "input_ids": input_encoding["input_ids"],
            "attention_mask": input_encoding["attention_mask"],
            "labels": target_encoding["input_ids"]
        })
    
    return tokenized_data

# Example usage
if __name__ == "__main__":
    # Load and tokenize the data
    data = load_data("data/acronym_data.csv")
    tokenized_data = tokenize_data(data)
    print("Data loaded and tokenized!")

    # Print first few tokenized samples for verification
    for sample in tokenized_data[:3]:  # Print first 3 samples as examples
        print(sample)