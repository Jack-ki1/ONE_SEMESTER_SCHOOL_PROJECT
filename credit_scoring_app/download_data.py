import os
import requests

def download_german_credit_dataset():
    """
    Download the German Credit dataset from UCI ML Repository and place it in the data folder
    """
    # Define the URLs for the German Credit dataset
    base_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/"
    files_to_download = {
        "german.data": "german_credit_raw.data",
        "german.data-numeric": "german_credit_numeric.data",
        "german.doc": "german_credit_description.txt"
    }
    
    # Define the data folder path
    data_folder = "data"
    
    # Create data folder if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)
    
    print("Downloading German Credit dataset...")
    
    for remote_file, local_file in files_to_download.items():
        remote_url = base_url + remote_file
        local_path = os.path.join(data_folder, local_file)
        
        try:
            print(f"Downloading {remote_file} -> {local_file}...")
            response = requests.get(remote_url)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Successfully downloaded {local_file}")
        except Exception as e:
            print(f"Failed to download {remote_file}: {str(e)}")
    
    print("\nDataset download completed!")

if __name__ == "__main__":
    download_german_credit_dataset()