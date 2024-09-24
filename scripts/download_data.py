import os
import psutil
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

def download_dataset():
    # Set the dataset name and repository
    dataset_name = "learning-ivs/lennon100-range-tau-10k"

    # Create a directory to store the dataset
    os.makedirs("data", exist_ok=True)

    # Load and download the dataset
    dataset = load_dataset(dataset_name)

    # Check memory usage before saving
    before_mem = psutil.virtual_memory().used

    # Save the dataset to disk
    dataset.save_to_disk("data/lennon100-range-tau-10k")

    # Check memory usage after saving
    after_mem = psutil.virtual_memory().used

    # Calculate memory usage
    memory_used = after_mem - before_mem

    # # prefer huggingface repo
    # print(f"Downloading dataset from HF: {dataset_name}...")
    # data_path = snapshot_download(repo_id=dataset_name, repo_type="dataset")

    print(f"Dataset '{dataset_name}' has been downloaded and saved to data/lennon100-range-tau-10k")
    print(f"Memory used: {memory_used / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    download_dataset()
