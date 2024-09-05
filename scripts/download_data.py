import os
from datasets import load_dataset

def download_dataset():
    # Set the dataset name and repository
    dataset_name = "learning-ivs/lennon100-range-tau-10k"

    # Create a directory to store the dataset
    os.makedirs("data", exist_ok=True)

    # Load and download the dataset
    dataset = load_dataset(dataset_name)

    # Save the dataset to disk
    dataset.save_to_disk("data/lennon100-range-tau-10k")

    print(f"Dataset '{dataset_name}' has been downloaded and saved to 'data/lennon100-range-tau-10k'")

if __name__ == "__main__":
    download_dataset()
