import os
import requests
import csv
import json


base_dir = "dataset"


subsets = ["train", "test", "validation"]
for subset in subsets:
    os.makedirs(os.path.join(base_dir, subset), exist_ok=True)

def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            return True
        else:
            print(f"Failed to download {url}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def save_labels(image_name, labels, subset_folder, format="txt"):
    if format == "txt":
        label_file = os.path.join(subset_folder, f"{image_name}.txt")
        with open(label_file, 'w') as f:
            f.write(" ".join(labels))
    elif format == "json":
        label_file = os.path.join(subset_folder, f"{image_name}.json")
        with open(label_file, 'w') as f:
            json.dump({"labels": labels}, f)


csv_file = "painting_dataset_2021.csv"  # Path to your CSV file
limit = 15000
i= 0
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        i +=1
        if i == limit:
            break
        image_url = row['Image URL']
        subset = row['Subset'].strip().lower()[1:-1]  # Ensure lowercase for folder names
        labels = row['Labels'][1:-1].strip().split()  # Split labels by space
        
        # Image name (last part of URL)
        image_name = image_url.split('/')[-1]
        
        # Define subset folder (train, test, validation)
        subset_folder = os.path.join(base_dir, subset)
        
        # Download the image and save
        image_save_path = os.path.join(subset_folder, image_name)
        if download_image(image_url, image_save_path):
            # If the image is successfully downloaded, save its labels
            save_labels(image_name.split('.')[0], labels, subset_folder, format="txt")

print("Dataset download and organization completed.")
