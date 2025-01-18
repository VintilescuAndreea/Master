
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import matplotlib.pyplot as plt

class MultiLabelDataset(Dataset):
    def __init__(self, root_dir, subset, label_mapping=None, transform=None, label_format="txt"):
        """
        Args:
            root_dir (string): Directory with all the images (root of 'train', 'test', 'validation').
            subset (string): Subset folder (train, test, validation).
            label_mapping (dict, optional): A dictionary mapping label names to integer values.
            transform (callable, optional): Optional transform to be applied on an image.
            label_format (string): The format of the labels ('txt' or 'json').
        """
        self.root_dir = os.path.join(root_dir, subset)
        self.transform = transform
        self.label_format = label_format
        self.label_mapping = label_mapping or {}  # Initialize label mapping if not provided
        self.image_paths = []
        self.labels = []
        self.num_classes = 10
        
        for img_file in os.listdir(self.root_dir):
            if img_file.endswith(".jpg") or img_file.endswith(".png"):  # Check for image files
                img_path = os.path.join(self.root_dir, img_file)
                img_name = os.path.splitext(img_file)[0]

                # Determine the label file path
                if self.label_format == "txt":
                    label_file = os.path.join(self.root_dir, f"{img_name}.txt")
                elif self.label_format == "json":
                    label_file = os.path.join(self.root_dir, f"{img_name}.json")
                else:
                    continue  # Skip unsupported formats

                # Check if both image and label file exist
                if not os.path.exists(label_file):
                    continue  # Skip this entry if the label file is missing

                # Load labels
                try:
                    if self.label_format == "txt":
                        with open(label_file, 'r') as f:
                            labels = f.read().split()
                    elif self.label_format == "json":
                        with open(label_file, 'r') as f:
                            labels = json.load(f)["labels"]

                    encoded_labels = torch.tensor([self.label_mapping[label] for label in labels])
                    label_vector = torch.zeros(self.num_classes)
                    label_vector[encoded_labels] = 1

                    self.labels.append(label_vector.clone().detach())
                    self.image_paths.append(img_path)
                except (KeyError, ValueError, FileNotFoundError) as e:
                    print(f"Skipping {img_path} due to error: {e}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Open image
        labels = self.labels[idx].clone().detach()  # Convert labels to tensor

        if self.transform:
            image = self.transform(image)  # Apply transforms

        return image, labels


label_mapping = {
    'aeroplane': 0, 
    'bird': 1, 
    'boat': 2, 
    'chair': 3, 
    'cow': 4, 
    'diningtable': 5, 
    'dog': 6, 
    'horse': 7, 
    'sheep': 8, 
    'train': 9
    }


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = MultiLabelDataset(root_dir="dataset", 
                                subset="train", 
                                label_mapping=label_mapping, 
                                transform=transform, 
                                label_format="txt")
  
    image, labels = dataset[2]
    image = image.permute(1, 2, 0) 
    plt.imshow(image)
    plt.show()
    print(image.size(), labels)
