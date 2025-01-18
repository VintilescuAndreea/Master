import os
import json

def collect_unique_labels(root_dir, subsets=["train", "test"], label_format="txt"):
    unique_labels = set()
    
    for subset in subsets:
        subset_folder = os.path.join(root_dir, subset)
        
        for img_file in os.listdir(subset_folder):
            if img_file.endswith(".jpg") or img_file.endswith(".png"):
                img_name = os.path.splitext(img_file)[0]
                

                if label_format == "txt":
                    label_file = os.path.join(subset_folder, f"{img_name}.txt")
                    with open(label_file, 'r') as f:
                        labels = f.read().split() 
                elif label_format == "json":
                    label_file = os.path.join(subset_folder, f"{img_name}.json")
                    with open(label_file, 'r') as f:
                        labels = json.load(f)["labels"]
                
                unique_labels.update(labels) 
    
    return sorted(list(unique_labels))  

def create_label_mapping(root_dir, subsets=["train", "test"], label_format="txt"):

    unique_labels = collect_unique_labels(root_dir, subsets, label_format)

    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    
    return label_mapping


if __name__ == "__main__":
    root_dir = "dataset" 
    label_mapping = create_label_mapping(root_dir, subsets=["train", "test"], label_format="txt")
    
    print("Label Mapping:", label_mapping)

    with open('label_mapping.json', 'w') as f:
        json.dump(label_mapping, f)

