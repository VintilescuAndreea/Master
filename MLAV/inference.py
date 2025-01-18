import torch
from torchvision import transforms
from PIL import Image
from train import ResNetMultiLabel, num_classes, label_mapping
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetMultiLabel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('resnet_model_lr_0.0001_bs_32_criterion_BCEWithLogitsLoss_accuracy_0.543_f1_0.641_precision_0.781_recall_0.564.pth',
                                 map_location=torch.device('cpu')))
model.eval() 


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),         
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def run_inference(image_path, threshold=0.3):

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)


    image = image.to(device)

    with torch.no_grad(): 
        outputs = model(image)
        probabilities = torch.sigmoid(outputs)

    # Get the predicted classes
    predicted_labels = (probabilities > threshold).int() 

    return predicted_labels.squeeze(0).cpu().numpy()
## dataset/train/ABD_AAG_AG002568-001.jpg
## dataset/validation/DOR_BRC_BORGM_01317-001.jpg
image_path = 'dataset/train/ABD_AAG_AG002568-001.jpg'
image = plt.imread(image_path)
plt.imshow(image)
plt.show()
predicted_labels = run_inference(image_path)
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
inverse_label_mapping[7]='house'
predicted_class_names = [inverse_label_mapping[i] for i in range(len(predicted_labels)) if predicted_labels[i] == 1]
print(f'Predicted labels: {predicted_class_names}')
