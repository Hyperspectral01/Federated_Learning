import os
import torch
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
from torchvision import transforms
import math, random
import SimpleITK as sitk
from scipy.ndimage import label, find_objects
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torchvision.models as models



###################################################### MODEL ########################################3
class sample_CNN_model(nn.Module):

    def __init__(self):
      super(sample_CNN_model, self).__init__()

      # Load the pretrained ResNet18 model
      self.model = models.resnet18(pretrained=True)

      # Ensure all layers require gradients
      for param in self.model.parameters():
          param.requires_grad = True

      # Modify the final fully connected layer
      num_features = self.model.fc.in_features
      self.model.fc = nn.Linear(num_features, 4, bias=True)

      # Initialize the new layer's weights
      # nn.init.kaiming_normal_(self.model.fc.weight, mode='fan_out', nonlinearity='relu')
      nn.init.xavier_uniform_(self.model.fc.weight)  # For the final fully connected layer
      self.model.fc.bias.data.fill_(0.01)

    def forward(self, x):
        return self.model(x)


############################################ CUSTOM DATASET ##########################
class custom_dataset():
    def __init__(self,folder_path=None,transform=None):
        self.image_paths=[]
        self.output_classes=[]
        self.transform=transform
        if (folder_path!=None):
            for categories in os.listdir(folder_path):
              for image_name in os.listdir(os.path.join(folder_path,categories)):
                self.image_paths.append(os.path.join(folder_path,categories,image_name))
                if (categories=="Diverticulosis"):
                  self.output_classes.append(0)
                elif (categories=="Neoplasm"):
                  self.output_classes.append(1)
                elif (categories=="Peritonitis"):
                  self.output_classes.append(2)
                else:
                  self.output_classes.append(3)

            # # Zip the lists together and convert to a list of pairs
            # combined = list(zip(self.image_paths, self.output_classes))

            # # Shuffle the combined list
            # random.shuffle(combined)

            # # Unzip the combined s back into two lists
            # self.image_paths, self.output_classes = zip(*combined)

            # # Convert back to list type
            # self.image_paths = list(self.image_paths)
            # self.output_classes = list(self.output_classes)

    def __len__(self):
        return len(self.image_paths)


    # def bias_field_correction(self, channel_np):
    #     """
    #     Apply bias field correction to a single-channel image.
    #     """
    #     sitk_image = sitk.GetImageFromArray(channel_np)
    #     # Create a mask using Otsu thresholding
    #     mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    #     corrected = sitk.N4BiasFieldCorrection(sitk_image, mask)
    #     corrected_np = sitk.GetArrayFromImage(corrected)
    #     return corrected_np

    # def crop_to_white_blob(self, image_np):
    #     """
    #     Crop the image to the bounding box of the connected component 
    #     (from the white blob) that contains the center of the image.
    #     """
    #     # Convert first three channels (RGB) to grayscale by averaging
    #     gray = np.mean(image_np[:, :, :3], axis=2)
    #     # Create a binary mask; adjust threshold as needed for your data
    #     binary = gray > 200

    #     # Label connected regions and find bounding boxes
    #     labeled, num_features = label(binary)
    #     center_y, center_x = gray.shape[0] // 2, gray.shape[1] // 2
    #     selected_bbox = None
    #     objects = find_objects(labeled)
    #     for i, slice_tuple in enumerate(objects, start=1):
    #         y_slice, x_slice = slice_tuple
    #         # Check if the image center lies within the component's bounding box
    #         if (y_slice.start <= center_y < y_slice.stop) and (x_slice.start <= center_x < x_slice.stop):
    #             selected_bbox = slice_tuple
    #             break

    #     if selected_bbox is not None:
    #         y_slice, x_slice = selected_bbox
    #         cropped = image_np[y_slice, x_slice, :]
    #         return cropped
    #     else:
    #         # If no white blob is detected near the center, return the original image
    #         return image_np

    # def zscore_normalize(self, image_np):
    #     """
    #     Apply per-channel z-score normalization.
    #     """
    #     # Process each channel independently
    #     for c in range(image_np.shape[2]):
    #         channel = image_np[:, :, c]
    #         mean = channel.mean()
    #         std = channel.std()
    #         if std > 0:
    #             image_np[:, :, c] = (channel - mean) / std
    #         else:
    #             image_np[:, :, c] = channel - mean
    #     return image_np

    def __getitem__(self, idx):
        # Load the preprocessed image (assumed to already have the desired size and channels)
        image = Image.open(self.image_paths[idx])
        
        # Optionally apply additional transformations if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Convert image to a numpy array and then to a torch tensor with shape (C, H, W)
            image_np = np.array(image)
            image = torch.from_numpy(image_np.transpose(2, 0, 1))
        
        return image, self.output_classes[idx]


    def take_out_items(self, ratio):

        indices_label0 = [i for i, label in enumerate(self.output_classes) if label == 0]
        indices_label1 = [i for i, label in enumerate(self.output_classes) if label == 1]
        indices_label2 = [i for i, label in enumerate(self.output_classes) if label == 2]
        indices_label3 = [i for i, label in enumerate(self.output_classes) if label == 3]

        num_to_remove_0 = math.floor(len(indices_label0) * ratio)
        num_to_remove_1 = math.floor(len(indices_label1) * ratio)
        num_to_remove_2 = math.floor(len(indices_label2) * ratio)
        num_to_remove_3 = math.floor(len(indices_label3) * ratio)

        selected_indices_0 = random.sample(indices_label0, num_to_remove_0) if num_to_remove_0 > 0 else []
        selected_indices_1 = random.sample(indices_label1, num_to_remove_1) if num_to_remove_1 > 0 else []
        selected_indices_2 = random.sample(indices_label2, num_to_remove_2) if num_to_remove_2 > 0 else []
        selected_indices_3 = random.sample(indices_label3, num_to_remove_3) if num_to_remove_3 > 0 else []

        selected_indices = set(selected_indices_0 + selected_indices_1 + selected_indices_2 + selected_indices_3)

        removed_image_paths = [self.image_paths[i] for i in selected_indices]
        removed_labels = [self.output_classes[i] for i in selected_indices]

        new_image_paths = []
        new_output_classes = []
        for idx, (img_path, label) in enumerate(zip(self.image_paths, self.output_classes)):
            if idx not in selected_indices:
                new_image_paths.append(img_path)
                new_output_classes.append(label)

        self.image_paths = new_image_paths
        self.output_classes = new_output_classes

        return removed_image_paths, removed_labels

    def add_items(self, list_of_image_paths,list_of_labels):
        self.image_paths.extend(list_of_image_paths)
        self.output_classes.extend(list_of_labels)



############################################## MAIN CODE ##############################################################

# ====================== METRICS & PLOTTING ======================
def test_global_model(global_model, test_dataloader):
    """Returns test loss, accuracy, precision, recall, F1"""
    global_model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_dataloader):
            outputs = global_model(images)
            # print("OUTPUTS:")
            # print(outputs)
            # print("LABELS:")
            # print(labels)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    # print("The Accuracy of local model after training: ",accuracy)
    return avg_loss, accuracy, precision, recall, f1

def plot_learning_curves(metrics_history, hyperparams):
    """Plots metrics vs communication rounds for a single hyperparameter configuration"""
    rounds = list(range(1, len(metrics_history['loss'])+1))
    
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"Test Ratio: {hyperparams['test_ratio']} | Batch Size: {hyperparams['batch_size']}\n"
                 f"Rounds: {hyperparams['num_rounds']} | Epochs: {hyperparams['num_epochs']} | LR: {hyperparams['learning_rate']}", 
                 y=1.02)
    
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    titles = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for i, (metric, title) in enumerate(zip(metrics, titles), 1):
        plt.subplot(2, 3, i)
        plt.plot(rounds, metrics_history[metric], marker='o')
        plt.xlabel('Communication Round')
        plt.ylabel(title)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"./out/learning_curves_ratio_{hyperparams['test_ratio']}_bs_{hyperparams['batch_size']}_rounds_{hyperparams['num_rounds']}_epochs_{hyperparams['num_epochs']}_lr_{hyperparams['learning_rate']}.png")
    plt.close()

def plot_hyperparameter_comparisons(all_results):
    """Generates plots comparing metrics across different hyperparameters"""
    hyperparams = ['test_ratio', 'batch_size', 'num_rounds', 'num_epochs', 'learning_rate']
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss']
    
    for hp in hyperparams:
        plt.figure(figsize=(12, 8))
        for metric in metrics:
            x = [res['hyperparams'][hp] for res in all_results]
            y = [res['final_'+metric] for res in all_results]
            plt.scatter(x, y, label=metric, alpha=0.6)
        
        plt.xlabel(hp)
        plt.ylabel('Metric Value')
        plt.title(f'Impact of {hp} on Metrics')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./out/hyperparam_{hp}_comparison.png")
        plt.close()

# ====================== FEDERATED LEARNING CORE ======================
def train_local_model(global_model, num_epochs, train_dataloader, test_dataloader, learning_rate):

    print("############### LOCAL MODEL LOGS ############")
    local_model = copy.deepcopy(global_model)
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Use a learning rate scheduler to prevent getting stuck
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
      optimizer, max_lr=0.001, 
      steps_per_epoch=len(train_dataloader), epochs=num_epochs
    )
    
    local_model.train()
    for _ in range(num_epochs):
        for images, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = local_model(images)
            #target_onehot = F.one_hot(labels, num_classes=4).float()
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
            optimizer.step()
            


        val_loss, val_acc, _, _, _ = test_global_model(local_model, test_dataloader)
        scheduler.step(val_loss)

    avg_loss, accuracy, precision, recall, f1=test_global_model(local_model,test_dataloader)
    print("After_Training Local Model:")
    print("avg_loss: ",avg_loss)
    print("Accuracy: ",accuracy)
    print("precision: ",precision)
    print("recall: ",recall)
    
    return local_model.state_dict()

def federated_learning_algo(model, train_dataloaders, test_dataloader, num_rounds, num_epochs, learning_rate, hyperparams):
    global_model = copy.deepcopy(model)
    total_clients = len(train_dataloaders)
    total_batches_across_dataloaders=sum([len(train_dataloader) for train_dataloader in train_dataloaders])
    print("Total Number of batches across dataloaders: ",total_batches_across_dataloaders)
    metrics_history = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    print(f"\n=== Starting Training with: Test Ratio={hyperparams['test_ratio']}, Batch Size={hyperparams['batch_size']}, "
          f"Rounds={num_rounds}, Epochs={num_epochs}, LR={learning_rate} ===")

    client_optimizers = [
        torch.optim.Adam(global_model.parameters(), lr=learning_rate)
        for _ in range(len(train_dataloaders))
    ]
    
    for round in range(1, num_rounds+1):
        print(f"\nRound {round}/{num_rounds}:")
        
        # Local Training
        client_weights = []
        for client_id, (dataloader, optimizer) in enumerate(zip(train_dataloaders, client_optimizers), 1):
            print(f"  Client {client_id} training...", end='\r')
            weights = train_local_model(global_model, num_epochs, dataloader, test_dataloader, learning_rate)
            client_weights.append(weights)
        
        # Aggregation (Federated Averaging)
        global_weights = {}
        for key in client_weights[0].keys():
            for client, train_dataloader in zip(client_weights, train_dataloaders):
                if (key not in global_weights.keys()):
                    global_weights[key] = (len(train_dataloader)/total_batches_across_dataloaders)*client[key]
                    # print("ratio: ",(len(train_dataloader)/total_batches_across_dataloaders))
                else:
                    global_weights[key] += (len(train_dataloader)/total_batches_across_dataloaders)*client[key]                
                    # print("ratio: ",(len(train_dataloader)/total_batches_across_dataloaders))
                    
        global_model.load_state_dict(global_weights)
        
        # Testing
        test_loss, acc, prec, rec, f1 = test_global_model(global_model, test_dataloader)
        metrics_history['loss'].append(test_loss)
        metrics_history['accuracy'].append(acc)
        metrics_history['precision'].append(prec)
        metrics_history['recall'].append(rec)
        metrics_history['f1'].append(f1)

        print("############## GLOBAL MODEL LOGS ##############")
        print(f"  Round {round} Metrics - Loss: {test_loss:.4f} | Accuracy: {acc:.4f} | "
              f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}       ")
    
    # Save learning curves for this configuration
    plot_learning_curves(metrics_history, hyperparams)

    torch.save(global_model,"./out/"+str(hyperparams["test_ratio"])+"_"+str(hyperparams["batch_size"])+"_"+str(hyperparams["num_rounds"])+"_"+str(hyperparams["num_epochs"])+"_"+str(hyperparams["learning_rate"])+".pth")
    torch.save(global_model.state_dict(),"./out/"+str(hyperparams["test_ratio"])+"_"+str(hyperparams["batch_size"])+"_"+str(hyperparams["num_rounds"])+"_"+str(hyperparams["num_epochs"])+"_"+str(hyperparams["learning_rate"])+"_state_dict.pth")
    
    # Return final metrics for hyperparameter comparison
    return {
        'hyperparams': hyperparams,
        'final_accuracy': metrics_history['accuracy'][-1],
        'final_precision': metrics_history['precision'][-1],
        'final_recall': metrics_history['recall'][-1],
        'final_f1': metrics_history['f1'][-1],
        'final_loss': metrics_history['loss'][-1]
    }

# ====================== MAIN EXECUTION ======================

    
# Assume model and datasets are predefined
model = sample_CNN_model()  # Your model definition

assert all(p.requires_grad for p in model.parameters()), "Some weights have requires_grad=False"

# For training data (with augmentations)
train_transforms = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# For validation/test data (no augmentations)
test_transforms = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])

center_datasets = [custom_dataset(folder_path="Center_1",transform=train_transforms),custom_dataset(folder_path="Center_2",transform=train_transforms), custom_dataset(folder_path="Center_3",transform=train_transforms), custom_dataset(folder_path="Center_4",transform=train_transforms)]  # Your central datasets

# Hyperparameter Search Space
test_ratios = [0.2,0.3,0.4]
batch_sizes = [100]
num_rounds_list = [8]
num_epochs_list = [3]
learning_rates = [0.001] #,0.00007,0.0001,0.0007

all_results = []

for test_ratio in test_ratios:
    
    # Dataset preparation
    to_put_image_paths=[]
    to_put_image_labels=[]
    for center_dataset in center_datasets:
        to_append_image_paths,to_append_image_labels=center_dataset.take_out_items(test_ratio)
        to_put_image_paths.extend(to_append_image_paths)
        to_put_image_labels.extend(to_append_image_labels)
    test_dataset=custom_dataset(transform=test_transforms)#Here the folder path is None so that it makes an empty dataset
    test_dataset.add_items(to_put_image_paths,to_put_image_labels)
    
    for batch_size in batch_sizes:
        train_dataloaders = [DataLoader(center_dataset, batch_size=batch_size,shuffle=True) for center_dataset in center_datasets]
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
        
        for num_rounds in num_rounds_list:
            for num_epochs in num_epochs_list:
                for lr in learning_rates:
                    hyperparams = {
                        'test_ratio': test_ratio,
                        'batch_size': batch_size,
                        'num_rounds': num_rounds,
                        'num_epochs': num_epochs,
                        'learning_rate': lr
                    }
                    
                    # Run federated learning
                    result = federated_learning_algo(
                        model, train_dataloaders, test_dataloader, 
                        num_rounds, num_epochs, lr, hyperparams
                    )
                    all_results.append(result)
                    

# Generate hyperparameter comparison plots
plot_hyperparameter_comparisons(all_results)





