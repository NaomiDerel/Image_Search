"""
This module implements a Siamese Network for image similarity using the CLIP model for feature extraction.
Classes:
    ImageSimilarityDataset: Custom dataset class for loading image pairs and their similarity labels.
    SiameseNetwork: Defines the architecture of the Siamese Network as used in Active Learning.
    NormSiameseNetwork: Defines a normalized version of the Siamese Network with dropout for a retrospectively-trained model.
    ContrastiveLoss: Custom loss function for training the Siamese Network.
Functions:
    load_clip_model: Loads the pretrained CLIP model and processor, and sets up image transformation pipelines.
    load_data: Loads and preprocesses the dataset, splits it into training and evaluation sets, and creates DataLoaders.
    train_siamese_network: Trains the Siamese Network using the provided DataLoader, criterion, and optimizer.
    train_siamese_network_scheduler: Trains the Siamese Network with a learning rate scheduler and early stopping.
    test_siamese_network: Evaluates the trained Siamese Network on a test dataset and prints accuracy, F1 score, and confusion matrix.
    main: Main function to load data, train the model, and evaluate the model.
Usage:
    Run the script to train and evaluate a Siamese Network for image similarity.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import f1_score, confusion_matrix

if torch.cuda.is_available():
    device = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device('cpu')

print(f"Using device: {device}")


class ImageSimilarityDataset(Dataset):
    """ As shown in active_learning_pipeline.ipynb """

    def __init__(self, dataframe, model, processor, transform):
        self.data = dataframe
        self.model = model
        self.processor = processor
        self.transform = transform
        self.master_path = ''

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load images from the paths
        image1_path = self.master_path + self.data.iloc[idx, 0].strip("()")
        image2_path = self.master_path + self.data.iloc[idx, 1].strip("()")

        # Load images
        image1 = Image.open(image1_path).convert("RGB")
        image2 = Image.open(image2_path).convert("RGB")

        # Apply CLIP transforms and augmentations if provided (transforms should convert to tensor)
        if self.transform:
            # Apply transforms including ToTensor
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Get image features using CLIP
        images_features = []
        for img in [image1, image2]:
            # Add batch dimension for processing
            image_tensor = img.unsqueeze(0)
            inputs = self.processor(images=image_tensor, return_tensors="pt")
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            # Ensure it's a 512-dimensional tensor
            images_features.append(image_features.squeeze())

        # Get similarity score and label (1 for similar, 0 for dissimilar)
        similarity = self.data.iloc[idx, 2]
        label = 0 if similarity < 3 else 1

        return image1_path, image2_path, images_features[0], images_features[1], torch.tensor(label, dtype=torch.float32)


class SiameseNetwork(nn.Module):
    """ As shown in active_learning_pipeline.ipynb """

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        # self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()

    def forward_one(self, x):
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


class NormSiameseNetwork(nn.Module):
    """
    A Siamese Network with normalization layers for comparing pairs of inputs, attempted to retrospectively train the model post active learning. This model did not show significant improvement over the original Siamese Network.

    Args:
        dropout_rate (float): The dropout rate to be applied after each ReLU activation. Default is 0.1.
    Attributes:
        network (nn.Sequential): The sequential container of layers forming the network.
    Methods:
        forward_one(x):
            Passes a single input through the network.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The output tensor after passing through the network.
        forward(input1, input2):
            Passes two inputs through the network and returns their respective outputs.
            Args:
                input1 (torch.Tensor): The first input tensor.
                input2 (torch.Tensor): The second input tensor.
            Returns:
                tuple: A tuple containing the outputs for input1 and input2.
    """


    def __init__(self, dropout_rate=0.1):
        super(NormSiameseNetwork, self).__init__()
        self.network = nn.Sequential(
            # First block
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second block
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Third block
            nn.Linear(256, 128)
        )
        
    def forward_one(self, x):
        x = self.network(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


# define the loss function
class ContrastiveLoss(torch.nn.Module):
    """ As shown in active_learning_pipeline.ipynb """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        label = y  # binary
        euclidean_distance = nn.functional.pairwise_distance(x0, x1)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +  # similar
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))  # dissimilar
        return loss_contrastive


def load_clip_model():
    """ Based on code shown in active_learning_pipeline.ipynb """

    # Load the pretrained CLIP model and processor from Hugging Face
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", do_rescale=False)

    # Set up the image transformation pipeline
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    augment_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomApply([transforms.RandomResizedCrop(224)], p=0.3),
        transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.3),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)], p=0.3),
        transforms.ToTensor()  # Always apply ToTensor
    ])

    return model, processor, augment_transform, eval_transform


def load_data(clip_model, clip_processor, augment_transform, eval_transform, batch_size=8, train_ratio=0.8):
    """ Based on code shown in active_learning_pipeline.ipynb """

    total_rounds = 4
    base_path = 'active_learning_labels/'
    full_data_paths = pd.DataFrame()

    for i in range(0, total_rounds):
        path = base_path + 'round_' + str(i) + '.csv'
        data = pd.read_csv(path)
        full_data_paths = pd.concat([full_data_paths, data], ignore_index=True)
        print(f"Round {i} data loaded")

    data_paths = full_data_paths[['image1_path', 'image2_path', 'similarity']]
    print("Found", len(data_paths), "tagged image pairs")

    # split the data into training and testing
    train_data = data_paths.sample(frac=train_ratio, random_state=42)
    eval_data = data_paths.drop(train_data.index)

    # Create datasets
    train_dataset = ImageSimilarityDataset(
        train_data, clip_model, clip_processor, transform=augment_transform)
    eval_dataset = ImageSimilarityDataset(
        eval_data, clip_model, clip_processor, transform=eval_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    return train_loader, eval_loader


def train_siamese_network(model, train_loader, eval_loader, criterion, optimizer, num_epochs, try_num):
    """ Based on code shown in active_learning_pipeline.ipynb """

    Evaluate_Flag = True
    model.train()
    max_train_f1 = 0
    max_eval_f1 = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        tp, fp, tn, fn = 0, 0, 0, 0

        for img1_path, img2_path, img1, img2, labels in tqdm(train_loader):
            # Move tensors to the appropriate device
            img1, img2, labels = img1.to(device), img2.to(
                device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Accuracy
            dist = F.pairwise_distance(output1, output2)
            predicted = (dist < 1.0).float()
            correct += (predicted == labels).sum().item()

            # F1 score
            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

        f1 = 2 * tp / (2 * tp + fp + fn)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {correct / len(train_loader.dataset):.2f}, Train F1 Score: {f1:.2f}")

        with open(f'active_learning_results/log_try_{try_num}.txt', 'a') as f:
            f.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {correct / len(train_loader.dataset):.2f}, Train F1 Score: {f1:.2f}\n")

        if not Evaluate_Flag and f1 >= max_train_f1:
            max_train_f1 = f1
            best_model = model.state_dict()
            print("Best model updated")

        if Evaluate_Flag and (epoch + 1) % 5 == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for img1_path, img2_path, img1, img2, labels in eval_loader:
                    img1, img2, labels = img1.to(device), img2.to(
                        device), labels.to(device)
                    output1, output2 = model(img1, img2)
                    dist = F.pairwise_distance(output1, output2)

                    predicted = (dist < 1.0).float()
                    tp += ((predicted == 1) & (labels == 1)).sum().item()
                    fp += ((predicted == 1) & (labels == 0)).sum().item()
                    tn += ((predicted == 0) & (labels == 0)).sum().item()
                    fn += ((predicted == 0) & (labels == 1)).sum().item()

            eval_f1 = 2 * tp / (2 * tp + fp + fn)
            print(f"Evaluation F1: {eval_f1:.2f}")
            with open(f'active_learning_results/log_try_{try_num}.txt', 'a') as f:
                f.write(f"Evaluation F1: {eval_f1:.2f}\n")

            # Save the model with the best evaluation accuracy
            if eval_f1 >= max_eval_f1:
                max_eval_f1 = eval_f1
                best_model = model.state_dict()
                torch.save(best_model, 'active_learning_models/final_best_model.pth')
                print("Best model updated")

    return model, best_model


def train_siamese_network_scheduler(model, train_loader, eval_loader, criterion, optimizer, num_epochs, patience=10):
    """
    Trains a Siamese network with the addition of a learning rate scheduler and early stopping.
    Args:
        model (torch.nn.Module): The Siamese network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        eval_loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        criterion (callable): Loss function to be used.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        num_epochs (int): Number of epochs to train the model.
        patience (int, optional): Number of epochs with no improvement after which training will be stopped. Default is 5.
    Returns:
        tuple: A tuple containing:
            - best_model (dict): The state dictionary of the best model.
            - history (dict): A dictionary containing training and evaluation loss and F1 score history.
    """

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Early stopping setup
    best_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [], 'train_f1': [],
        'eval_loss': [], 'eval_f1': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []
        
        for img1_path, img2_path, img1, img2, labels in tqdm(train_loader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, labels)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Store predictions and losses
            train_losses.append(loss.item())
            dist = F.pairwise_distance(output1, output2)
            preds = (dist < 1.0).float()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Evaluation phase
        model.eval()
        eval_losses = []
        eval_preds = []
        eval_labels = []
        
        with torch.no_grad():
            for img1_path, img2_path, img1, img2, labels in eval_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, labels)
                
                eval_losses.append(loss.item())
                dist = F.pairwise_distance(output1, output2)
                preds = (dist < 1.0).float()
                eval_preds.extend(preds.cpu().numpy())
                eval_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_loss = np.mean(train_losses)
        train_f1 = f1_score(train_labels, train_preds)
        eval_loss = np.mean(eval_losses)
        eval_f1 = f1_score(eval_labels, eval_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['eval_loss'].append(eval_loss)
        history['eval_f1'].append(eval_f1)
        
        # Learning rate scheduling
        scheduler.step(eval_loss)
        
        # Early stopping check
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"Eval Loss: {eval_loss:.4f}, Eval F1: {eval_f1:.4f}")
    
    return best_model, history


def test_siamese_network(model, test_loader, loader_name):
    """ Based on code shown in active_learning_pipeline.ipynb """

    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for i, (img1_path, img2_path, img1, img2, labels) in enumerate(test_loader):
            # Move tensors to the appropriate device
            img1, img2, labels = img1.to(device), img2.to(
                device), labels.to(device)

            # Forward pass
            output1, output2 = model(img1, img2)

            # Calculate the euclidean distance between the outputs
            dist = F.pairwise_distance(output1, output2)

            # Get predictions
            predicted = (dist < 1.0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels)
            all_predictions.extend(predicted)

            # print(f"Dist: {dist}, Predicted: {predicted}, Actual: {labels}")

            # if i == 10:
            #     break

    # accuracy:
    accuracy = correct / total
    print(f"{loader_name} Accuracy: {accuracy:.4f}")

    # move to cpu:
    all_labels = [label.cpu().numpy() for label in all_labels]
    all_predictions = [prediction.cpu().numpy()
                       for prediction in all_predictions]

    # f1 score:
    f1 = f1_score(all_labels, all_predictions)
    print(f"{loader_name} F1 Score: {f1:.4f}")

    # confusion matrix:
    matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues')
    plt.title(f"{loader_name} Confusion Matrix")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    
    # Save the confusion matrix
    plt.savefig(f'active_learning_results/final_{loader_name}.png')


def main():
    batch_size = 16
    num_epochs = 100
    patience = 5
    lr = 0.01

    clip_model, clip_processor, augment_transform, eval_transform = load_clip_model()
    train_loader, eval_loader = load_data(
        clip_model, clip_processor, augment_transform, eval_transform, batch_size, train_ratio=0.8)
    print("Data loaded")

    siamese_net = NormSiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(siamese_net.parameters(), lr)

    # Train the model
    print("Training the model")
    # best_model, history = train_siamese_network_scheduler(siamese_net, train_loader, eval_loader, criterion, optimizer, num_epochs, patience)
    trained_model, best_model = train_siamese_network(siamese_net, train_loader, eval_loader, criterion, optimizer, num_epochs, try_num=1)

    # Save the best model
    torch.save(optimizer.state_dict(), 'active_learning_models/final_optimizer.pth')
    torch.save(trained_model, 'active_learning_models/final_trained_model.pth')
    print("Model saved")

    # Evaluate the model
    loaded_model = NormSiameseNetwork().to(device)
    loaded_model.load_state_dict(torch.load('active_learning_models/final_best_model.pth'))

    test_siamese_network(loaded_model, train_loader, "Train")
    test_siamese_network(loaded_model, eval_loader, "Evaluation")

if __name__ == "__main__":
    main()