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
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()

    def forward_one(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


class ImprovedSiameseNetwork(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ImprovedSiameseNetwork, self).__init__()
        self.network = nn.Sequential(
            # First block
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second block
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Third block
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Final projection
            nn.Linear(128, 64)
        )
        
        # L2 normalization layer
        self.l2norm = lambda x: F.normalize(x, p=2, dim=1)
        
    def forward_one(self, x):
        x = self.network(x)
        x = self.l2norm(x)  # L2 normalize embeddings
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


# define the loss function
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    """

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
    # Load the pretrained CLIP model and processor from Hugging Face
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Set up the image transformation pipeline
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    augment_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        # 20% chance of random resized crop
        transforms.RandomApply([transforms.RandomResizedCrop(224)], p=0.3),
        # 20% chance of horizontal flip
        transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.3),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.3),  # 20% chance of color jitter
        transforms.ToTensor()  # Always apply ToTensor
    ])

    return model, processor, augment_transform, eval_transform


def load_data(clip_model, clip_processor, augment_transform, eval_transform, batch_size=8, train_ratio=0.8):

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


def train_siamese_network(model, train_loader, eval_loader, criterion, optimizer, num_epochs):
    Evaluate_Flag = False
    model.train()  # Set the model to training mode
    max_train_f1 = 0
    max_eval_f1 = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        tp, fp, tn, fn = 0, 0, 0, 0

        for i, (img1_path, img2_path, img1, img2, labels) in tqdm(enumerate(train_loader)):
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

        # Save best model:

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

            # Save the model with the best evaluation accuracy
            if eval_f1 >= max_eval_f1:
                max_eval_f1 = eval_f1
                best_model = model.state_dict()
                print("Best model updated")

    return model, best_model


def train_siamese_network_scheduler(model, train_loader, eval_loader, criterion, optimizer, num_epochs, patience=5):
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
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
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
    model.eval()  # Set the model to evaluation mode
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
    plt.show()


def main():
    batch_size = 16
    num_epochs = 150
    patience = 5
    lr = 0.01

    clip_model, clip_processor, augment_transform, eval_transform = load_clip_model()
    train_loader, eval_loader = load_data(
        clip_model, clip_processor, augment_transform, eval_transform, batch_size, train_ratio=0.8)
    print("Data loaded")

    siamese_net = ImprovedSiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(siamese_net.parameters(), lr)

    # Train the model
    print("Training the model")
    best_model, history = train_siamese_network_scheduler(
        siamese_net, train_loader, eval_loader, criterion, optimizer, num_epochs, patience)

    # Save the best model
    torch.save(best_model, 'active_learning_models/final_model.pth')
    torch.save(optimizer.state_dict(), 'active_learning_models/final_optimizer.pth')
    torch.save(history, 'active_learning_models/final_history.pth')
    print("Model saved")

    # Evaluate the model
    loaded_model = ImprovedSiameseNetwork().to(device)
    loaded_model.load_state_dict(torch.load('active_learning_models/final_model.pth'))

    test_siamese_network(loaded_model, train_loader, "Train")
    test_siamese_network(loaded_model, eval_loader, "Evaluation")

if __name__ == "__main__":
    main()