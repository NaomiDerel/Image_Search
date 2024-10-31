"""
This script implements a complete image search pipeline, using saved CLIP and Siamese network models. It demonstrates all the functionalities of our work, excluding training.

Classes:
- SiameseNetworkEmbedder: A neural network model for generating image embeddings using a Siamese network architecture.
Functions:
- load_models(best_model_path): Loads and initializes the necessary models and transformations for the image search pipeline.
- load_and_preprocess_image(image_path, clip_transform): Loads an image from the specified file path and preprocesses it for use with the CLIP model.
- get_image_vector_clip(image_tensor, clip_model, clip_processor): Generates a feature vector for a given image tensor using the CLIP model.
- get_image_vector_siamese(image_tensor, clip_model, clip_processor, siamese_model): Generates an image embedding vector using a Siamese model.
- embed_image_vectors(image_names, clip_model, clip_processor, clip_transform, siamese_model, image_folder): Processes a list of image names, extracts feature vectors using two different models (CLIP and Siamese), and saves the resulting vectors to .npy files.
- build_faiss_hnsw_index(index_vectors, dim, nlinks): Builds a Faiss HNSW index.
- faiss_search(query_vectors, index, k): Uses a Faiss index to search for the k-nearest neighbors of query_vectors.
- plot_example(query_image, clip_images, siamese_images, k): Plots the query image along with the retrieval results from Siamese and CLIP models.
- main(): The main function that orchestrates the image search pipeline.
"""

import os
from PIL import Image
import torch
from torch import nn, optim
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
import time
from collections import defaultdict
from typing import Tuple, Dict, List
from matplotlib import pyplot as plt

# Set up device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


### ----- Embedding Models ----- ###


class SiameseNetworkEmbedder(nn.Module):
    """ As shown in active_learning_pipeline.ipynb,
     with the addition of a forward_one method to get the embedding of a single image. """

    def __init__(self):
        super(SiameseNetworkEmbedder, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()

    def forward_one(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc3(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

    def get_embedding(self, x):
        return self.forward_one(x)


def load_models(best_model_path="active_learning_models/net_round4.pth"):
    """
    Loads and initializes the necessary models and transformations for the image search pipeline.
    Args:
        best_model_path (str): Path to the pre-trained Siamese network model. Defaults to "active_learning_models/net_round4.pth".
    Returns:
        tuple: A tuple containing the following elements:
            - clip_model: The CLIP model loaded from the "openai/clip-vit-base-patch32" checkpoint.
            - clip_processor: The CLIP processor loaded from the "openai/clip-vit-base-patch32" checkpoint.
            - clip_transform: The image transformation pipeline for the CLIP model.
            - siamese_model: The Siamese network model loaded from the specified checkpoint.
    """

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", do_rescale=False)

    # Set up the image transformation pipeline
    clip_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    siamese_model = SiameseNetworkEmbedder()
    siamese_model.to(device)
    siamese_model.eval()
    siamese_model.load_state_dict(torch.load(
        best_model_path, map_location=torch.device(device)))

    return clip_model, clip_processor, clip_transform, siamese_model


def load_and_preprocess_image(image_path, clip_transform):
    """
    Loads an image from the specified file path and preprocesses it for use with the CLIP model.
    Args:
        image_path (str): The file path to the image to be loaded.
    Returns:
        PIL.Image.Image: The preprocessed image in RGB format.
    Raises:
        IOError: If there is an error loading the image.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        return clip_transform(image)
    except Exception as e:
        raise IOError(f"Error loading image '{image_path}': {e}")


def get_image_vector_clip(image_tensor, clip_model, clip_processor):
    """
    Generates a feature vector for a given image tensor using the CLIP model.
    Args:
        image_tensor (torch.Tensor): A tensor representing the image. 
                                     Expected shape is (C, H, W).
    Returns:
        numpy.ndarray: A flattened numpy array containing the image features.
    """
    image_tensor = image_tensor.unsqueeze(0)
    inputs = clip_processor(images=image_tensor, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features.cpu().numpy().flatten()


def get_image_vector_siamese(image_tensor, clip_model, clip_processor, siamese_model):
    """
    Generates an image embedding vector using a Siamese model, by processing the image tensor through clip and then the pre-trained siamese model.
    Args:
        image_tensor (torch.Tensor): A tensor representing the image to be processed.
    Returns:
        numpy.ndarray: A numpy array representing the image embedding vector.
    """
    image_tensor = image_tensor.unsqueeze(0)
    inputs = clip_processor(images=image_tensor, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    images_features = image_features.squeeze()

    image_emb = siamese_model.get_embedding(images_features.to(device))
    image_emb = image_emb.cpu().detach().numpy()
    return image_emb


def embed_image_vectors(image_names, clip_model, clip_processor, clip_transform, siamese_model, image_folder="datasets/house_styles/all_images"):
    """
    Processes a list of image names, extracts feature vectors using two different models (CLIP and Siamese),
    and saves the resulting vectors to .npy files.
    Args:
        image_names (list of str): List of image file names to process.
        output_file_path (str): Path (excluding extension) where the output .npy files will be saved.
    Returns:
        tuple: A tuple containing two numpy arrays:
            - clip_vectors (numpy.ndarray): Array of feature vectors extracted using the CLIP model.
            - siamese_vectors (numpy.ndarray): Array of feature vectors extracted using the Siamese model.
    """

    clip_vectors_list = []
    siamese_vectors_list = []

    for image_name in tqdm(image_names):
        image_path = os.path.join(image_folder, image_name)

        # process the image
        image_tensor = load_and_preprocess_image(image_path, clip_transform)
        image_vector_clip = get_image_vector_clip(
            image_tensor, clip_model, clip_processor)
        image_vector_siamese = get_image_vector_siamese(
            image_tensor, clip_model, clip_processor, siamese_model)

        # save the vectors
        clip_vectors_list.append(image_vector_clip)
        siamese_vectors_list.append(image_vector_siamese)

    clip_vectors = np.stack(clip_vectors_list)
    siamese_vectors = np.stack(siamese_vectors_list)

    return clip_vectors, siamese_vectors


### ----- Faiss Index ----- ###


def build_faiss_hnsw_index(
        index_vectors: np.ndarray,
        dim: int,
        nlinks: int,
):
    """
    This function builds a Faiss HNSW index.
    Args:
        index_vectors: An array of shape (n_index, dim) containing the index vectors.
        dim: The dimensionality of the vectors. 
        nlinks: The number of links to use in the graph.
    Returns:
        A Faiss HNSW index.
    """
    index = faiss.IndexHNSWFlat(dim, nlinks, faiss.METRIC_L2)
    index.add(index_vectors)
    return index


def faiss_search(
        query_vectors: np.ndarray,
        index: faiss.Index,
        k: int,
):
    """
    This function uses a Faiss index to search for the k-nearest neighbors of query_vectors.
    Args:
        query_vectors: An array of shape (n_queries, dim) containing the query vectors. 
        index: A Faiss index.
        k: The number of nearest neighbors to retrieve.
    Returns:
        An array of shape (, ) containing the indices of the k-nearest neighbors for each query vector.
    """
    if not isinstance(index, faiss.Index):
        raise ValueError("The index must be a Faiss index.")
    distances, indices = index.search(query_vectors, k)
    return indices


### ----- Plotting Results ----- ###


def plot_example(query_image, clip_images, siamese_images, k):
    """
    Plots the query image along with the retrieval results from Siamese and CLIP models.
    Parameters:
    query_image (ndarray): The image used as the query.
    clip_images (list of ndarray): List of images retrieved by the CLIP model.
    siamese_images (list of ndarray): List of images retrieved by the Siamese model.
    k (int): The number of images to display.
    The function creates a 3x5 grid of subplots:
    - The first row displays the query image.
    - The second row displays the top k images retrieved by the Siamese model.
    - The third row displays the top k images retrieved by the CLIP model.
    Each image is displayed with a title indicating its position in the retrieval results.
    """

    # Create a figure to display the images
    fig, axs = plt.subplots(3, k, figsize=(2*k, 6))

    # Show the query image
    axs[0, 0].imshow(query_image)
    axs[0, 0].set_title("Query Image")
    axs[0, 0].axis('off')

    # remove the other columns
    for i in range(1, k):
        axs[0, i].axis('off')

    # Load and display Siamese index images
    for i in range(k):
        if i == 0:
            axs[1, i].set_ylabel("Siamese Model")
        axs[1, i].imshow(siamese_images[i])
        axs[1, i].set_title(f"Siamese Result {i + 1}")
        axs[1, i].axis('off')

    # Load and display CLIP index images
    for i in range(k):
        if i == 0:
            axs[2, i].set_ylabel("CLIP Model")
        axs[2, i].imshow(clip_images[i])
        axs[2, i].set_title(f"CLIP Result {i + 1}")
        axs[2, i].axis('off')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


### ----- Main Pipeline ----- ###


def main():
    # Load models
    best_model_path = "active_learning_models/net_round4.pth"
    clip_model, clip_processor, clip_transform, siamese_model = load_models(
        best_model_path)

    # Load the image names
    image_folder = "datasets/house_styles/all_images"
    image_names = os.listdir(image_folder)

    # Save the image vectors
    CREATE_VECTORS = False

    if CREATE_VECTORS:
        clip_vectors, siamese_vectors = embed_image_vectors(
            image_names, clip_model, clip_processor, clip_transform, siamese_model, image_folder=image_folder)
    else:
        clip_vectors = np.load(
            "vector_dbs/full_index/vectors_index_clip.npy")
        siamese_vectors = np.load(
            "vector_dbs/full_index/vectors_index_siamese.npy")

    # Create Faiss index for models
    dim_clip = clip_vectors.shape[1]
    dim_siamese = siamese_vectors.shape[1]
    nlinks = 16

    index_clip = build_faiss_hnsw_index(clip_vectors, dim_clip, nlinks)
    index_siamese = build_faiss_hnsw_index(siamese_vectors, dim_siamese, nlinks)

    #### Select a query image of interest:
    # Examples: 553_f514f73f.jpg, 441_c1205bad.jpg, 001_d2c7428a.jpg
    query_image_name = "267_4433e7a0.jpg"  
    query_image_path = os.path.join(image_folder, query_image_name)

    # Load and preprocess the query image
    query_image_tensor = load_and_preprocess_image(query_image_path, clip_transform)

    # Get the query image vectors
    query_image_vector_clip = get_image_vector_clip(query_image_tensor, clip_model, clip_processor)
    query_image_vector_siamese = get_image_vector_siamese(query_image_tensor, clip_model, clip_processor, siamese_model)

    # Perform Faiss search
    k = 5  # no more than 15 for clear plots
    clip_indices = faiss_search(
        query_image_vector_clip.reshape(1, -1), index_clip, k+1)[0][1:] # exclude the query image itself
    siamese_indices = faiss_search(
        query_image_vector_siamese.reshape(1, -1), index_siamese, k+1)[0][1:] # exclude the query image itself
    
    print(siamese_indices)

    # Save retrieved image names
    clip_images = [Image.open(os.path.join(
        image_folder, image_names[i])) for i in clip_indices]
    siamese_images = [Image.open(os.path.join(
        image_folder, image_names[i])) for i in siamese_indices]
    
    print("Query Image:", query_image_name)
    print("CLIP Results:", [image_names[i] for i in clip_indices])
    print("Siamese Results:", [image_names[i] for i in siamese_indices])

    # Display the results
    query_image = Image.open(query_image_path)
    plot_example(query_image, clip_images, siamese_images, k)


if __name__ == "__main__":
    main()
