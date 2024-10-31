# Image_Search

A project for Data Analysis and Visualization Lab course, implementing a retrieval pipeline for image search using ANN, and tests of the effect of fine-tuning the similarity metric with Active Learning methods.

## Goal

Abstract of the project

Pipeline image

## Installation

The required packages for the project can be imported by creating a new environment with the `environment.yml` file. The environment can be created with the following command:

```bash
conda env create -f environment.yml
```

Or run on Azure Machine with kernel `azureml_py38`.

## Datasets

Download the house styles dataset from [Kaggle](https://www.kaggle.com/datasets/kelvingothman/house-typestyle-detection) - to access the `all_images` folder containing the .jpg files.

Our contribution to the data is in the labeling of pairs of images by similarity, acquired both from `data_tagging.ipynb` and our manual work.

As described in `data_tagging.ipynb`, we sampled 150 images from each class, 450 overall. The sampled images are those in the `sampled_labels.csv` file, and all the pairs of images with similarity labels are in the `sampled_paired_labels_shuffled.csv` file. The similarity labels in this file could be null, automatically 0 due to different classes, or 1 through 3 for manually labeled pairs during active learning.

The expected data configuration is then:
```
|- datasets
    |- house_styles
        |- all_images
        labels.csv
        sampled_labels.csv
        sampled_paired_labels_shuffled.csv
```

A blind test set, independent of the model's selection, is also provided in `active_learning_labels` folder.

Additional tagged data can be found in `evaluation_paired_labels.csv`, used to evaluate the model's performance by the retrieved pairs.


## Models

Our approach to an image retrieval pipeline contains two main models: an embedding model and a retrieval model.

### Embedding Model - Siamese Network

The embedding model is a Siamese Network, fine tuned from CLIP embeddings using contrastive loss. The model is part of `active_learning_pipeline.ipynb` and is also shown separately in `siamese_network.py`.

#### Active Learning

The active learning process is implemented in `active_learning_pipeline.ipynb`. The process of active learning - trained model, data samples to tag, and results from each round - is logged in `active_learning_models`, `active_learning_labels`, and `active_learning_results` folders, respectively.

### Retrieval Model - FAISS

The retrieval model is based on FAISS, a library for efficient similarity search and clustering of dense vectors. Analysis and comparison of ANN methods are in `ann_evaluation.ipynb`, and saved vector databases for embedded indexes can be found in the `vector_dbs` folder.

## Evaluation

In addition to other results shown in the notebook, the folder `report_results` contains code and figures as shown in the report.