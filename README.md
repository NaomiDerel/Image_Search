# Image_Search

A project for Data Analysis and Visualization Lab course, implementing a retrieval pipeline for image search using ANN, and tests of the effect of fine-tuning the similarity metric with Active Learning methods.

## Datasets

Download the house styles dataset from [Kaggle](https://www.kaggle.com/datasets/kelvingothman/house-typestyle-detection).

After downloading we expect the following files in this repository to run the code:
```
|- datasets
    |- house_styles
        |- all_images
        labels.csv
```

Our contribution is in the labeling of pairs of images by similarity, acquired both from `data_tagging.ipynb` and our manual work.

This dataset can be accessed from...

The expected data configuration is then:
```
|- datasets
    |- house_styles
        |- all_images
        labels.csv
        sampled_labels.csv
        sampled_paired_labels.csv
```

