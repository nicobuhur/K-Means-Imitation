# Custom Implementation of K-Means Clustering with PCA

## Overview

This project implements a K-Means clustering algorithm from scratch, without using any built-in K-Means functions or libraries. The goal is to manually replicate the clustering process, allowing for a deeper understanding of the algorithm. Additionally, Principal Component Analysis (PCA) is employed to reduce the dimensionality of the data, making it easier to visualize the clusters in a 2D space.

## Introduction

K-Means clustering is a popular unsupervised learning algorithm used to group data points into a specified number of clusters (k). This project demonstrates a manual implementation of the K-Means algorithm, providing a hands-on approach to understanding its inner workings. By avoiding the use of built-in K-Means functions, this project highlights the core mechanics of the algorithm, from initializing centroids to iteratively updating them based on the assigned data points.

## Project Goals

- **Manual Implementation**: Replicate the K-Means clustering algorithm without using any pre-built functions or libraries specifically for K-Means.
- **Dimensionality Reduction**: Use PCA to reduce the dataset to two dimensions for easy visualization.
- **Visualization**: Plot the clustered data to assess the effectiveness of the manually implemented algorithm.

## Features

- **K-Means Clustering**: Fully implemented from scratch, including:
  - Initialization of centroids.
  - Assignment of data points to the nearest centroid.
  - Updating centroids based on the mean position of assigned points.
  - Iterative refinement until convergence.
- **PCA Integration**: Reduce high-dimensional data to 2D for visualization purposes.
- **Data Visualization**: Generate 2D plots to visualize the clusters and the convergence process.

## Installation

Ensure you have Python installed, along with the following dependencies:

```bash
pip install numpy pandas matplotlib
