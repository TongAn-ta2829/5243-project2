# Project 2 – Interactive Data Processing Application

This project implements an interactive web application for data loading and preprocessing using **Python Shiny**.  
The application allows users to upload datasets, perform data cleaning operations, and inspect the results through an interactive web interface.

The application is designed as part of **Project 2** and focuses on building an interactive data processing workflow.

---

# Application Features

The application is organized into several modules corresponding to typical stages of a data analysis pipeline.

## 1. Loading Datasets

Users can load datasets in multiple ways:

- Upload their own datasets
- Select built-in example datasets

Supported file formats include:

- CSV
- Excel (.xlsx)
- JSON
- RDS

After loading a dataset, the application displays:

- Dataset overview
- Column summary
- Data preview

These tools help users quickly understand the structure of the dataset.

---

## 2. Data Cleaning & Preprocessing

The preprocessing module allows users to apply several common data cleaning techniques interactively.

Supported preprocessing operations include:

- Removing duplicate rows
- Handling missing values
- Dropping rows with missing values
- Filling missing values using mean, median or KNN
- Scaling numerical variables (StandardScaler / MinMaxScaler)

The cleaned dataset is updated automatically so that users can immediately observe the effect of the preprocessing steps.

---

## 3. Feature Engineering

The feature engineering module allows users to create new variables derived from existing features.

Possible transformations include:

- Mathematical transformations
- Interaction features
- Derived variables

This module helps enhance the analytical value of the dataset.

---

## 4. Exploratory Data Analysis (EDA)

The EDA module provides interactive tools for exploring the dataset.

Users can generate visualizations such as:

- Histograms
- Scatter plots
- Boxplots

These visualizations allow users to explore distributions, relationships, and patterns in the data.

---


