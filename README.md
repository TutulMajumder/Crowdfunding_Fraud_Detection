# Crowdfunding Fraud Detection

This repository contains a complete workflow for detecting potentially fraudulent crowdfunding campaigns using machine learning. The project is based on the [Kickstarter Projects dataset](https://www.kaggle.com/datasets/ulrikthygepedersen/kickstarter-projects) from Kaggle.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Data Preparation & Feature Engineering](#data-preparation--feature-engineering)
- [Fraud Labeling & Anomaly Detection](#fraud-labeling--anomaly-detection)
- [Model Training & Evaluation](#model-training--evaluation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## Overview

The goal of this project is to identify suspicious or potentially fraudulent Kickstarter campaigns using a combination of unsupervised anomaly detection and supervised classification models. The workflow includes data cleaning, feature engineering, anomaly scoring, fraud labeling, and extensive model evaluation.

## Dataset

- **Source:** [Kickstarter Projects on Kaggle](https://www.kaggle.com/datasets/ulrikthygepedersen/kickstarter-projects)
- **Files Used:**  
  - `data/kickstarter_projects.csv` (raw data)
  - `data/combined_all_featuers_model_df.csv` (feature-enriched)
  - `data/reduced_df.csv` (final modeling set)

## Project Structure

```
├── data/
│   ├── kickstarter_projects.csv
│   ├── combined_all_featuers_model_df.csv
│   └── reduced_df.csv
├── notebooks/
│   ├── data_prep.ipynb
│   ├── model_training_and_evaluation_(1).ipynb
│   └── catboost_info/
├── requirements.txt
└── .gitignore
```

- **[notebooks/data_prep.ipynb](notebooks/data_prep.ipynb):** Data cleaning, feature engineering, anomaly detection, and fraud labeling.
- **[notebooks/model_training_and_evaluation_(1).ipynb](notebooks/model_training_and_evaluation_(1).ipynb):** Model training, evaluation, and feature importance analysis.

## Data Preparation & Feature Engineering

- **Cleaning:** Handles missing values, removes duplicates, and ensures type consistency.
- **Feature Engineering:**  
  - **Monetary:** Goal, Pledged, Backers, pledge/goal ratio, log(goal), avg pledge per backer.
  - **Temporal:** Duration, launch month, day of week, year, quarter.
  - **Categorical:** Encoded category, subcategory, country, state.
  - **Textual:** Name length (chars/words), uppercase ratio, punctuation count.
  - **Frequency:** Category/subcategory/country frequency.
  - **Other:** Binary flags for zero pledges, has backers, goal ambition buckets.

## Fraud Labeling & Anomaly Detection

- **Unsupervised Anomaly Detection:**  
  - Uses Isolation Forest to assign an anomaly score to each campaign.
  - Top-scoring campaigns are flagged as "Fraud Suspected".
- **Labeling:**  
  - Binary label (`is_fraud`) and descriptive label (`fraud_label`: "Normal" or "Fraud Suspected") are added for supervised learning.

## Model Training & Evaluation

- **Supervised Models:**  
  - Decision Tree
  - Logistic Regression
  - Random Forest
  - XGBoost
  - CatBoost
  - Gradient Boost
  - Gaussian Naive Bayes
  - K-Nearest Neighbors
  - Stacking Classifier (ensemble)
  - Voting Classifier 


- **Evaluation Metrics:**  
  - Accuracy, Precision, Recall, F1-score, ROC-AUC
  - Cross-validation and feature importance analysis

## Requirements

Install dependencies with:

```sh
pip install -r requirements.txt
```

## Usage

1. **Clone the Repository**
```bash
git clone https://github.com/TutulMajumder/Crowdfunding_Fraud_Detection.git
cd Crowdfunding_Fraud_Detection
```

2. **Set Up Environment**
```bash
python -m venv .venv
# Activate environment (choose your OS)
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac
```

```bash
pip install -r requirements.txt
```

3. **Download Dataset**

Download the [Kickstarter dataset](https://www.kaggle.com/datasets/ulrikthygepedersen/kickstarter-projects)  
and place it inside the `data/` directory (the folder is ignored in git, so you need to add it manually).

4. **Run Data Preparation**

Open and execute all cells in:

```
notebooks/data_prep.ipynb
```

This step preprocesses the dataset and generates features/labels.

5. **Train and Evaluate Models**

Open and run:

```
notebooks/model_training_and_evaluation_(1).ipynb
```

This trains multiple machine learning models and evaluates their performance.

## Results

- **Feature Importance:** Top features include `has_backers`, `pledge_goal_ratio`, `name_upper_ratio`, `avg_pledge_per_backer`, and `log_goal`.
- **Model Performance:** CatBoost and XGBoost models achieve the highest accuracy and recall for fraud detection, with detailed metrics and visualizations available in the notebooks. Other models perform less effectively on this dataset.
- **EDA:** The notebooks include extensive exploratory data analysis, including distributions, boxplots, KDE plots, and correlation heatmaps.

## License

This project is for educational and research purposes only.  
Dataset © Kaggle/Ulrik Thyge Pedersen.

---