# Cricket Player Role Prediction

A CSE422 (Artificial Intelligence Lab) project. We test whether a cricket player's role (**Batsman, Bowler, Allrounder or Wicketkeeper**) can be predicted from biographical and playing-style information alone, without any match statistics.

**Short answer:** not very well. The available features carry limited signal, and that negative result turned out to be the most interesting finding of the project (see [Results](#results)).

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Why Is Performance Low?](#why-is-performance-low)
- [Figures](#figures)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Built With](#built-with)
- [Contributors](#contributors)

---

## Problem Statement

In practice, a player's role is decided by coaches and scouts based on how the player actually performs. We wanted to check whether a machine learning model can guess the role using only:

- batting style
- bowling style
- continent
- gender
- age

## Dataset

- **Source:** provided by our course faculty (shared here with permission).
- **Main file:** `players_data_with_all_info.csv` with **17,385 rows and 16 columns**.
- **Secondary file:** `teams.csv` (78 rows). We inspected it but did not use it as a feature, because only about 15% of players matched a team by `country_id`, and the remaining useful information was already covered by `continent_name`. It is kept in the repository for reference.

**Target variable:** `position` (Batsman / Bowler / Allrounder / Wicketkeeper). The classes are heavily imbalanced: Bowler alone is about 33% of the data, while Wicketkeeper is under 10%.

**Features used**

| Feature | Description |
|---|---|
| `battingstyle` | Batting style of the player |
| `bowlingstyle` | Bowling style of the player |
| `continent_name` | Continent the player belongs to |
| `gender` | Gender of the player |
| `age` | Engineered by us from date of birth |
| `is_dob_estimated` | Flag we added for suspected placeholder birthdates (explained below) |

## Methodology

### 1. Exploratory Data Analysis
We checked class balance, built a correlation heatmap, and studied how batting style, bowling style, continent, age and gender relate to position.

### 2. Data Cleaning
- `dateofbirth` contained **25 completely invalid values** (such as `0000-00-00`).
- A less obvious problem: about **6,081 rows** had a birthdate of exactly `01-01`, mostly in the years 2000 to 2004. A random distribution would give roughly 0.27% of rows on that date, so we are fairly confident these are placeholder dates entered when the real date of birth was unknown.
- Instead of deleting these rows or trusting them blindly, we created the **`is_dob_estimated`** flag so models can account for the uncertainty.

### 3. Preprocessing
- One-hot encoding for categorical features
- Scaling of `age`
- Stratified 80/20 train-test split
- Median-age imputation for missing ages, **fitted on the training set only** to avoid data leakage

### 4. Models
We trained six models through the same pipeline so the comparison is fair:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Naive Bayes
5. Neural Network (small MLP)
6. Random Forest

**Evaluation:** Macro F1 is the main metric, because accuracy alone is misleading on imbalanced classes. We also report accuracy and ROC-AUC, and ran **5-fold cross-validation** to confirm the results are not a lucky split.

### 5. Bonus: K-Means Clustering
We ran K-Means on the same features to see whether the data naturally forms four clusters that match the real positions.

## Results

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| **Random Forest** | 0.3612 | **0.3563** | 0.6413 |
| Decision Tree | 0.3592 | 0.3516 | 0.6199 |
| Logistic Regression | 0.3566 | 0.3430 | 0.6458 |
| KNN | 0.3558 | 0.3302 | 0.6145 |
| Neural Network | 0.3831 | 0.3345 | 0.6467 |
| Naive Bayes | 0.1521 | 0.1303 | 0.6060 |

**Best model:** Random Forest (Macro F1 of about 0.356). Cross-validation confirmed it is stable, with a mean Macro F1 of **0.3512** across folds.

**K-Means:** silhouette scores stayed flat (about 0.30 to 0.34) regardless of the number of clusters, so there is no strong natural grouping that lines up with the four real positions.

## Why Is Performance Low?

We believe this is simply a hard problem with the given features, not a bug in the code:

- Batting style and bowling style give the models some signal.
- Continent, gender and age help very little.
- Because Bowler is about 33% of the data, even a model that always predicts "Bowler" would score close to these accuracy numbers.
- The flat K-Means silhouette scores support the same conclusion: these columns do not carry enough information to separate the four roles.

We chose to report this honestly rather than tune the setup to make the numbers look better.

## Figures

All saved plots are in `reports/figures/`, including:

- Class distribution
- Model comparison (Accuracy and Macro F1)
- Bowling style vs. position
- Confusion matrix for each model
- K-Means elbow and PCA plots

## Repository Structure

```
├── CSE422_Project_PlayerRole.ipynb   # EDA, cleaning, models, evaluation
├── CSE422_Project_Report.pdf         # Written report
├── players_data_with_all_info.csv    # Main dataset
├── teams.csv                         # Inspected, not used as a feature
├── reports/figures/                  # Saved plots
├── requirements.txt
└── README.md
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/ZARIFYAMIN/cricket-player-role-prediction.git
   cd cricket-player-role-prediction
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook CSE422_Project_PlayerRole.ipynb
   ```

The dataset CSVs are included, so no path changes are needed as long as `players_data_with_all_info.csv` and `teams.csv` stay in the same folder as the notebook.

> **Note:** the project was originally built in Google Colab and loaded data from Google Drive. If you see any leftover Drive paths in a cell, point them to the local CSV files instead.

## Built With

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Contributors

- **Kazi Zarif Yamin** ([@ZARIFYAMIN](https://github.com/ZARIFYAMIN))
- **Syeda Fatima Tasnia ** ([@tasniaanika100-commits](https://github.com/tasniaanika100-commits))
