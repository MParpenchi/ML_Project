# ML_Project
# ⚽ Data-Driven Analysis of European Football
### From Shot Patterns to Goal Prediction

## 📌 Overview
This project analyzes European football event data to predict the probability of a shot resulting in a goal
and to identify team tactical profiles using machine learning techniques.

The study applies supervised and unsupervised learning models on detailed match event data
from five major European leagues.

## 📂 Dataset
The dataset is derived from three main sources:
- events.csv (event-level match data)
- ginf.csv (match metadata)
- dictionary.txt (categorical decoding)

Main characteristics:
- Total shots: 229,135
- Leagues: Serie A, La Liga, Ligue 1, Bundesliga, Premier League
- Features: 19 variables
- Target: is_goal (1 = Goal, 0 = No Goal)

## ⚙️ Preprocessing
Data preprocessing was performed using Python and KNIME:
- Categorical decoding using dictionary file
- Data integration with match metadata
- Filtering only shot events (event_type = 1)
- Handling missing values:
  - Assist Method → "Solo Run"
  - Player Names → "Unknown"
- Feature selection and cleaning

## 🤖 Models

### Supervised Learning (Goal Prediction)
Used to predict shot success probability:
- Logistic Regression
- Decision Tree

Training setup:
- 70% Training / 30% Testing
- Stratified Sampling

### Unsupervised Learning (Team Profiling)
Used to identify tactical team profiles:
- k-Means Clustering (k = 3)
- Min-Max Normalization
- Features: Shot Volume & Average Goals

Clusters:
- Elite Teams
- Mid-table Teams
- Defensive/Struggling Teams

### Association Analysis
Pattern mining using:
- Apriori Algorithm
- Lift and Confidence metrics

Example rule:
{Cross → Header → Goal}

## 📈 Evaluation

### Classification Performance
Main evaluation metrics:
- Accuracy
- ROC Curve
- AUC

Results:

| Model              | Accuracy | AUC   |
|--------------------|----------|-------|
| Decision Tree      | 93.1%    | 0.941 |
| Logistic Regression| 94.3%    | 0.967 |

Best Model: Logistic Regression

### Clustering Evaluation
k-Means clustering successfully identified meaningful team tactical profiles
based on shot efficiency and volume.

## 🛠️ Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- KNIME Analytics Platform
- Matplotlib
- Jupyter Notebook

## ▶️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/your-repo-name.git
