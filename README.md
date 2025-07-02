# 🚢 Titanic Survival Prediction

A machine learning project to predict the survival of Titanic passengers using **Logistic Regression** and **Random Forest Classifier**.  
Optimized hyperparameters using **GridSearchCV** to improve prediction accuracy and model performance.

---

## 📌 Project Overview

The infamous Titanic disaster is one of the most classic classification problems in machine learning.  
This project uses supervised learning models to predict whether a passenger survived based on their characteristics.

---

## 📊 Dataset

The dataset is sourced from the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic).

### Features:
- Passenger Class (`Pclass`)
- Sex
- Age
- Siblings/Spouses Aboard (`SibSp`)
- Parents/Children Aboard (`Parch`)
- Fare
- Embarked Port

**Target:**  
- `Survived` → 0 = No, 1 = Yes

---

## 🛠️ Tools & Technologies

- Python 3.9+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
  - Logistic Regression
  - Random Forest Classifier
  - GridSearchCV

---

## 🚀 Workflow

1. Data Cleaning & Missing Value Handling
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Feature Scaling (for Logistic Regression)
5. Model Building:
   - Logistic Regression
   - Random Forest Classifier
6. Hyperparameter Tuning with GridSearchCV
7. Model Evaluation & Comparison

---

## 📈 Model Performance

| Model                     | Accuracy (Test Set) |
|:--------------------------|:------------------|
| Logistic Regression        | 78%                |
| Random Forest (optimized)  | 82%                |

---

## ⚙️ Hyperparameter Tuning with GridSearchCV

Applied GridSearchCV to Random Forest to find the best combination of:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

**Best Parameters Example:**
```python
{
  'max_depth': 6,
  'max_features': 'sqrt',
  'min_samples_leaf': 2,
  'min_samples_split': 5,
  'n_estimators': 200
}

## 📬 How to Run

### 1️. Clone the Repository
```bash
git clone https://github.com/Manik178/Titanic_Survival_Prediction.git
cd titanic-survival-prediction

### 2. Run the notebook
```bash
jupyter notebook

Then open Titanic_Survival_Prediction.ipynb from the Jupyter interface.

### 3. Run All Cells

Once the notebook opens:

From the Jupyter toolbar → click Kernel → Restart & Run All



