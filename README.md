# Customer Churn Prediction using Machine Learning

![Customer Churn](images/customer_churn.png)

## 📖 Overview

This project predicts customer churn (whether a customer will leave or stay) for an e-commerce platform using machine learning techniques. Early identification of potential churners allows businesses to take proactive measures to retain customers.

We use customer data to analyze patterns and build predictive models, including:

* Logistic Regression
* Support Vector Machines (SVM)
* Decision Tree
* Random Forest
* XGBoost

---

## 📊 Dataset

The dataset contains **5630 customers** with **20 columns** describing their behavior and demographics.
Key column: `Churn` – indicates whether a customer has left (1) or stayed (0).

### Sample data:

| CustomerID | Gender | MaritalStatus | Tenure | OrderCount | CashbackAmount | Complain | Churn |
| ---------- | ------ | ------------- | ------ | ---------- | -------------- | -------- | ----- |
| 1          | Male   | Married       | 12     | 5          | 10             | No       | 0     |
| 2          | Female | Single        | 3      | 1          | 0              | Yes      | 1     |

### Churn distribution:

![Churn Pie Chart](images/churn_pie.png)

> 4682 customers stayed, 948 customers left (~17% churn rate)

---

## 🔎 Exploratory Data Analysis (EDA)

* **Numerical features:** Tenure, OrderCount, CashbackAmount
* **Categorical features:** Gender, MaritalStatus, Complain

### Examples:

#### Tenure Distribution

![Tenure Histogram](images/tenure_hist.png)

#### Gender vs Churn

![Gender vs Churn](images/gender_churn.png)

---

## 🔧 Data Preprocessing

1. Handle missing values (dropped rows with NaN)
2. Encode categorical variables using `pd.get_dummies()`
3. Standardize features using `StandardScaler()`
4. Split data into training and test sets (`train_test_split`)

---

## 🤖 Machine Learning Models

### Logistic Regression

* Accuracy: 0.90
* ROC Curve:
  ![ROC LR](images/roc_lr.png)

### Support Vector Machine

* Accuracy: 0.88
* ROC Curve:
  ![ROC SVM](images/roc_svm.png)

### Decision Tree

* Accuracy: 0.85
* Feature importance visualized:
  ![Decision Tree](images/decision_tree.png)

### Random Forest

* Accuracy: 0.91
* Confusion Matrix:
  ![Confusion Matrix RF](images/confusion_matrix_rf.png)

### XGBoost

* Accuracy: 0.92
* Confusion Matrix:
  ![Confusion Matrix XGB](images/confusion_matrix_xgb.png)

---

## 📚 Insights

* Customers with **short tenure** and **recent complaints** are more likely to churn.
* **Cashback usage** and **order frequency** are important features for predicting churn.
* Ensemble models (Random Forest, XGBoost) provide the highest accuracy.

---

## ⚡ Installation & Usage

```bash
git clone https://github.com/YourUsername/Customer-Churn-Prediction-ML.git
cd Customer-Churn-Prediction-ML
pip install -r requirements.txt
jupyter notebook 05-ML-14-Customer-churn.ipynb
```

---

## 📂 Repository Structure

```
Customer-Churn-Prediction-ML/
│
├── 05-ML-14-Customer-churn.ipynb
├── data/
│   └── E-Commerce-Dataset.xlsx
├── images/
│   ├── customer_churn.png
│   ├── churn_pie.png
│   ├── tenure_hist.png
│   ├── gender_churn.png
│   ├── roc_lr.png
│   ├── roc_svm.png
│   ├── decision_tree.png
│   ├── confusion_matrix_rf.png
│   └── confusion_matrix_xgb.png
├── README.md
└── requirements.txt
```

---

## 🔗 References

* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [XGBoost Documentation](https://xgboost.readthedocs.io/)
* E-Commerce Dataset: [GitHub link](https://github.com/anvarnarz/praktikum_datasets)

---

## 👤 Author

**Behruz Maxmudov**

* Email: [behruzmaxmudov263@gmail.com](mailto:behruzmaxmudov263@gmail.com)
* GitHub: [https://github.com/BehruzMaxmudov1203](https://github.com/BehruzMaxmudov1203)

---

## 🔧 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
openpyxl
jupyter
```
