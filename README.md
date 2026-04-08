# 📡 Telecom Churn Prediction Using Big Data & Deep Learning

This repository presents an **end-to-end predictive analytics pipeline** for customer churn prediction in the telecom sector.
The solution integrates **Big Data preprocessing with PySpark** and **Deep Learning (Keras/TensorFlow)** to deliver scalable, accurate churn prediction.

The project combines:

* Practical implementation (see `Big_Data_processing.ipynb`)
* In-depth research, literature review, and critical analysis (see research document)

---

## 🚀 Project Overview

**Goal:** Predict customer churn in the telecom industry by combining distributed data processing and deep learning techniques.

**Problem Statement:**
Customer churn is a major cost driver for telecom providers. Accurate churn prediction enables **targeted retention strategies**, reducing revenue loss.

**Approach:**

* Data preprocessing and feature engineering with **PySpark**
* Predictive modeling using **Deep Neural Networks (Keras)**
* Handling class imbalance with **SMOTE** and **class weighting**
* Generating **business insights** from churn-related patterns

---

## 📊 Dataset

* **Source:** [Iranian Churn Dataset, UCI ML Repository](https://archive.ics.uci.edu/)
* **Size:** 3,150 customer records
* **Features:** 13 attributes (calls, complaints, demographics, usage behavior, etc.)
* **Target Variable:** Binary churn label → `0 = Retained`, `1 = Churned`

---

## 🛠️ Technologies Used

* **Big Data Processing:** Apache Spark (PySpark)
* **Deep Learning:** TensorFlow / Keras
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Evaluation Metrics:** Scikit-learn (Precision, Recall, F1, ROC-AUC)

---

## 🔄 Workflow

1. **Library Import & Spark Initialization**
2. **Data Loading & Inspection** (as Spark DataFrame)
3. **Data Preprocessing**

   * Categorical encoding, numerical standardization, outlier handling
   * Feature vector assembly & train/test split
4. **Conversion** to Pandas/NumPy for model training
5. **Neural Network Modeling**

   * Multi-layer DNN (ReLU + Dropout, Sigmoid output)
   * Early stopping for optimal validation loss
6. **Training with Imbalance Handling**

   * SMOTE oversampling + class weights
7. **Model Evaluation**

   * Accuracy, Precision, Recall, F1, ROC-AUC
8. **Exploratory Data Analysis (EDA)**

   * Churn distribution, plan impact, subscription length, complaints, customer value

---

## ⚙️ Setup & Usage

### Requirements

* Python 3.x
* Apache Spark & PySpark
* TensorFlow
* Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

### Installation

```bash
pip install pyspark tensorflow pandas numpy matplotlib seaborn scikit-learn
```

### Running the Project

1. Open `Big_Data_processing.ipynb` in **Jupyter Notebook** or **Google Colab**
2. Follow the steps sequentially (update dataset path if required)
3. Run all cells to execute the full pipeline

---

## 📈 Key Results

* **Validation Accuracy:** ~82–89%
* **Precision/Recall:** Good for majority class; lower for churners (mitigated via SMOTE/class weighting)
* **Feature Insights:**

  * Churn strongly linked with **complaints** and **lower customer value**
  * Certain tariff plans show higher/lower churn rates
* **Limitations:**

  * Neural networks act as a **black box** (low interpretability)
  * Class imbalance still affects minority predictions
  * Results based on one dataset; computationally expensive

---

## 📚 Research Insights

* Comprehensive review of **Big Data & churn prediction literature**
* Comparison of **deep neural networks vs. ensemble models**
* Highlighted challenges:

  * Need for **explainable AI (XAI)**
  * Real-time churn prediction using streaming (Kafka, Spark Streaming)
  * Cross-industry generalization

---

## 🔮 Limitations & Future Work

* **Explainability:** Integrate SHAP/LIME for interpretability
* **Real-Time Prediction:** Extend to streaming frameworks
* **Generalizability:** Apply pipeline to diverse datasets
* **Model Benchmarking:** Compare with Random Forests, XGBoost, hybrid approaches

---

## 📖 References

* Core references are provided in the research document, covering **Big Data, Deep Learning, and Customer Churn Prediction** methodologies.
