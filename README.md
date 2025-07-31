# 🦟 Malaria Incidence Prediction and Analysis

## 📌 Project Overview

This project explores and predicts malaria incidence rates using global health data. It is developed as part of a capstone project for the **Introduction to Big Data Analytics (INSY 8413)** course.

Using historical malaria metrics from national health units, we analyze trends and build a predictive model to estimate future incidence rates across various countries.

---

## 🎯 Objective

**Can we analyze and predict malaria trends over time using country-level data on incidence, mortality, and infection prevalence from a unified global health dataset?**

---

## 📁 Dataset

- **Source**: [OpenAfrica – Global Malaria National Unit Data](https://open.africa/dataset/malaria-national_unit-data/resource/9dbe7be3-f196-4b10-953b-84c36e05d99f)
- **File Used**: `openafrica-_-malaria-_-national_unit-data-raw-national_unit-data.csv.csv`

---

## 🧼 Step 1: Data Cleaning (Python 🐍)

- Removed irrelevant columns
- Handled missing values
- Formatted country codes and years
- Saved cleaned dataset as `malaria_cleaned.csv`

---

## 📊 Step 2: Exploratory Data Analysis (EDA)

- Identified key trends in malaria incidence over time
- Visualized country-level metrics
- Used heatmaps, bar charts, and line graphs

---

## 🤖 Step 3: Machine Learning – Prediction

- Used **Linear Regression** to predict malaria incidence rates by country
- Split data into training/test sets
- Evaluated model using MSE and R² score
- Generated `malaria_predictions.csv` containing forecasted results

---

## 📈 Step 4: Visualization in Power BI

- Created an interactive dashboard with:
  - Yearly trends
  - Top affected countries
  - Predicted incidence charts
  - (Optional) Map using `malaria_cleaned_with_coords.csv`

---

## 🧰 Tools Used

- **Python (pandas, matplotlib, scikit-learn)**
- **Power BI**
- **VS Code**
- **Jupyter Notebook**

---

## 👨‍🎓 Author

**Rugandura Maurice**  
Capstone Project – Big Data Analytics  
Instructor: *Eric Maniraguha*

---


## 📬 Contact

For any questions or feedback, feel free to reach out!

---


