# 🎓 AI-Based Student Performance Prediction System

## 📌 Overview
This project is a web-based application that uses **Machine Learning** to predict a student's **CGPA (Cumulative Grade Point Average)** based on key academic factors.

The system also provides:
- 📊 Future performance trends
- 🧠 Data-driven explanations
- 📄 Automated PDF reports
- 🔍 Explainable AI (feature impact analysis)

---

## 🚀 Features

### 🔹 Prediction System
- Predicts CGPA based on:
  - Attendance
  - Study Hours
  - Previous SGPA

### 🔹 Data Visualization
- Displays future CGPA trend using interactive charts

### 🔹 AI Explanation
- Provides insights by comparing user input with dataset averages

### 🔹 Explainable AI
- Shows how each feature affects the prediction

### 🔹 PDF Report Generation
- Generates a professional report including:
  - Summary
  - Analysis
  - Feature impact
  - Graphs

### 🔹 User Authentication
- Login and registration system using SQLite

---

## 🧠 Machine Learning

- Model: **Gradient Boosting Regressor**
- Tuning: **GridSearchCV**
- Preprocessing:
  - Data cleaning
  - Feature scaling
  - Handling missing values
- Feature Engineering:
  - `study_effectiveness = attendance × study_hours`

---

## 🗂️ Project Structure
project/
│── app.py
│── Students_Performance_data_set.xlsx
│── users.db
└── templates/
├── login.html
└── dashboard.html

---

## ⚙️ Installation

### 1. Clone or Download the Project

### 2. Install Dependencies

```bash
pip install flask flask-socketio pandas numpy matplotlib scikit-learn reportlab openpyxl
