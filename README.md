# AI-Powered Food Waste Prediction System

## Overview

The AI-Powered Food Waste Prediction System is a machine learning-based solution designed to estimate food waste using inventory, sales, product, and environmental data. The project combines predictive analytics with business intelligence to support inventory optimization and reduce unnecessary food wastage in retail environments.

---

## Problem Statement

Food retailers often experience losses due to overstocking, changing customer demand, and perishable inventory. This project addresses the problem by predicting food waste quantities, enabling better inventory planning and data-driven decision-making.

---

## Features

- Data preprocessing and feature engineering
- Predictive modeling using multiple regression algorithms
- Model comparison and evaluation
- SQL-based business analysis
- Interactive Power BI dashboard
- Streamlit web application for real-time predictions
- Export predicted results as CSV

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Database | MySQL |
| Data Visualization | Power BI |
| Web Application | Streamlit |
| Model Serialization | Joblib |

---

## Project Structure

```
AI_Food_Waste_Prediction/
│
├── data/
├── models/
├── notebooks/
├── reports/
├── sql/
├── src/
├── streamlit_app/
├── README.md
└── requirements.txt
```

---

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Model Selection
7. Prediction
8. Dashboard Development
9. Streamlit Deployment

---

## Model Evaluation

The following regression models were trained and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Based on evaluation metrics including MAE, RMSE, and R² Score, the Random Forest Regressor demonstrated the best overall performance and was selected as the final prediction model.

---

## Dashboard

The Power BI dashboard provides insights into:

- Total Waste Quantity
- Total Waste Cost
- Average Waste
- Waste by Category
- Waste by City
- Waste Risk Distribution
- Top Waste-Producing Stores

---

## Streamlit Application

The web application enables users to:

- Upload processed inventory data
- Generate food waste predictions
- View prediction summary
- Download prediction results

---

## Future Enhancements

- Cloud deployment
- Real-time prediction pipeline
- Automated inventory recommendations
- REST API integration
- Advanced forecasting models

---

## Author

**Dhanusri Abburi**

B.Tech – Computer Science and Engineering

Email: *your-email@example.com*

LinkedIn: https://www.linkedin.com/in/dhanusri-abburi-23979b28b/

GitHub: https://github.com/Dhanusree-max