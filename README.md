# Car Sales Price Prediction Using Machine Learning

## 📌 Overview
This project explores machine learning techniques for predicting second-hand car prices using various supervised and unsupervised learning models. By leveraging regression and clustering techniques, the goal is to develop an accurate model that helps determine car prices based on parameters such as **mileage, year of manufacture, fuel type, engine size, model, and manufacturer**.

## 📂 Project Structure
- **`car_sales_dataset.csv`** - The dataset containing 50,000 car sales records.
- **`Car_Sales.pdf`** - A detailed scientific report on the car price prediction models used.
- **`A_Abidoye_Artificial_Intelligence_Car_Sales_Submission.ipynb`** - Jupyter Notebook containing data preprocessing, model training, and evaluation.
- **`figures/`** - Directory for storing plots and evaluation visuals.

## 🛠️ Technologies Used
- **Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras)**
- **Jupyter Notebook**
- **Supervised Learning Models:**
  - Simple Linear Regression
  - Polynomial Regression (2nd Degree)
  - Multiple Linear Regression
  - Random Forest Regression
  - Artificial Neural Networks (ANN)
- **Unsupervised Learning Models:**
  - k-Means Clustering
  - Agglomerative Hierarchical Clustering
  - DBSCAN Clustering
- **Evaluation Metrics:**
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score
  - Davies Bouldin Index
  - Silhouette Score

## 🔹 Key Features
### **1. Data Processing & Preprocessing**
- Cleaning and preparing car sales data.
- Handling missing values and outliers.
- Feature engineering and transformations.

### **2. Machine Learning Models for Car Price Prediction**
- **Regression Models:** Evaluating **linear and non-linear** approaches to predict car prices.
- **Random Forest Regression:** Best-performing model with **99.8% R² accuracy**.
- **Artificial Neural Network (ANN):** Evaluating hyperparameter tuning and dropout rates.

### **3. Unsupervised Learning for Data Clustering**
- **k-Means Clustering:** Identifying pricing patterns based on key features.
- **DBSCAN & Agglomerative Clustering:** Grouping cars based on similar characteristics.

### **4. Model Evaluation & Results**
- Comparison of regression models based on **R² scores** and **error metrics**.
- Heatmaps and correlation analysis between features.
- Optimal cluster determination using **Silhouette Scores and Davies Bouldin Index**.

## 📊 Visualizations & Insights
The repository includes multiple **data visualizations** such as:
- **Heatmaps** for feature correlation.
- **Prediction vs Actual Price Plots** for regression models.
- **Clustering results** with optimal **K values**.
- **ANN hyperparameter tuning loss plots**.

## 🚀 Getting Started
### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/iabidoye/Car_Sales_Prediction.git
   cd Car_Sales_Prediction


2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn tensorflow keras
   
3. Open the Jupyter Notebook and run the analysis:
  ```bash
  jupyter notebook A_Abidoye_Artificial_Intelligence_Car_Sales_Submission.ipynb

🤝 Contribution
Contributions are welcome! Feel free to submit pull requests with improvements, additional models, or new datasets.

📧 Contact
For inquiries or collaborations, please reach out.
