# Automated Dataset Explorer

## Project Goal

Open Dataset Explorer is designed to streamline and automate the process of exploring and analyzing datasets using machine learning. Built with ease-of-use in mind, this app enables users to upload any dataset, select features and targets, train multiple ML models, and gain insights through evaluation metrics and explainability techniques like SHAP. The goal is to empower data enthusiasts, researchers, and professionals to rapidly derive meaningful insights without deep coding or setup complexity.

---

## Features

- Upload your own CSV dataset  
- Select target and feature columns dynamically  
- Train and compare multiple models: Random Forest, Logistic Regression, KNN, XGBoost  
- View model evaluation metrics: accuracy, precision, recall, F1 score  
- Visualize feature importance and confusion matrices  
- Explore model explainability using SHAP visualizations  
- Download the trained model and SHAP plots for later use

---

## Getting Started

### Prerequisites

- Python 3.8 or higher installed on your system  
- Recommended: a virtual environment to keep dependencies isolated

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/mccainwa/Automated-Dataset-Explorer.git
   cd Automated-Dataset-Explorer

2. (Optional but recommended) Create and activate a virtual environment:

   On Windows:
      ```bash
      python -m venv venv
      venv\Scripts\activate

4. Install required Python packages:

      ```bash
   pip install -r requirements.txt

5. (Optional) Generate the sample Iris dataset:

      ```bash
   python generate_iris.py

6. Run the Streamlit app:

      ```bash
   streamlit run app.py

7. Open your browser at the address shown in the terminal (usually http://localhost:8501).

---

Required Packages

All required packages are listed in requirements.txt and include:

- streamlit
- pandas
- scikit-learn
- xgboost
- matplotlib
- shap

---

Usage Tips

- Upload your dataset using the app interface
- Select the appropriate features and target columns
- Choose the ML model you want to train
- Adjust train-test split and random seed for reproducibility
- Review model evaluation and interpretability plots
- Download your trained model and SHAP plots for sharing or future analysis

---

License

This project is licensed under the MIT License.

---

Contact

Created by Walter McCain III
GitHub Profile: https://github.com/mccainwa
