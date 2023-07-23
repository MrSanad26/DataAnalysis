
# Telco Customer Churn Analysis
This repository contains a Python script for analyzing Telco customer churn data and building a predictive model using random forest classification. The analysis is performed in a Google Colab notebook.

## Getting Started

To run the code and reproduce the analysis, follow these steps:

1- Open the provided link to access the Google Colab notebook.

2- Install the required library 'pandas-profiling' by executing the cell:

#### !pip install pandas-profiling --upgrade

3- Import the necessary libraries by running the initial code cells.

4- Load the Telco customer churn data from the specified CSV file in Google Drive:

#### df = pd.read_csv("From your files/WA_Fn-UseC_-Telco-Customer-Churn.csv")

5- Explore the data using various methods like **info(), describe()**, and visualizations.

6- Perform data preprocessing steps like handling missing values, converting data types, and encoding categorical variables.

7- Split the dataset into training and testing sets using **train_test_split**.

8- Build and evaluate three models: Logistic Regression, Decision Tree, and Random Forest Classifier.

9- Perform hyperparameter tuning using Grid Search to find the best model hyperparameters.

10- Create the final Random Forest model using the best hyperparameters obtained from Grid Search.

11- Save the final model as a pickle file for future use.

## Model Evaluation
After training the final Random Forest model, the script evaluates its performance on the test set and provides the following metrics:

**Confusion Matrix**: A table that shows the true positive, true negative, false positive, and false negative predictions.

**Accuracy Score**: The proportion of correctly classified instances among all instances.

**Precision Score:** The ability of the model to identify only relevant instances among the predicted positive instances.

**Recall Score**: The ability of the model to find all the positive instances.

## Dependencies
The analysis uses the following Python libraries:

pandas
numpy
seaborn
matplotlib
scikit-learn

## Note

Ensure you have access to the Telco customer churn dataset in your Google Drive at the specified path to execute the code seamlessly.

Feel free to modify the code or add further analyses based on your requirements.

Happy coding!
