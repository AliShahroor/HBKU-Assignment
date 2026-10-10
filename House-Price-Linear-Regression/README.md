# House Price Prediction Using Linear Regression in R

This project demonstrates how to predict house prices using **Multiple Linear Regression** in R through Google Colab.

The notebook is designed for beginners and includes step-by-step explanations of the code, model training, predictions, and evaluation.

## Objective

The objective is to predict house prices (`Price`) using four numerical features:

- **SquareFeet:** Size of the house in square feet
- **Bedrooms:** Number of bedrooms
- **Bathrooms:** Number of bathrooms
- **Age:** Age of the house

The dataset contains **100 houses**, each with these four features and its corresponding price.

## Project Structure

```text
House-Price-Linear-Regression/
│
├── House_Price_Linear_Regression_R.ipynb
├── house_prices.csv
└── README.md
```

- `House_Price_Linear_Regression_R.ipynb` — Google Colab notebook containing the R code, explanations, and analysis.
- `house_prices.csv` — Dataset containing 100 house records.
- `README.md` — Project documentation.

## How to Run in Google Colab

1. Open `House_Price_Linear_Regression_R.ipynb` in Google Colab.
2. Make sure the notebook is using an **R runtime**.
3. Upload `house_prices.csv` to the Colab Files panel.
4. Run the code cells from top to bottom.
5. Review the model summary, predicted prices, evaluation metrics, and visualization.

**Note:** Google Colab uses temporary runtime storage, so you may need to upload the dataset again when starting a new session.

The project uses **base R**, so no additional packages are required.

## Methodology

### Step 1: Load and Explore the Dataset

Load the CSV file into R and examine the first few rows and dataset structure.

### Step 2: Prepare and Split the Data

Randomly shuffle the dataset using `set.seed(123)` for reproducibility.

Split the 100 houses into:

- **Training set (80%):** 80 houses used to train the model.
- **Testing set (20%):** 20 houses used to evaluate the model.

### Step 3: Train the Linear Regression Model

Use R's `lm()` function to train a multiple linear regression model.

```r
model <- lm(Price ~ SquareFeet + Bedrooms + Bathrooms + Age,
            data = train_data)
```

The model learns the relationship between the four input features and house prices.

### Step 4: Predict House Prices

Use the trained model to predict prices for the 20 houses in the testing dataset.

```r
predictions <- predict(model, newdata = test_data)
```

Compare the predicted prices with the actual prices.

### Step 5: Evaluate Model Performance

Evaluate the model using three regression metrics:

- **R² (R-squared):** Measures how well the model predicts house prices compared with predicting the test-set mean. Higher values generally indicate better performance.
- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted prices.
- **RMSE (Root Mean Squared Error):** Measures prediction error while giving greater weight to larger mistakes.

For MAE and RMSE, lower values indicate better predictions.

### Step 6: Visualize the Results

Create a scatter plot comparing actual and predicted house prices.

A red reference line represents perfect predictions. Points closer to the line indicate more accurate predictions.

### Optional: Predict a New House Price

Enter the characteristics of a new house and use the trained model to estimate its price.

## Results and Interpretation

The notebook evaluates the trained model on **20 unseen houses**.

The results include:

- Predicted versus actual house prices
- Test R²
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Actual versus predicted price visualization

The evaluation helps determine how closely the model's predicted prices match the actual prices.

**Note:** Because this is a regression problem with a continuous numerical target, a confusion matrix is not used. Confusion matrices are typically used for classification problems.

## Technologies Used

- **R** — Programming language for statistical analysis
- **Google Colab** — Cloud-based environment for running R notebooks
- **Linear Regression** — Machine learning method for predicting numerical values
- **GitHub** — Project storage and version control

## Learning Outcomes

By completing this project, students will learn how to:

1. Load and explore a dataset using R.
2. Split data into training and testing sets.
3. Train a multiple linear regression model.
4. Predict numerical values using a trained model.
5. Evaluate prediction errors using regression metrics.
6. Interpret and visualize model performance.