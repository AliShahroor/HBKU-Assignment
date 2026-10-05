# Iris Classification Using C5.0 Decision Tree in R

This project implements a C5.0 decision tree classifier using the Iris dataset in R.

## Objective

The objective is to train a decision tree model to classify Iris flowers into three varieties:

- Setosa
- Versicolor
- Virginica

The classification is based on four features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

## Method

The dataset contains 150 samples.

The data was randomly shuffled using a fixed seed for reproducibility and then divided into:

- 70% training data (105 samples)
- 30% testing data (45 samples)

The C5.0 decision tree was trained using the four flower measurements as input features and `variety` as the target class.

## Results

The trained model was evaluated on the 45 unseen test samples.

- Correct predictions: 44
- Incorrect predictions: 1
- Test accuracy: 97.78%
- Test error rate: 2.22%

### Confusion Matrix

| Actual | Setosa | Versicolor | Virginica |
|---|---:|---:|---:|
| Setosa | 14 | 0 | 0 |
| Versicolor | 0 | 17 | 1 |
| Virginica | 0 | 0 | 13 |

The model correctly classified all Setosa and Virginica samples in the test set. Only one Versicolor sample was incorrectly classified as Virginica.

The decision tree primarily used `petal.length` and `petal.width` to distinguish between the three classes.

## Technologies

- R
- Google Colab
- C50
- gmodels

## Files

- `Iris_C5_Decision_Tree.ipynb` – R notebook containing the complete analysis
- `iris.csv` – Iris dataset used for training and testing

## Running the Notebook

The notebook can be opened and executed using Google Colab with an R runtime.

The required R packages are:

```r
install.packages("C50")
install.packages("gmodels")
