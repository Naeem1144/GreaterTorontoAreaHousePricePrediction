# Greater Toronto Area (GTA) House Price Prediction

## Project Overview

This project focuses on predicting house prices in the Greater Toronto Area (GTA), one of Canada's most dynamic and competitive real estate markets. Using a dataset of historical home prices and relevant features, I developed and evaluated several machine learning models to forecast average house prices. The goal is to provide accurate and reliable predictions that can be valuable for buyers, sellers, investors, and real estate professionals in the GTA.

## Dataset

The dataset used in this project is `GTA_HomePrice_History.csv`. It contains a rich set of features related to house sales in the GTA, including:

*   **Geographic Information:** Area, Municipality, Community
*   **Sales and Price Data:** Sales, Dollar Volume, Average Price
*   **Listing Information:** New Listings, Average SP/LP (Selling Price to Listing Price ratio)
*   **Market Dynamics:** Average DOM (Days on Market)
*   **Temporal Information:** Year, Quarter, Year_Quarter_Key
*   **Building Type**

The dataset is preprocessed to handle missing values (using KNN imputation) and categorical variables (using one-hot encoding).

## Key Features of the Project

*   **Data Preprocessing:** Implemented a comprehensive data preprocessing pipeline, including one hot encoding for categorical features and numerical features.
*   **Model Selection:** Explored various machine learning models, including:
    *   Linear Regression
    *   Decision Tree Regressor
    *   Gradient Boosting Regressor
    *   Artificial Neural Network (ANN) using PyTorch
*   **Model Evaluation:** Rigorously evaluated models using metrics such as R-squared (R2), Mean Squared Error (MSE), and Mean Absolute Error (MAE).
*   **Cross-Validation:** Employed 10-fold cross-validation to ensure the robustness and generalizability of the Gradient Boosting Regressor.
*   **ANN Implementation:** Developed a custom ANN architecture with multiple layers, including BatchNorm, to capture complex relationships in the data.
*   **Early Stopping:** Implemented early stopping during ANN training to prevent overfitting and improve model performance.
*   **Data Visualization:** Included visualizations (KDE plots, line plots) to demonstrate the effectiveness of the data preprocessing steps and model training.
*   **Model Interpretability:** Used residual plots to check the assumptions of linear models.

## Technologies Used

*   **Programming Language:** Python
*   **Libraries:**
    *   Pandas
    *   NumPy
    *   Scikit-learn
    *   PyTorch
    *   Matplotlib
    *   Seaborn

## Getting Started

### Prerequisites

To run this project, you need to have Python 3 installed along with the following libraries:

1. You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch
```
<center>OR</center>

1. You can use the text file to import the necessary libraries
```bash
pip install -r requirements.txt
```
2. start with cloning the repository
```bash
git clone https://github.com/Naeem1144/GreaterTorontoAreaHousePricePrediction.git
```
3. Navigate to the project directory:
```bash
cd GreaterTorontoAreaHousePricePrediction
```
4. Place the GTA_HomePrice_History.csv dataset in the project directory.

# Greater Toronto Area House Price Prediction

This project focuses on predicting house prices in the Greater Toronto Area using machine learning models. The best-performing models include the Gradient Boosting Regressor and an Artificial Neural Network (ANN).

## Usage

1. Open the Jupyter Notebook `GreaterTorontoAreaHousePricePrediction.ipynb` using **Jupyter** or **Google Colab**.
2. Run the cells sequentially to execute the data preprocessing, model training, and evaluation steps.

## Results

### Gradient Boosting Regressor
The Gradient Boosting Regressor achieved the following results on the test set:

- **R2 Score:** 0.9897899326815385  
- **Mean Squared Error:** 0.00974579489651158  
- **Mean Absolute Error:** 0.04747853751128085  

### Artificial Neural Network (ANN)
The ANN also demonstrated strong performance with the following results:

- **R2 Score:** 0.9844423  
- **Mean Squared Error:** 0.014850256  

## Model Performance

### Gradient Boosting Regressor
The Gradient Boosting Regressor performed exceptionally well, achieving an R2 score of approximately **0.99** on the test set. The cross-validation scores over 10 iterations are visualized below:

![GBR Cross-Validation Plot](https://github.com/Naeem1144/GreaterTorontoAreaHousePricePrediction/blob/main/gbr_cv_plot.png)

### Artificial Neural Network (ANN)
The ANN model also showed strong performance, achieving an R2 score of approximately **0.98**. The training and test loss curves are shown below:

![ANN Loss Curves](https://github.com/Naeem1144/GreaterTorontoAreaHousePricePrediction/blob/main/ann_loss_curves.png)
