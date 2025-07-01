# Housing Price Prediction — Data Preprocessing and Custom Transformers

This Jupyter notebook contains the full walkthrough of Chapter 2 from **Aurélien Géron's "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow"**, focused on data preprocessing, feature engineering, and building custom transformers.

---

## Contents

- **Data Loading and Exploration**  
  Initial inspection of the housing dataset, exploration of key features and labels.

- **Data Cleaning and Imputation**  
  Handling missing values using `SimpleImputer`.

- **Feature Engineering**  
  Creating new features such as ratios (e.g., `rooms_per_household`, `people_per_household`).

- **Custom Transformer: KNeighborsRegressor Wrapper**  
  Implemented a transformer that wraps a k-Nearest Neighbors regressor, training it on latitude and longitude to generate a new feature estimating median housing price from nearby districts.

- **Custom Transformer: StandardScalerClone**  
  Built a custom scaler replicating `StandardScaler` functionality, including:
  - Fit and transform methods
  - Inverse transform support
  - Proper handling of feature names

- **Pipelines and ColumnTransformers**  
  Combined various preprocessing steps into pipelines for numerical and categorical features. Integrated custom transformers seamlessly.

- **Hyperparameter Tuning**  
  Discussed tuning hyperparameters of the custom KNN regressor transformer within the pipeline.

- **Feature Names and Interpretability**  
  Added `get_feature_names_out()` methods to custom transformers for pipeline compatibility and clearer feature naming.

---

## Key Learnings

- How to implement custom transformers by extending `BaseEstimator` and `TransformerMixin`.
- Importance of handling pandas DataFrames vs. numpy arrays in transformers.
- Usage of `check_array`, `check_is_fitted`, and managing input/output feature names.
- Constructing reusable preprocessing pipelines with `Pipeline` and `ColumnTransformer`.
- Techniques for incorporating machine learning models (like KNN) as feature generators inside transformers.
- How to expose and tune hyperparameters of nested estimators.
- Best practices for making custom transformers compatible with scikit-learn tools.

---

## Usage

- Run the notebook sequentially to follow the preprocessing and feature engineering steps.
- Modify parameters in the custom transformers and pipelines to experiment with different setups.
- Use the provided code snippets to build similar custom transformers for other regressors or estimators.

---

## Requirements

- Python 3.7+
- scikit-learn
- numpy
- pandas
- matplotlib (optional, for visualization)

You can install dependencies using:

```bash
pip install scikit-learn numpy pandas matplotlib
