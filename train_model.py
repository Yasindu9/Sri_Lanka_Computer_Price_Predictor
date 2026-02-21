import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_and_eval():
    print("Loading data...")
    df = pd.read_csv('data/processed/computers_data_withprocessed.csv')
    
    # 1. Prepare Features & Target
    print("Preparing features...")
    X = df.drop('Price (Target)', axis=1)
    y = df['Price (Target)']
    
    # XGBoost handles categorical variables naturally if type is set to 'category'
    # For versions >= 1.5.0
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].astype('category')
        
    # 2. Train/Validation/Test Split
    # We want 70% train, 15% validation, 15% test
    print("Splitting data -> Train 70%, Val 15%, Test 15%...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    # Remaining 85% is in X_temp, we need 15% of total for validation. 
    # 15 / 85 is approx 0.17647
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), random_state=42)
    
    print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 3. Model Definition and Hyperparameters
    print("Defining XGBoost model...")
    # These hyperparameters are chosen to handle potential overfitting
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.05,
        'max_depth': 6,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'enable_categorical': True, # Important for categorical features
        'random_state': 42,
        'early_stopping_rounds': 20
    }
    
    model = xgb.XGBRegressor(**params)
    
    # 4. Model Training
    print("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    # 5. Evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R^2: {r2:.4f}")
    
    # 6. Plots & Results
    os.makedirs('reports', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    
    # Plot 1: Feature Importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10, height=0.5, grid=False)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png')
    plt.close()
    
    # Plot 2: Actual vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Price (Rs)')
    plt.ylabel('Predicted Price (Rs)')
    plt.title('Actual vs Predicted Prices (Test Set)')
    plt.tight_layout()
    plt.savefig('reports/figures/actual_vs_predicted.png')
    plt.close()
    
    # Produce Markdown Report text
    report = f"""# Model Training and Evaluation Report

## 1. Train/Validation/Test Split
The dataset was split into three distinct sets to ensure robust training and fair evaluation:
- **Train Set (70%, {X_train.shape[0]} samples):** Used to fit the XGBoost model.
- **Validation Set (15%, {X_val.shape[0]} samples):** Used during training for early stopping to prevent overfitting.
- **Test Set (15%, {X_test.shape[0]} samples):** Held out until the end. Used to measure the final generalized performance.

## 2. Hyperparameter Choices
We selected the `XGBRegressor` model with the following key hyperparameter initialization:
- `learning_rate` (**0.05**): A smaller learning rate prevents the model from updating too aggressively and prevents overshooting the optimal weights.
- `max_depth` (**6**): Limits the depth of decision trees. Allows capturing non-linear patterns without deeply overfitting to noise.
- `n_estimators` (**300**): Maximum number of boosted trees to build.
- `subsample` (**0.8**): Trains each tree using a random 80% subsample of the rows. Reduces variance.
- `colsample_bytree` (**0.8**): Uses a random 80% of features for each tree. Acts as regularization similarly to Random Forest.
- `early_stopping_rounds` (**20**): If the validation error does not improve for 20 consecutive rounds, training stops early to enforce generalization.

## 3. Performance Metrics
Because we are predicting continuous values (`Price`), we use regression metrics instead of classification metrics (like Accuracy, F1, or AUC):
- **Root Mean Squared Error (RMSE):** Represents the square root of differences between predicted and actual values. It penalizes large errors heavily.
- **Mean Absolute Error (MAE):** The average absolute difference between predicted and actual prices. It is less sensitive to outliers than RMSE and directly interpretable as "how far off the prediction was on average".
- **R-squared ($R^2$):** Represents the proportion of variance in the dependent variable (Price) predictable from the independent variables (features). Best possible score is 1.0.

## 4. Results Obtained
- **Test RMSE:** Rs {rmse:,.2f}
- **Test MAE:** Rs {mae:,.2f}
- **Test $R^2$:** {r2:.4f}

### What these results indicate:
The $R^2$ of **{r2:.4f}** indicates that the model successfully explains approximately {r2 * 100:.1f}% of the variance in computer prices on the test data. The Mean Absolute Error (MAE) shows that, on average, the model's price prediction is off by **Rs {mae:,.2f}**. Given the price scale of computers and the noise inherent in user-generated classified listings (ikman.lk), this represents a robust initial estimate!

## 5. Visualizations

### Actual vs Predicted
*(Since this is a textual summary artifact, you can view the actual graphical artifact located at `d:/Projects/Sri_Lanka_Computer_Price_Predictor/reports/figures/actual_vs_predicted.png`)*  
The Actual vs Predicted scatterplot plots a 45-degree line. Points closely following this red dashed line indicate highly accurate predictions.

### Feature Importance
*(Located at `d:/Projects/Sri_Lanka_Computer_Price_Predictor/reports/figures/feature_importance.png`)*  
By observing the feature importance plot, we can see exactly which variables the XGBoost tree nodes decided were most crucial for splitting the data to determine the price. Usually, hardware capabilities like `RAM_GB`, `CPU_Tier`, and `Has_Dedicated_GPU` drive the cost variations heavily.
"""

    with open('reports/model_evaluation.md', 'w') as f:
        f.write(report)
        
    # Save Model and Categories
    import json
    os.makedirs('src/models', exist_ok=True)
    model.save_model('src/models/xgb_model.json')
    categories_dict = {col: list(X[col].cat.categories) for col in categorical_cols}
    with open('src/models/categories.json', 'w') as f:
        json.dump(categories_dict, f)

    print("Finished evaluating and saving reports and model.")
    

if __name__ == "__main__":
    train_and_eval()
