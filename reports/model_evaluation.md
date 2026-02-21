# Model Training and Evaluation Report

## 1. Train/Validation/Test Split
The dataset was split into three distinct sets to ensure robust training and fair evaluation:
- **Train Set (70%, 3917 samples):** Used to fit the XGBoost model.
- **Validation Set (15%, 840 samples):** Used during training for early stopping to prevent overfitting.
- **Test Set (15%, 840 samples):** Held out until the end. Used to measure the final generalized performance.

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
- **Test RMSE:** Rs 42,242.80
- **Test MAE:** Rs 22,399.48
- **Test $R^2$:** 0.8358

### What these results indicate:
The $R^2$ of **0.8358** indicates that the model successfully explains approximately 83.6% of the variance in computer prices on the test data. The Mean Absolute Error (MAE) shows that, on average, the model's price prediction is off by **Rs 22,399.48**. Given the price scale of computers and the noise inherent in user-generated classified listings (ikman.lk), this represents a robust initial estimate!

## 5. Visualizations

### Actual vs Predicted
*(Since this is a textual summary artifact, you can view the actual graphical artifact located at `d:/Projects/Sri_Lanka_Computer_Price_Predictor/reports/figures/actual_vs_predicted.png`)*  
The Actual vs Predicted scatterplot plots a 45-degree line. Points closely following this red dashed line indicate highly accurate predictions.

### Feature Importance
*(Located at `d:/Projects/Sri_Lanka_Computer_Price_Predictor/reports/figures/feature_importance.png`)*  
By observing the feature importance plot, we can see exactly which variables the XGBoost tree nodes decided were most crucial for splitting the data to determine the price. Usually, hardware capabilities like `RAM_GB`, `CPU_Tier`, and `Has_Dedicated_GPU` drive the cost variations heavily.
