import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt

# Load the macroeconomic and VIX datasets
macro_data = pd.read_csv("us-financial-indicators-1974-to-2024.csv")
vix_data = pd.read_csv("VIX_History.csv")

# Convert the source date columns
macro_data["date"] = pd.to_datetime(macro_data["date"], errors="coerce")
vix_data["DATE"] = pd.to_datetime(vix_data["DATE"], errors="coerce")

# Select the final available VIX closing value in each month
vix_data["Month"] = vix_data["DATE"].dt.to_period("M")
monthly_vix = vix_data.sort_values("DATE").groupby("Month", as_index=False).last()
monthly_vix["date"] = monthly_vix["Month"].dt.to_timestamp()
monthly_vix = monthly_vix[["date", "CLOSE"]].rename(columns={"CLOSE": "VIX_Close"})

# Merge the monthly macroeconomic and VIX observations
df = macro_data.merge(monthly_vix, on="date", how="inner").sort_values("date").reset_index(drop=True)

# Calculate the monthly S&P 500 return
df["SP500_Return"] = df["sp500"].pct_change() * 100

# Shift the return backward so each row predicts the following month
df["Next_Month_Return"] = df["SP500_Return"].shift(-1)

# Transform the macroeconomic and market sentiment variables
df["InterestRate_Change"] = df["Interest_Rate"].diff()
df["Inflation_YoY"] = df["Inflation"].pct_change(12) * 100
df["GDP_Growth_YoY"] = df["GDP"].pct_change(12) * 100
df["Unemployment_Change"] = df["Unemployment"].diff()
df["IndProd_Growth_YoY"] = df["Ind_Prod"].pct_change(12) * 100
df["VIX_Change"] = df["VIX_Close"].pct_change() * 100

# Define the transformed predictors to investigate
base_features = [
    "InterestRate_Change",
    "Inflation_YoY",
    "GDP_Growth_YoY",
    "Unemployment_Change",
    "IndProd_Growth_YoY",
    "VIX_Close",
    "VIX_Change"
]

# Create one-month and three-month lagged versions of each predictor
candidate_features = base_features.copy()
for feature in base_features:
    for lag in [1, 3]:
        lagged_feature = f"{feature}_Lag{lag}"
        df[lagged_feature] = df[feature].shift(lag)
        candidate_features.append(lagged_feature)

# Keep complete observations for comparable diagnostics
analysis_data = df[["date", "Next_Month_Return"] + candidate_features].dropna().copy()

# Run the ADF stationarity test for each variable
def run_adf_test(series, variable):
    result = adfuller(series)
    return {
        "Variable": variable,
        "ADF_Statistic": result[0],
        "P_Value": result[1],
        "Stationary_At_5_Percent": result[1] < 0.05
    }

adf_results = [run_adf_test(analysis_data["Next_Month_Return"], "Next_Month_Return")]
for feature in candidate_features:
    adf_results.append(run_adf_test(analysis_data[feature], feature))
adf_results = pd.DataFrame(adf_results)

# Calculate correlations between the candidate predictors and next month's return
target_correlations = (
    analysis_data[["Next_Month_Return"] + candidate_features]
    .corr()["Next_Month_Return"]
    .drop("Next_Month_Return")
    .sort_values(key=abs, ascending=False)
    .rename("Correlation")
    .reset_index()
    .rename(columns={"index": "Variable"})
)

# Calculate VIF values for all candidate predictors
vif_input = analysis_data[candidate_features].copy()
vif_results = pd.DataFrame({
    "Variable": candidate_features,
    "VIF": [
        variance_inflation_factor(vif_input.values, index)
        for index in range(vif_input.shape[1])
    ]
}).sort_values("VIF", ascending=False)

# Calculate the majority-direction benchmark
positive_return_percentage = (analysis_data["Next_Month_Return"] > 0).mean() * 100
negative_return_percentage = (analysis_data["Next_Month_Return"] < 0).mean() * 100
majority_direction_accuracy = max(positive_return_percentage, negative_return_percentage)

# Display the diagnostic dataset coverage
print("\nDIAGNOSTIC DATASET")
print("Start:", analysis_data["date"].min())
print("End:", analysis_data["date"].max())
print("Observations:", len(analysis_data))
print("Candidate features:", len(candidate_features))

# Display the target direction balance
print("\nTARGET DIRECTION BALANCE")
print("Positive months:", round(positive_return_percentage, 2))
print("Negative months:", round(negative_return_percentage, 2))
print("Majority-direction benchmark:", round(majority_direction_accuracy, 2))

# Display correlations with next month's return
print("\nCORRELATIONS WITH NEXT MONTH RETURN")
print(target_correlations.to_string(index=False))

# Display the stationarity test results
print("\nADF STATIONARITY RESULTS")
print(adf_results.to_string(index=False))

# Display the full candidate feature multicollinearity results
print("\nVIF RESULTS")
print(vif_results.to_string(index=False))

# Save the full diagnostic dataset for traceability
analysis_data.to_csv("Stage3_Diagnostic_Dataset.csv", index=False)
print("\nDiagnostic dataset saved as Stage3_Diagnostic_Dataset.csv")

# Define predictors that would have been available at the forecast origin
selected_features = [
    "InterestRate_Change_Lag3",
    "Inflation_YoY_Lag1",
    "GDP_Growth_YoY_Lag1",
    "Unemployment_Change_Lag1",
    "VIX_Change"
]

# Create the reduced diagnostic dataset
reduced_data = analysis_data[["date", "Next_Month_Return"] + selected_features].dropna().copy()

# Display the reduced feature correlation matrix
print("\nREDUCED FEATURE CORRELATION MATRIX")
print(reduced_data[["Next_Month_Return"] + selected_features].corr())

# Calculate VIF values for the reduced feature set
reduced_vif = pd.DataFrame({
    "Variable": selected_features,
    "VIF": [
        variance_inflation_factor(reduced_data[selected_features].values, index)
        for index in range(len(selected_features))
    ]
}).sort_values("VIF", ascending=False)

# Display the reduced feature VIF results
print("\nREDUCED FEATURE VIF RESULTS")
print(reduced_vif.to_string(index=False))

# Display the reduced dataset size
print("\nREDUCED DATASET SUMMARY")
print("Rows:", len(reduced_data))
print("Features:", len(selected_features))


# Define the final modelling features
final_features = selected_features.copy()

# Create the final modelling dataset using the selected lagged predictors
model_data = df[["date", "SP500_Return", "Next_Month_Return"] + final_features].dropna().copy()

# Produce descriptive statistics for the final modelling variables
descriptive_statistics = (
    model_data[
        [
            "Next_Month_Return",
            "InterestRate_Change_Lag3",
            "Inflation_YoY_Lag1",
            "GDP_Growth_YoY_Lag1",
            "Unemployment_Change_Lag1",
            "VIX_Change"
        ]
    ]
    .describe()
    .T
)

print("\nFINAL DESCRIPTIVE STATISTICS")
print(descriptive_statistics)

descriptive_statistics.to_csv(
    "Final_Descriptive_Statistics.csv"
)

print("\nSaved: Final_Descriptive_Statistics.csv")

# Split the dataset chronologically into 80 percent training and 20 percent testing data
split_index = int(len(model_data) * 0.80)
train = model_data.iloc[:split_index].copy()
test = model_data.iloc[split_index:].copy()

# Separate the predictors and target variable
X_train = train[final_features]
X_test = test[final_features]
y_train = train["Next_Month_Return"]
y_test = test["Next_Month_Return"]

# Display the chronological training and testing periods
print("\nTRAINING AND TESTING SUMMARY")
print("Total observations:", len(model_data))
print("Training observations:", len(train))
print("Testing observations:", len(test))
print("Training period:", train["date"].min(), "to", train["date"].max())
print("Testing period:", test["date"].min(), "to", test["date"].max())

# Calculate the directional benchmark within the unseen test period
test_positive_percentage = (y_test > 0).mean() * 100
test_negative_percentage = (y_test < 0).mean() * 100
test_majority_direction = max(test_positive_percentage, test_negative_percentage)

# Display the test-period direction balance
print("\nTEST PERIOD DIRECTION BALANCE")
print("Positive months:", round(test_positive_percentage, 2))
print("Negative months:", round(test_negative_percentage, 2))
print("Majority-direction benchmark:", round(test_majority_direction, 2))

# Create the historical mean benchmark using only training data
historical_mean_prediction = np.repeat(y_train.mean(), len(y_test))

# Create the previous-return benchmark using each observation's current return
previous_return_prediction = test["SP500_Return"].to_numpy()

# Create a linear regression pipeline with standardised predictors
linear_regression = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# Fit the linear regression model using the training data
linear_regression.fit(X_train, y_train)

# Generate linear regression predictions for the test period
linear_predictions = linear_regression.predict(X_test)

# Create a function for calculating consistent model evaluation metrics
def evaluate_model(model_name, actual, predicted):
    actual_values = np.asarray(actual)
    predicted_values = np.asarray(predicted)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values, predicted_values)
    r_squared = r2_score(actual_values, predicted_values)
    directional_accuracy = (
        np.sign(actual_values) == np.sign(predicted_values)
    ).mean() * 100
    return {
        "Model": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R_Squared": r_squared,
        "Directional_Accuracy": directional_accuracy
    }

# Evaluate the benchmark and linear regression models
results = [
    evaluate_model(
        "Historical Mean Baseline",
        y_test,
        historical_mean_prediction
    ),
    evaluate_model(
        "Previous Return Baseline",
        y_test,
        previous_return_prediction
    ),
    evaluate_model(
        "Multiple Linear Regression",
        y_test,
        linear_predictions
    )
]

# Create the model performance comparison table
performance_results = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

# Display the model performance results
print("\nMODEL PERFORMANCE")
print(
    performance_results.to_string(
        index=False,
        formatters={
            "MSE": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
            "MAE": "{:.4f}".format,
            "R_Squared": "{:.4f}".format,
            "Directional_Accuracy": "{:.2f}%".format
        }
    )
)

# Standardise the training predictors for statistical interpretation
ols_scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    ols_scaler.fit_transform(X_train),
    columns=final_features,
    index=X_train.index
)

# Add an intercept and fit the OLS regression model
X_train_ols = sm.add_constant(X_train_scaled)
ols_model = sm.OLS(y_train, X_train_ols).fit()

# Extract confidence intervals for the regression coefficients
confidence_intervals = ols_model.conf_int()
confidence_intervals.columns = [
    "Confidence_Interval_Lower",
    "Confidence_Interval_Upper"
]

# Combine the regression coefficient and significance results
coefficient_results = pd.DataFrame({
    "Coefficient": ols_model.params,
    "Standard_Error": ols_model.bse,
    "T_Statistic": ols_model.tvalues,
    "P_Value": ols_model.pvalues
}).join(confidence_intervals)

# Display the regression coefficient results
print("\nLINEAR REGRESSION COEFFICIENT RESULTS")
print(coefficient_results.to_string())

# Display the overall regression statistics
print("\nLINEAR REGRESSION MODEL STATISTICS")
print("Training R-squared:", round(ols_model.rsquared, 4))
print("Training adjusted R-squared:", round(ols_model.rsquared_adj, 4))
print("F-statistic:", round(ols_model.fvalue, 4))
print("F-statistic p-value:", ols_model.f_pvalue)
print("Durbin-Watson statistic:", round(sm.stats.stattools.durbin_watson(ols_model.resid), 4))

# Test the residuals for normality using the Jarque-Bera test
jarque_bera_statistic, jarque_bera_p_value, residual_skewness, residual_kurtosis = (
    sm.stats.stattools.jarque_bera(ols_model.resid)
)

# Display the residual normality results
print("\nLINEAR REGRESSION RESIDUAL NORMALITY")
print("Jarque-Bera statistic:", round(jarque_bera_statistic, 4))
print("Jarque-Bera p-value:", jarque_bera_p_value)
print("Residual skewness:", round(residual_skewness, 4))
print("Residual kurtosis:", round(residual_kurtosis, 4))

# Create a table containing test dates and model predictions
prediction_results = pd.DataFrame({
    "date": test["date"].to_numpy(),
    "Actual_Return": y_test.to_numpy(),
    "Historical_Mean_Prediction": historical_mean_prediction,
    "Previous_Return_Prediction": previous_return_prediction,
    "Linear_Regression_Prediction": linear_predictions
})

# Save the diagnostic and modelling outputs
adf_results.to_csv("ADF_Stationarity_Results.csv", index=False)
target_correlations.to_csv("Candidate_Feature_Correlations.csv", index=False)
vif_results.to_csv("Candidate_Feature_VIF_Results.csv", index=False)
reduced_vif.to_csv("Final_Feature_VIF_Results.csv", index=False)
model_data.to_csv("Final_Modelling_Dataset.csv", index=False)
performance_results.to_csv("Initial_Model_Performance.csv", index=False)
coefficient_results.to_csv("Linear_Regression_Coefficients.csv")
prediction_results.to_csv("Initial_Model_Predictions.csv", index=False)

# Create expanding time-series validation folds for model tuning
time_series_cv = TimeSeriesSplit(n_splits=5)

# Define the Random Forest model with reproducible results
random_forest = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

# Define a focused Random Forest tuning grid
random_forest_parameters = {
    "n_estimators": [200, 500],
    "max_depth": [3, 5, None],
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf": [2, 5, 10],
    "max_features": ["sqrt", 1.0]
}

# Tune Random Forest using only chronologically ordered training folds
random_forest_search = GridSearchCV(
    estimator=random_forest,
    param_grid=random_forest_parameters,
    scoring="neg_root_mean_squared_error",
    cv=time_series_cv,
    n_jobs=-1,
    refit=True
)

# Fit the Random Forest tuning process on the training period
random_forest_search.fit(X_train, y_train)

# Select the best Random Forest model
best_random_forest = random_forest_search.best_estimator_

# Generate Random Forest predictions for training and testing data
random_forest_train_predictions = best_random_forest.predict(X_train)
random_forest_test_predictions = best_random_forest.predict(X_test)

# Define the Gradient Boosting model with reproducible results
gradient_boosting = GradientBoostingRegressor(
    random_state=42
)

# Define a focused Gradient Boosting tuning grid
gradient_boosting_parameters = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.03, 0.05],
    "max_depth": [1, 2, 3],
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf": [2, 5, 10],
    "subsample": [0.7, 1.0]
}

# Tune Gradient Boosting using only chronologically ordered training folds
gradient_boosting_search = GridSearchCV(
    estimator=gradient_boosting,
    param_grid=gradient_boosting_parameters,
    scoring="neg_root_mean_squared_error",
    cv=time_series_cv,
    n_jobs=-1,
    refit=True
)

# Fit the Gradient Boosting tuning process on the training period
gradient_boosting_search.fit(X_train, y_train)

# Select the best Gradient Boosting model
best_gradient_boosting = gradient_boosting_search.best_estimator_

# Generate Gradient Boosting predictions for training and testing data
gradient_boosting_train_predictions = best_gradient_boosting.predict(X_train)
gradient_boosting_test_predictions = best_gradient_boosting.predict(X_test)

# Display the selected model parameters and validation scores
print("\nBEST MACHINE LEARNING PARAMETERS")
print("Random Forest:", random_forest_search.best_params_)
print(
    "Random Forest cross-validation RMSE:",
    round(-random_forest_search.best_score_, 4)
)
print("Gradient Boosting:", gradient_boosting_search.best_params_)
print(
    "Gradient Boosting cross-validation RMSE:",
    round(-gradient_boosting_search.best_score_, 4)
)

# Add the machine learning test results to the existing comparison
results.append(
    evaluate_model(
        "Random Forest Regression",
        y_test,
        random_forest_test_predictions
    )
)
results.append(
    evaluate_model(
        "Gradient Boosting Regression",
        y_test,
        gradient_boosting_test_predictions
    )
)

# Create the complete out-of-sample model comparison table
final_performance_results = (
    pd.DataFrame(results)
    .drop_duplicates(subset="Model", keep="last")
    .sort_values("RMSE")
    .reset_index(drop=True)
)

# Display the complete test-period model comparison
print("\nFINAL TEST MODEL PERFORMANCE")
print(
    final_performance_results.to_string(
        index=False,
        formatters={
            "MSE": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
            "MAE": "{:.4f}".format,
            "R_Squared": "{:.4f}".format,
            "Directional_Accuracy": "{:.2f}%".format
        }
    )
)

# Evaluate training performance to identify possible overfitting
training_results = [
    evaluate_model(
        "Multiple Linear Regression",
        y_train,
        linear_regression.predict(X_train)
    ),
    evaluate_model(
        "Random Forest Regression",
        y_train,
        random_forest_train_predictions
    ),
    evaluate_model(
        "Gradient Boosting Regression",
        y_train,
        gradient_boosting_train_predictions
    )
]

# Create the training performance table
training_performance_results = (
    pd.DataFrame(training_results)
    .sort_values("RMSE")
    .reset_index(drop=True)
)

# Display training performance for comparison with unseen test results
print("\nTRAINING MODEL PERFORMANCE")
print(
    training_performance_results.to_string(
        index=False,
        formatters={
            "MSE": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
            "MAE": "{:.4f}".format,
            "R_Squared": "{:.4f}".format,
            "Directional_Accuracy": "{:.2f}%".format
        }
    )
)

# Combine training and testing metrics to assess model generalisation
generalisation_results = (
    training_performance_results[
        ["Model", "RMSE", "MAE", "R_Squared", "Directional_Accuracy"]
    ]
    .rename(
        columns={
            "RMSE": "Training_RMSE",
            "MAE": "Training_MAE",
            "R_Squared": "Training_R_Squared",
            "Directional_Accuracy": "Training_Directional_Accuracy"
        }
    )
    .merge(
        final_performance_results[
            ["Model", "RMSE", "MAE", "R_Squared", "Directional_Accuracy"]
        ].rename(
            columns={
                "RMSE": "Testing_RMSE",
                "MAE": "Testing_MAE",
                "R_Squared": "Testing_R_Squared",
                "Directional_Accuracy": "Testing_Directional_Accuracy"
            }
        ),
        on="Model",
        how="left"
    )
)

# Calculate the difference between training and testing errors
generalisation_results["RMSE_Gap"] = (
    generalisation_results["Testing_RMSE"]
    - generalisation_results["Training_RMSE"]
)

# Display the training and testing comparison
print("\nMODEL GENERALISATION COMPARISON")
print(
    generalisation_results.to_string(
        index=False,
        formatters={
            "Training_RMSE": "{:.4f}".format,
            "Training_MAE": "{:.4f}".format,
            "Training_R_Squared": "{:.4f}".format,
            "Training_Directional_Accuracy": "{:.2f}%".format,
            "Testing_RMSE": "{:.4f}".format,
            "Testing_MAE": "{:.4f}".format,
            "Testing_R_Squared": "{:.4f}".format,
            "Testing_Directional_Accuracy": "{:.2f}%".format,
            "RMSE_Gap": "{:.4f}".format
        }
    )
)

# Extract the built-in Random Forest feature importance scores
random_forest_importance = pd.DataFrame({
    "Feature": final_features,
    "Random_Forest_Importance": best_random_forest.feature_importances_
})

# Extract the built-in Gradient Boosting feature importance scores
gradient_boosting_importance = pd.DataFrame({
    "Feature": final_features,
    "Gradient_Boosting_Importance": best_gradient_boosting.feature_importances_
})

# Calculate Random Forest permutation importance on unseen test data
random_forest_permutation = permutation_importance(
    best_random_forest,
    X_test,
    y_test,
    scoring="neg_root_mean_squared_error",
    n_repeats=30,
    random_state=42,
    n_jobs=-1
)

# Calculate Gradient Boosting permutation importance on unseen test data
gradient_boosting_permutation = permutation_importance(
    best_gradient_boosting,
    X_test,
    y_test,
    scoring="neg_root_mean_squared_error",
    n_repeats=30,
    random_state=42,
    n_jobs=-1
)

# Store the test-set permutation importance results
permutation_results = pd.DataFrame({
    "Feature": final_features,
    "Random_Forest_Permutation_Mean": random_forest_permutation.importances_mean,
    "Random_Forest_Permutation_Std": random_forest_permutation.importances_std,
    "Gradient_Boosting_Permutation_Mean": gradient_boosting_permutation.importances_mean,
    "Gradient_Boosting_Permutation_Std": gradient_boosting_permutation.importances_std
})

# Combine built-in and test-set feature importance measures
feature_importance_results = (
    random_forest_importance
    .merge(
        gradient_boosting_importance,
        on="Feature",
        how="inner"
    )
    .merge(
        permutation_results,
        on="Feature",
        how="inner"
    )
)

# Calculate average built-in importance for descriptive ranking
feature_importance_results["Average_Tree_Importance"] = (
    feature_importance_results[
        [
            "Random_Forest_Importance",
            "Gradient_Boosting_Importance"
        ]
    ].mean(axis=1)
)

# Sort features using average built-in importance
feature_importance_results = (
    feature_importance_results
    .sort_values(
        "Average_Tree_Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

# Display the machine learning feature importance results
print("\nMACHINE LEARNING FEATURE IMPORTANCE")
print(
    feature_importance_results.to_string(
        index=False,
        formatters={
            "Random_Forest_Importance": "{:.4f}".format,
            "Gradient_Boosting_Importance": "{:.4f}".format,
            "Random_Forest_Permutation_Mean": "{:.4f}".format,
            "Random_Forest_Permutation_Std": "{:.4f}".format,
            "Gradient_Boosting_Permutation_Mean": "{:.4f}".format,
            "Gradient_Boosting_Permutation_Std": "{:.4f}".format,
            "Average_Tree_Importance": "{:.4f}".format
        }
    )
)

# Add machine learning predictions to the existing prediction table
prediction_results["Random_Forest_Prediction"] = random_forest_test_predictions
prediction_results["Gradient_Boosting_Prediction"] = gradient_boosting_test_predictions

# Calculate residuals for each fitted forecasting model
prediction_results["Linear_Regression_Residual"] = (
    prediction_results["Actual_Return"]
    - prediction_results["Linear_Regression_Prediction"]
)
prediction_results["Random_Forest_Residual"] = (
    prediction_results["Actual_Return"]
    - prediction_results["Random_Forest_Prediction"]
)
prediction_results["Gradient_Boosting_Residual"] = (
    prediction_results["Actual_Return"]
    - prediction_results["Gradient_Boosting_Prediction"]
)

# Plot actual returns against all model predictions
figure, axis = plt.subplots(figsize=(13, 6))
axis.plot(
    prediction_results["date"],
    prediction_results["Actual_Return"],
    label="Actual Return",
    color="black",
    linewidth=2
)
axis.plot(
    prediction_results["date"],
    prediction_results["Linear_Regression_Prediction"],
    label="Multiple Linear Regression",
    linewidth=1.2
)
axis.plot(
    prediction_results["date"],
    prediction_results["Random_Forest_Prediction"],
    label="Random Forest",
    linewidth=1.2
)
axis.plot(
    prediction_results["date"],
    prediction_results["Gradient_Boosting_Prediction"],
    label="Gradient Boosting",
    linewidth=1.2
)
axis.axhline(0, color="grey", linestyle="--", linewidth=0.8)
axis.set_title("Actual and Predicted Next-Month S&P 500 Returns")
axis.set_xlabel("Forecast Origin Date")
axis.set_ylabel("Next-Month Return (%)")
axis.legend()
figure.tight_layout()
figure.savefig("Actual_vs_Predicted_Returns.png", dpi=300)
plt.close(figure)

# Plot the performance metrics for all models
performance_plot = final_performance_results.set_index("Model")
figure, axes = plt.subplots(2, 1, figsize=(12, 10))
performance_plot[["RMSE", "MAE"]].plot(
    kind="bar",
    ax=axes[0]
)
axes[0].set_title("Out-of-Sample Prediction Error")
axes[0].set_xlabel("Model")
axes[0].set_ylabel("Error in Percentage Points")
axes[0].tick_params(axis="x", rotation=15)
performance_plot[["Directional_Accuracy"]].plot(
    kind="bar",
    ax=axes[1],
    legend=False
)
axes[1].axhline(
    test_majority_direction,
    color="black",
    linestyle="--",
    label="Majority-Direction Benchmark"
)
axes[1].set_title("Out-of-Sample Directional Accuracy")
axes[1].set_xlabel("Model")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_ylim(0, 100)
axes[1].tick_params(axis="x", rotation=15)
axes[1].legend()
figure.tight_layout()
figure.savefig("Final_Model_Performance.png", dpi=300)
plt.close(figure)

# Plot the built-in tree model feature importance scores
feature_plot = (
    feature_importance_results
    .set_index("Feature")[
        [
            "Random_Forest_Importance",
            "Gradient_Boosting_Importance"
        ]
    ]
)
figure, axis = plt.subplots(figsize=(11, 6))
feature_plot.plot(
    kind="barh",
    ax=axis
)
axis.set_title("Machine Learning Feature Importance")
axis.set_xlabel("Importance Score")
axis.set_ylabel("Predictor")
figure.tight_layout()
figure.savefig("Machine_Learning_Feature_Importance.png", dpi=300)
plt.close(figure)

# Plot test-set permutation importance for both machine learning models
permutation_plot = (
    feature_importance_results
    .set_index("Feature")[
        [
            "Random_Forest_Permutation_Mean",
            "Gradient_Boosting_Permutation_Mean"
        ]
    ]
)
figure, axis = plt.subplots(figsize=(11, 6))
permutation_plot.plot(
    kind="barh",
    ax=axis
)
axis.axvline(0, color="black", linewidth=0.8)
axis.set_title("Test-Set Permutation Importance")
axis.set_xlabel("Increase in Predictive Value")
axis.set_ylabel("Predictor")
figure.tight_layout()
figure.savefig("Permutation_Feature_Importance.png", dpi=300)
plt.close(figure)

# Plot residuals across the unseen testing period
figure, axis = plt.subplots(figsize=(13, 6))
axis.plot(
    prediction_results["date"],
    prediction_results["Linear_Regression_Residual"],
    label="Multiple Linear Regression",
    linewidth=1.2
)
axis.plot(
    prediction_results["date"],
    prediction_results["Random_Forest_Residual"],
    label="Random Forest",
    linewidth=1.2
)
axis.plot(
    prediction_results["date"],
    prediction_results["Gradient_Boosting_Residual"],
    label="Gradient Boosting",
    linewidth=1.2
)
axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
axis.set_title("Out-of-Sample Model Residuals")
axis.set_xlabel("Forecast Origin Date")
axis.set_ylabel("Residual Return (%)")
axis.legend()
figure.tight_layout()
figure.savefig("Model_Residuals.png", dpi=300)
plt.close(figure)

# Store the best parameters and validation scores
tuning_results = pd.DataFrame([
    {
        "Model": "Random Forest Regression",
        "Best_Parameters": str(random_forest_search.best_params_),
        "Cross_Validation_RMSE": -random_forest_search.best_score_
    },
    {
        "Model": "Gradient Boosting Regression",
        "Best_Parameters": str(gradient_boosting_search.best_params_),
        "Cross_Validation_RMSE": -gradient_boosting_search.best_score_
    }
])

# Save the final modelling results
final_performance_results.to_csv(
    "Final_Model_Performance.csv",
    index=False
)
training_performance_results.to_csv(
    "Training_Model_Performance.csv",
    index=False
)
generalisation_results.to_csv(
    "Model_Generalisation_Comparison.csv",
    index=False
)
feature_importance_results.to_csv(
    "Machine_Learning_Feature_Importance.csv",
    index=False
)
tuning_results.to_csv(
    "Machine_Learning_Tuning_Results.csv",
    index=False
)
prediction_results.to_csv(
    "Final_Model_Predictions.csv",
    index=False
)

# Create a correlation matrix visual for the final modelling variables
import matplotlib.pyplot as plt

correlation_matrix = (
    reduced_data[
        ["Next_Month_Return"] + selected_features
    ]
    .corr()
)

figure, axis = plt.subplots(figsize=(9, 7))

image = axis.imshow(
    correlation_matrix.values,
    cmap="coolwarm",
    aspect="auto",
    vmin=-1,
    vmax=1
)

figure.colorbar(
    image,
    ax=axis,
    label="Correlation"
)

axis.set_xticks(
    range(len(correlation_matrix.columns))
)

axis.set_xticklabels(
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)

axis.set_yticks(
    range(len(correlation_matrix.index))
)

axis.set_yticklabels(
    correlation_matrix.index
)

for row in range(len(correlation_matrix.index)):
    for column in range(len(correlation_matrix.columns)):
        value = correlation_matrix.iloc[row, column]
        text_colour = (
            "white"
            if abs(value) > 0.50
            else "black"
        )

        axis.text(
            column,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            color=text_colour,
            fontsize=8
        )

axis.set_title(
    "Correlation Matrix of Final Modelling Variables"
)

figure.tight_layout()

figure.savefig(
    "Figure_5_1_Correlation_Matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(figure)

print(
    "\nSaved: Figure_5_1_Correlation_Matrix.png"
)