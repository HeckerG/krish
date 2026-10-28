# E-commerce Customer Behavior Prediction

## Title page
- **Project title:** E-commerce Customer Behavior Prediction Using Correlation and Regression
- **Student name / ID:** [Fill in]
- **Course / subject:** [Fill in]
- **Instructor:** [Fill in]
- **Institution:** [Fill in]
- **Submission date:** [Fill in]

## 1. Abstract
This project analyzes the relationship between customer engagement and yearly spending using an e-commerce customer dataset. Descriptive statistics and Pearson correlation are used to explore the data. A Linear Regression model is trained to predict yearly spending, and its performance is compared with a mean baseline and a Random Forest model using MAE, RMSE, and R². An interactive Streamlit dashboard is developed to display the analysis and estimate spending for hypothetical customer profiles. Replace this draft with a concise summary of your actual findings after running the project.

## 2. Introduction
E-commerce businesses collect information about how customers interact with websites and mobile applications. Understanding whether engagement measures are associated with spending can support better customer experience decisions. This project explores those relationships and evaluates whether the selected variables can predict yearly spending.

## 3. Objectives
1. Inspect and clean the customer dataset.
2. Calculate mean, median, standard deviation, and other descriptive statistics.
3. Measure Pearson correlation between engagement features and yearly spending.
4. Train and evaluate regression models.
5. Build an interactive dashboard for analysis and prediction.
6. Discuss findings, limitations, and possible improvements.

## 4. Dataset
- **Dataset name:** Ecommerce Customers
- **Source URL:** [Insert the exact URL used]
- **Number of rows after cleaning:** [Fill in from dashboard]
- **Input features:** Avg. Session Length, Time on App, Time on Website, Length of Membership
- **Target:** Yearly Amount Spent (USD)
- **Data preparation:** Required numeric columns were selected, values were converted to numeric types, incomplete rows and duplicate rows were removed, and identifying columns such as email/address were excluded.

The dataset is used for educational analysis. Describe the dataset source and any license/attribution conditions accurately.

## 5. Methodology

### 5.1 Descriptive statistics
Mean, median, standard deviation, minimum, maximum, and quartiles were calculated for each numeric variable.

### 5.2 Correlation analysis
Pearson's correlation coefficient was calculated to quantify the direction and strength of linear association. Correlation does not establish causation.

### 5.3 Regression
The dataset was split into training (80%) and test (20%) sets with `random_state=42`. The models compared were:
- Mean baseline
- Linear Regression
- Random Forest Regression

### 5.4 Evaluation metrics
- **MAE:** Average absolute prediction error, in USD. Lower is better.
- **RMSE:** Square root of mean squared error, in USD; it penalizes larger errors more strongly. Lower is better.
- **R²:** Fraction of target variation explained relative to a mean baseline. It can be negative; closer to 1 is generally better.

## 6. Results
Insert your actual results after running `app.py` or `analysis.py`.

### 6.1 Descriptive statistics
[Insert summary table and explain notable patterns.]

### 6.2 Correlation analysis
[Insert correlation heatmap. Report the correlation coefficients with Yearly Amount Spent and explain the strongest positive/negative associations.]

### 6.3 Model comparison
[Insert MAE, RMSE, and R² table from `outputs/model_comparison.csv` or the dashboard.]

| Model | MAE (USD) | RMSE (USD) | R² |
|---|---:|---:|---:|
| Mean baseline | [ ] | [ ] | [ ] |
| Linear Regression | [ ] | [ ] | [ ] |
| Random Forest | [ ] | [ ] | [ ] |

### 6.4 Interpretation
[Which model performed best on the test set? Did it improve on the baseline? What does the prediction plot show?]

## 7. Dashboard
The Streamlit dashboard provides:
- Dataset overview and quality checks
- Descriptive statistics and spending distribution
- Scatter plots and a Pearson correlation heatmap
- Model comparison and error visualizations
- Interactive spending estimates for hypothetical customer profiles

[Insert screenshots of the dashboard.]

## 8. Conclusion
[Summarize the strongest observed associations, the best test-set model and its metrics, and what the results may mean for e-commerce analysis. Do not claim that correlation proves a feature causes higher spending.]

## 9. Limitations
1. The dataset is relatively small and may not represent every retailer or customer population.
2. The available features do not capture all influences on spending.
3. A random hold-out split provides one estimate of performance; repeated cross-validation could give a more stable estimate.
4. Model predictions are estimates, not guaranteed customer outcomes.
5. Observational correlations should not be interpreted as causal effects.

## 10. Future scope
- Add cross-validation and prediction intervals.
- Test regularized regression and gradient boosting.
- Explore customer segmentation.
- Use a larger, more recent dataset with clear provenance.
- Evaluate the model on future-period data if dates become available.

## References
1. Dataset source: [Insert the exact source and access date].
2. Scikit-learn documentation: https://scikit-learn.org/stable/
3. Streamlit documentation: https://docs.streamlit.io/

## Appendix
Include important code, dependency versions, and instructions to run the application.
