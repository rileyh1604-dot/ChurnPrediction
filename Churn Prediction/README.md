# Telco Customer Churn Prediction

Predicting which telecom customers are likely to leave, so a retention team can contact them before they do.

**🔗 Live dashboard:** _add your Streamlit link here_

## The problem

About 26.5% of the 7,043 customers in the [IBM Telco Customer Churn](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113) dataset churned. Missing a customer who is about to leave usually costs more than contacting one who would have stayed, so I tuned the model for **recall first**: catch at least 70% of churners, then make precision as high as possible.

## Results (held-out test set, 1,409 customers)

| Metric | Score |
|---|---|
| Recall | 0.695 |
| Precision | 0.571 |
| ROC-AUC | 0.850 |

At the chosen threshold the model catches about 7 in 10 churners, and about 57% of the customers it flags really do churn. Random guessing would be right about 26.5% of the time.

## Approach

1. **Cleaning:** converted `Total Charges` to numbers (blank for brand-new customers, so set to 0).
2. **Avoiding leakage:** removed `Churn Score`, `Churn Reason` and `CLTV`, which are only known after the fact or are derived from churn, along with IDs and location columns.
3. **Preprocessing:** one-hot encoded the categorical columns and standardised tenure and charges.
4. **Model:** gradient boosted trees (scikit-learn) trained on SMOTE-oversampled data to handle class imbalance.
5. **Threshold tuning:** picked the decision threshold on a separate validation split (highest precision with recall ≥ 0.70) instead of using the default 0.5. The test set was only used for the final scores above.

## What didn't help (and why that's useful)

I compared alternatives with repeated 5-fold cross-validation on the training data:

| Approach | ROC-AUC | Precision @ recall ≥ 0.70 |
|---|---|---|
| Gradient boosting + SMOTE (final) | 0.861 | 0.606 |
| Logistic regression + engineered features | 0.860 | 0.608 |
| Logistic regression + boosting average | 0.863 | 0.608 |
| HistGradientBoosting / random forest | 0.859–0.861 | 0.589–0.594 |
| Random search over boosting parameters | ≤ 0.863 | ≤ 0.607 |

All of these land within about 0.003 of each other, which is the same size as the run-to-run variation. Engineered features (add-on counts, spend changes, tenure flags) and model combinations didn't add new information. **The limit is the data, not the model.** Getting meaningfully better would need behavioural data such as support tickets, usage trends or complaint history.

## Dashboard

Built with Streamlit:

- **Predict a customer:** enter a customer's details and get their churn probability and a risk flag.
- **Batch scoring:** upload a customer file and download it ranked by churn risk.
- **Model performance:** move the threshold and watch precision, recall and the confusion matrix change.
- **Insights:** feature importances and churn rates by contract, payment method, tenure and more.

## Run it locally

```bash
pip install -r requirements.txt
python train_model.py      # retrains and saves churn_model.joblib
streamlit run app.py
```

## Project structure

| File | Purpose |
|---|---|
| `churn_analysis.ipynb` | Exploration, cleaning and model comparison |
| `churn_pipeline.py` | Shared cleaning and preprocessing used by both training and the app |
| `train_model.py` | Reproducible training script that saves the model bundle |
| `app.py` | Streamlit dashboard |
| `churn_model.joblib` | Trained model, scaler, threshold and test results |
