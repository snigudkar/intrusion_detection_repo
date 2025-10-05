# ==================== hybrid_predict.py ====================
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

MODEL_DIR = "models"
DATA_PATH = "data/Test_data.csv"

print("Loading trained model and Apriori rules...")
with open(f"{MODEL_DIR}/rf_model.pkl", "rb") as f:
    rf = pickle.load(f)

rules_anomaly = pd.read_csv(f"{MODEL_DIR}/rules_anomaly.csv")

print("Loading test data...")
df_test = pd.read_csv(DATA_PATH)
print("Test Shape:", df_test.shape)

# ===== Prepare test data for Random Forest =====
X_test = pd.get_dummies(df_test, columns=['protocol_type','flag','service'])
model_features = rf.feature_names_in_
X_test = X_test.reindex(columns=model_features, fill_value=0)

# RF predictions
rf_preds = rf.predict(X_test)

# ===== Apriori Check =====
scaler = StandardScaler()
num_cols = X_test.select_dtypes(include=['int64','float64']).columns.tolist()
X_test[num_cols] = scaler.fit_transform(X_test[num_cols])
X_test[num_cols] = X_test[num_cols].applymap(lambda x: 1 if x > 0 else 0)

def apriori_flag(row, rules_anomaly):
    for _, rule in rules_anomaly.iterrows():
        antecedents = eval(rule['antecedents']) if isinstance(rule['antecedents'], str) else rule['antecedents']
        if all(row.get(item, 0) == 1 for item in antecedents):
            return 1
    return 0

apriori_preds = X_test.apply(lambda row: apriori_flag(row, rules_anomaly), axis=1)

hybrid_preds = ((rf_preds == 1) | (apriori_preds == 1)).astype(int)

print("\nHybrid detection complete!")
print("Detected anomalies:")
print(f" Random Forest: {sum(rf_preds)}")
print(f"Apriori Rules: {sum(apriori_preds)}")
print(f" Hybrid Combined: {sum(hybrid_preds)}")

# Save hybrid output
df_test['Prediction'] = hybrid_preds
df_test.to_csv("data/Test_with_predictions.csv", index=False)
print("\nSaved predictions to data/Test_with_predictions.csv")
