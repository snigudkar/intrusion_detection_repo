# ==================== train_model.py ====================
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from mlxtend.frequent_patterns import apriori, association_rules
import pickle
import os

# ==================== PATHS ====================
DATA_PATH = "data/Train_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================== LOAD DATA ====================
print("Loading training data...")
df_train = pd.read_csv(DATA_PATH)
print("Data Loaded. Shape:", df_train.shape)
print("Class distribution:\n", df_train['class'].value_counts())

# ==================== APRIORI RULE MINING ====================
print("\n Mining Apriori rules...")

df_ap = pd.get_dummies(df_train, columns=['protocol_type','flag','service','class'])
num_cols = df_train.select_dtypes(include=['int64','float64']).columns.tolist()

scaler = StandardScaler()
df_ap[num_cols] = scaler.fit_transform(df_ap[num_cols])
df_ap[num_cols] = df_ap[num_cols].applymap(lambda x: 1 if x > 0 else 0)

df_ap = df_ap.loc[:, (df_ap != 0).any(axis=0)]

frequent_itemsets = apriori(df_ap, min_support=0.02, use_colnames=True, max_len=2)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
rules = rules[(rules['lift'] > 1) & (rules['conviction'] > 1)]

rules_anomaly = rules[rules['consequents'].apply(lambda x: 'class_anomaly' in x)]
rules_normal = rules[rules['consequents'].apply(lambda x: 'class_normal' in x)]

rules_anomaly.to_csv(f"{MODEL_DIR}/rules_anomaly.csv", index=False)
rules_normal.to_csv(f"{MODEL_DIR}/rules_normal.csv", index=False)
print(" Apriori rules saved to 'models/' folder.")

# ==================== RANDOM FOREST MODEL ====================
print("\nTraining Random Forest model...")

X = df_train.drop(columns=['class'])
y = df_train['class'].apply(lambda x: 1 if x.lower() == "anomaly" else 0)
X = pd.get_dummies(X, columns=['protocol_type','flag','service'])

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

with open(f"{MODEL_DIR}/rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print(" Model saved as 'rf_model.pkl'")
print("\n Training complete!")
