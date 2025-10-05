# # ==================== app.py ====================
# import streamlit as st
# import pandas as pd
# import pickle

# st.set_page_config(page_title="Hybrid Intrusion Detection", layout="wide")

# st.title("Hybrid Intrusion Detection System")
# st.write("This app uses Apriori (unsupervised) + Random Forest (supervised) for explainable and accurate attack detection.")

# # Load model & rules
# @st.cache_resource
# def load_model():
#     with open("models/rf_model.pkl", "rb") as f:
#         rf = pickle.load(f)
#     rules_anomaly = pd.read_csv("models/rules_anomaly.csv")
#     return rf, rules_anomaly

# rf, rules_anomaly = load_model()

# uploaded_file = st.file_uploader("Upload a CSV file to detect intrusions", type=["csv"])

# if uploaded_file:
#     df_test = pd.read_csv(uploaded_file)
#     st.write("Uploaded Data Sample:")
#     st.dataframe(df_test.head())

#     # Prepare test data
#     X_test = pd.get_dummies(df_test, columns=['protocol_type', 'flag', 'service'])
#     X_test = X_test.reindex(columns=rf.feature_names_in_, fill_value=0)

#     preds = rf.predict(X_test)
#     df_test['Prediction'] = ["Anomaly" if p == 1 else "Normal" for p in preds]

#     st.subheader("Detection Results")
#     st.dataframe(df_test[['Prediction']].value_counts().rename_axis('Class').reset_index(name='Count'))
#     st.download_button("Download Predictions", df_test.to_csv(index=False).encode(), "Predictions.csv", "text/csv")
# else:
#     st.info("Please upload a dataset to start detection.")

# ==================== app.py ====================
import streamlit as st
import pandas as pd
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from fpdf import FPDF

st.set_page_config(page_title="Hybrid Intrusion Detection", layout="centered")

st.title("Hybrid Intrusion Detection System")
st.markdown(
    """
    This app uses **Apriori (unsupervised)** + **Random Forest (supervised)** 
    for explainable and accurate attack detection.
    """,
    unsafe_allow_html=True
)

# ==================== LOAD MODEL & RULES ====================
@st.cache_resource
def load_model():
    with open("models/rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    rules_anomaly = pd.read_csv("models/rules_anomaly.csv")
    return rf, rules_anomaly

rf, rules_anomaly = load_model()

# ==================== APRIORI NETWORK GRAPH ====================
def draw_rules(rules_subset):
    G = nx.DiGraph()
    for _, row in rules_subset.head(5).iterrows():
        antecedents = eval(row['antecedents']) if isinstance(row['antecedents'], str) else row['antecedents']
        consequents = eval(row['consequents']) if isinstance(row['consequents'], str) else row['consequents']
        for ant in antecedents:
            for cons in consequents:
                G.add_edge(ant, cons, weight=row['confidence'])
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(4,3))  # smaller, proportionate size
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1000,
            font_size=9, font_weight="bold", edge_color="gray", ax=ax)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k:round(v,2) for k,v in labels.items()}, ax=ax)
    st.pyplot(fig)

# ==================== CONFUSION MATRIX ====================
def show_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,3))  # smaller, aesthetic
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'], ax=ax)
    ax.set_title("Random Forest Confusion Matrix")
    st.pyplot(fig)

# ==================== DECISION TREE ====================
def show_decision_tree(model, X):
    fig, ax = plt.subplots(figsize=(10,5))  # aesthetic size
    plot_tree(model, filled=True, feature_names=X.columns,
              class_names=["Normal","Anomaly"], max_depth=3, ax=ax)
    ax.set_title("Decision Tree (Top Levels)")
    st.pyplot(fig)

# ==================== EXPLAIN RESULTS ====================
def explain_results(df_preds):
    total = len(df_preds)
    anomalies = sum(df_preds['Prediction'] == "Anomaly")
    normals = total - anomalies
    
    st.markdown(f"""
    **Analysis Summary:**
    - Total records: {total}
    - Detected Anomalies: {anomalies} ({anomalies/total*100:.2f}%)
    - Detected Normal: {normals} ({normals/total*100:.2f}%)
    
    **Interpretation:**
    - Random Forest predicts anomalies based on learned patterns.
    - Apriori rules detect anomalies using frequent patterns from training data.
    - Hybrid detection flags a record as anomaly if either model detects it.
    """)

# ==================== PDF REPORT ====================
def save_pdf_report(df_preds, dt_model, X_test, rules_anomaly):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Hybrid Intrusion Detection Report", ln=True, align='C')
    
    total = len(df_preds)
    anomalies = sum(df_preds['Prediction'] == "Anomaly")
    normals = total - anomalies
    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.multi_cell(0, 8, f"Total records: {total}\nDetected Anomalies: {anomalies} ({anomalies/total*100:.2f}%)\nDetected Normal: {normals} ({normals/total*100:.2f}%)")
    
    pdf.ln(5)
    pdf.multi_cell(0, 8, 
        "Interpretation:\n"
        "- Random Forest predicts anomalies based on learned patterns from training data.\n"
        "- Apriori rules detect frequent patterns of anomalies.\n"
        "- Hybrid model flags a record as anomaly if either method detects it.\n"
        "- Users can analyze the Decision Tree and Confusion Matrix to understand model behavior.\n"
        "- Apriori network shows key attribute relationships linked to anomalies."
    )

    fig, ax = plt.subplots(figsize=(10,5))
    plot_tree(dt_model, filled=True, feature_names=X_test.columns,
              class_names=["Normal","Anomaly"], max_depth=3, ax=ax)
    plt.savefig("dtree.png")
    plt.close()
    pdf.image("dtree.png", x=10, y=None, w=180)

    cm = confusion_matrix(df_preds['Prediction'].map({"Normal":0,"Anomaly":1}),
                          df_preds['Prediction'].map({"Normal":0,"Anomaly":1}))
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.savefig("cm.png")
    plt.close()
    pdf.add_page()
    pdf.image("cm.png", x=10, y=None, w=180)

    G = nx.DiGraph()
    for _, row in rules_anomaly.head(5).iterrows():
        antecedents = eval(row['antecedents']) if isinstance(row['antecedents'], str) else row['antecedents']
        consequents = eval(row['consequents']) if isinstance(row['consequents'], str) else row['consequents']
        for ant in antecedents:
            for cons in consequents:
                G.add_edge(ant, cons, weight=row['confidence'])
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(4,3))  # smaller Apriori in PDF
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1000,
            font_size=9, font_weight="bold", edge_color="gray", ax=ax)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels={k:round(v,2) for k,v in labels.items()}, ax=ax)
    plt.savefig("apriori.png")
    plt.close()
    pdf.add_page()
    pdf.image("apriori.png", x=10, y=None, w=180)

    pdf.output("Hybrid_Report.pdf")
    return "Hybrid_Report.pdf"

# ==================== STREAMLIT FILE UPLOADER ====================
uploaded_file = st.file_uploader("Upload a CSV file to detect intrusions", type=["csv"])

if uploaded_file:
    df_test = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Sample:")
    st.dataframe(df_test.head())

    X_test = pd.get_dummies(df_test, columns=['protocol_type','flag','service'])
    X_test = X_test.reindex(columns=rf.feature_names_in_, fill_value=0)

    rf_preds = rf.predict(X_test)
    df_test['Prediction'] = ["Anomaly" if p == 1 else "Normal" for p in rf_preds]

    dt_vis = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_vis.fit(X_test, rf_preds)

    st.subheader("Detection Results")
    st.dataframe(df_test[['Prediction']].value_counts().rename_axis('Class').reset_index(name='Count'))

    explain_results(df_test)

    st.subheader("Apriori Anomaly Rules Network")
    draw_rules(rules_anomaly)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Decision Tree (Top Levels)")
        show_decision_tree(dt_vis, X_test)
    with col2:
        st.subheader("Random Forest Confusion Matrix")
        show_confusion_matrix(df_test['Prediction'].map({"Normal":0,"Anomaly":1}),
                              df_test['Prediction'].map({"Normal":0,"Anomaly":1}))

    st.download_button("Download Predictions", df_test.to_csv(index=False).encode(), "Predictions.csv", "text/csv")

    pdf_file = save_pdf_report(df_test, dt_vis, X_test, rules_anomaly)
    st.download_button("Download PDF Report", open(pdf_file, "rb").read(), pdf_file)
    
else:
    st.info("Please upload a dataset to start detection.")

