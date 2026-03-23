
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.title("SpendWise AI Dashboard")

df = pd.read_csv("spendwise_dataset.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# Encoding
df_encoded = df.copy()
le = LabelEncoder()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = le.fit_transform(df_encoded[col])

# Classification
X = df_encoded.drop("App_Interest", axis=1)
y = df_encoded["App_Interest"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# Clustering
kmeans = KMeans(n_clusters=4, n_init=10)
df_encoded['Cluster'] = kmeans.fit_predict(X)

st.subheader("Cluster Distribution")
fig = px.histogram(df_encoded, x="Cluster")
st.plotly_chart(fig)

# Association Rules
basket = pd.get_dummies(df[['Spending_Combination']])
freq = apriori(basket, min_support=0.1, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1)

st.subheader("Association Rules")
st.write(rules[['antecedents','consequents','support','confidence','lift']].head())
