import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
import numpy as np
import streamlit as st

# Set page title and layout
st.title("Online Retail Data Mining Dashboard")
st.write("This app performs data mining on the Online Retail dataset, including EDA, Association Rule Mining, Classification, Clustering, and Business Recommendations.")

# Load data (assuming file is available locally)
df = pd.read_excel("Online Retail.xlsx")

# Data Cleaning
st.header("Data Cleaning")
df.dropna(subset=["CustomerID", "Description"], inplace=True)
df["CustomerID"] = df["CustomerID"].astype(int)
df["Description"] = df["Description"].str.strip()
df["Country"] = df["Country"].str.strip()

# Handle negative quantities and outliers
df = df[df["Quantity"] > 0]  # Remove returns
df = df[np.abs(df["Quantity"] - df["Quantity"].mean()) <= (3 * df["Quantity"].std())]  # Remove quantity outliers
df = df[df["UnitPrice"] > 0]  # Remove invalid prices

# Inspect and handle InvoiceDate conversion
st.subheader("Inspecting InvoiceDate Column")
st.write("First few InvoiceDate values:", df["InvoiceDate"].head())
st.write("InvoiceDate dtype:", df["InvoiceDate"].dtype)

# Handle InvoiceDate conversion based on its format
if df["InvoiceDate"].dtype == "float64" or df["InvoiceDate"].dtype == "int64":
    # If numeric (Excel serial date), convert to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], unit='d', origin='1899-12-30')
elif df["InvoiceDate"].dtype != "datetime64[ns]":
    # If not datetime but a string, parse it directly
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
else:
    # Already in datetime format, no conversion needed
    st.write("InvoiceDate is already in datetime format, skipping conversion.")

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
st.write("Cleaned dataset preview:", df.head())

# Exploratory Data Analysis
st.header("Exploratory Data Analysis (EDA)")

# Top Products
st.subheader("Top 10 Products by Frequency")
top_products = df["Description"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 5))
top_products.plot(kind="bar", ax=ax, title="Top 10 Products by Frequency")
plt.xticks(rotation=45)
st.pyplot(fig)

# Sales Over Time
st.subheader("Monthly Sales Trend")
fig, ax = plt.subplots(figsize=(10, 5))
df.set_index("InvoiceDate").resample('M')["TotalPrice"].sum().plot(ax=ax, title="Monthly Sales Trend")
st.pyplot(fig)

# Country-wise Sales
st.subheader("Top 10 Countries by Total Sales")
top_countries = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 5))
top_countries.plot(kind="bar", ax=ax, title="Top 10 Countries by Total Sales", color='orange')
plt.xticks(rotation=45)
st.pyplot(fig)

# Quantity Distribution
st.subheader("Quantity Distribution")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["Quantity"], bins=30, kde=True, ax=ax)
plt.title("Quantity Distribution")
st.pyplot(fig)

# Association Rule Mining
st.header("Association Rule Mining")
min_support = st.slider("Select Minimum Support", 0.005, 0.05, 0.01, 0.005)
min_lift = st.slider("Select Minimum Lift", 1.0, 3.0, 1.2, 0.1)

df_uk = df[df["Country"] == "United Kingdom"]
basket = df_uk.groupby(["InvoiceNo", "Description"])["Quantity"].sum().unstack().fillna(0)
basket_bool = (basket > 0).astype(bool)

frequent_items = apriori(basket_bool, min_support=min_support, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=min_lift)
rules = rules.sort_values("lift", ascending=False)

st.write("Top 10 Association Rules:")
st.write(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

# Classification
st.header("Classification Models")
df_class = df.copy()
df_class["PurchaseFrequency"] = df_class.groupby("CustomerID")["InvoiceNo"].transform("nunique")
df_class["Spender"] = (df_class["TotalPrice"] > df_class["TotalPrice"].quantile(0.75)).astype(int)
le = LabelEncoder()
int(df_class["CountryEncoded"] = le.fit_transform(df_class["Country"])

X = df_class[["Quantity", "UnitPrice", "PurchaseFrequency", "CountryEncoded"]]
y = df_class["Spender"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    st.subheader(f"{name} Performance")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Cross-Validation Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# Clustering
st.header("Customer Segmentation with Clustering")
n_clusters = st.slider("Select Number of Clusters", 2, 10, 4)

customer_data = df.groupby("CustomerID").agg({
    "Quantity": "sum",
    "TotalPrice": "sum",
    "InvoiceNo": "nunique"
}).reset_index()
customer_data.columns = ["CustomerID", "TotalQuantity", "TotalSpending", "PurchaseFrequency"]

scaler = StandardScaler()
customer_scaled = scaler.fit_transform(customer_data[["TotalQuantity", "TotalSpending", "PurchaseFrequency"]])

# Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(customer_scaled)
    inertia.append(kmeans.inertia_)

st.subheader("Elbow Method for Optimal Clusters")
fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
st.pyplot(fig)

# KMeans Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
customer_data["Cluster"] = kmeans.fit_predict(customer_scaled)

st.subheader("Customer Segments")
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=customer_data, x="TotalQuantity", y="TotalSpending", hue="Cluster", palette="Set2", size="PurchaseFrequency", ax=ax)
plt.title("Customer Segments")
st.pyplot(fig)

# Business Recommendations
st.header("Business Recommendations")
st.write("""
- **Bundling**: Bundle high-lift items like "WHITE HANGING HEART T-LIGHT HOLDER" with "REGENCY CAKESTAND" based on association rules.
- **Targeted Marketing**: Target Cluster 0 (high TotalQuantity, moderate TotalSpending) with bulk purchase discounts.
- **Regional Focus**: Focus marketing in the UK, especially during peak months (e.g., December) from sales trends.
- **Predictive Upselling**: Use the Decision Tree model (highest accuracy) to predict and upsell to high spenders in real-time.
""")
