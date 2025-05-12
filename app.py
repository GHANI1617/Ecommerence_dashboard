%%writefile app.py
import streamlit as st
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

# Page configuration
st.set_page_config(page_title="E-Commerce Data Mining Dashboard", layout="wide")

# Cache the data loading function
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df.dropna(subset=["CustomerID", "Description"], inplace=True)
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["Description"] = df["Description"].str.strip()
    df["Country"] = df["Country"].str.strip()
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])  # Convert to datetime
    return df

# Load data
uploaded_file = st.file_uploader("Upload Online Retail.xlsx", type=["xlsx"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    try:
        df = load_data("Online Retail.xlsx")
    except FileNotFoundError:
        st.error("File 'Online Retail.xlsx' not found. Please upload the dataset or place it in the directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.stop()

# Sidebar navigation
st.sidebar.title("📊 Dashboard Navigation")
menu = st.sidebar.radio("Choose Section", [
    "Dataset Overview", "EDA", "Association Rules",
    "Spender Prediction", "Clustering", "Business Insights"
])

st.title("🛍️ E-Commerce Data Mining Dashboard")

if menu == "Dataset Overview":
    st.subheader("📂 Dataset Sample & Info")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

elif menu == "EDA":
    st.subheader("📊 Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 10 Products**")
        top_products = df["Description"].value_counts().head(10)
        st.bar_chart(top_products)

    with col2:
        st.markdown("**Top 10 Countries by Sales**")
        top_countries = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_countries)

    st.markdown("**Quantity Distribution**")
    fig, ax = plt.subplots()
    sns.histplot(df["Quantity"], bins=30, kde=True, color="green", ax=ax)
    st.pyplot(fig)

    st.markdown("**Monthly Sales Trend**")
    fig, ax = plt.subplots(figsize=(10, 5))
    df.set_index("InvoiceDate").resample('M')["TotalPrice"].sum().plot(ax=ax, title="Monthly Sales Trend")
    st.pyplot(fig)

elif menu == "Association Rules":
    st.subheader("🔗 Association Rule Mining")
    min_support = st.slider("Select Minimum Support", 0.005, 0.05, 0.01, 0.005)
    min_lift = st.slider("Select Minimum Lift", 1.0, 3.0, 1.2, 0.1)

    df_uk = df[df["Country"] == "United Kingdom"]
    basket = df_uk.groupby(["InvoiceNo", "Description"])["Quantity"].sum().unstack().fillna(0)
    basket_bool = (basket > 0).astype(bool)

    if basket_bool.empty or basket_bool.shape[1] < 2:
        st.warning("Not enough transactions for rule mining.")
    else:
        frequent_items = apriori(basket_bool, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_items, metric="lift", min_threshold=min_lift)
        rules = rules.sort_values("lift", ascending=False)
        st.write("**Top 10 Association Rules:**")
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

elif menu == "Spender Prediction":
    st.subheader("🎯 Classification: Spender Prediction")
    df_class = df.copy()
    df_class["PurchaseFrequency"] = df_class.groupby("CustomerID")["InvoiceNo"].transform("nunique")
    df_class["Spender"] = (df_class["TotalPrice"] > df_class["TotalPrice"].quantile(0.75)).astype(int)
    le = LabelEncoder()
    df_class["CountryEncoded"] = le.fit_transform(df_class["Country"])

    X = df_class[["Quantity", "UnitPrice", "PurchaseFrequency", "CountryEncoded"]]
    y = df_class["Spender"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_choice = st.selectbox("Select Model", ["Decision Tree", "Naive Bayes", "KNN"])
    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "Naive Bayes":
        model = GaussianNB()
    else:
        model = KNeighborsClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X, y, cv=5)
    st.text(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    st.metric("Accuracy", f"{round(accuracy_score(y_test, y_pred) * 100, 2)}%")
    st.write(f"Cross-Validation Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

    st.markdown("### 🔍 Try It Yourself")
    col1, col2, col3 = st.columns(3)
    with col1:
        qty = st.slider("Quantity", 1, 100, 1)
    with col2:
        price = st.slider("Unit Price", 0.1, 100.0, 1.0)
    with col3:
        country = st.selectbox("Country", df["Country"].unique())
    country_code = le.transform([country])[0]
    pred = model.predict([[qty, price, 1, country_code]])[0]  # Assuming PurchaseFrequency=1 for simplicity
    st.success(f"Prediction: {'High Spender' if pred else 'Low Spender'}")

elif menu == "Clustering":
    st.subheader("🔵 Customer Segmentation (K-Means)")
    customer_data = df.groupby("CustomerID").agg({
        "Quantity": "sum",
        "TotalPrice": "sum",
        "InvoiceNo": "nunique"
    }).reset_index()
    customer_data.columns = ["CustomerID", "TotalQuantity", "TotalSpending", "PurchaseFrequency"]

    k = st.slider("Number of Clusters", 2, 6, 4)
    scaler = StandardScaler()
    customer_scaled = scaler.fit_transform(customer_data[["TotalQuantity", "TotalSpending", "PurchaseFrequency"]])
    kmeans = KMeans(n_clusters=k, random_state=0)
    customer_data["Cluster"] = kmeans.fit_predict(customer_scaled)

    st.subheader("Elbow Method for Optimal Clusters")
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(customer_scaled)
        inertia.append(kmeans.inertia_)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(range(1, 11), inertia, marker="o")
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=customer_data, x="TotalQuantity", y="TotalSpending", hue="Cluster", palette="Set2", size="PurchaseFrequency", ax=ax)
    plt.title("Customer Segments")
    st.pyplot(fig)

elif menu == "Business Insights":
    st.subheader("💡 Business Recommendations")
    st.markdown("""
    ### Key Insights:
    - 💰 High spenders can be identified using classification models.
    - 🔄 Association rules suggest bundling opportunities (e.g., "WHITE HANGING HEART T-LIGHT HOLDER" with "REGENCY CAKESTAND").
    - 🌍 UK dominates sales, with peak months in December.
    - 🧠 Clustering reveals high-value customer segments.

    ### Actions:
    - 🎁 Implement loyalty programs for high-spending clusters.
    - 📦 Create bundled offers based on association rules.
    - 🧠 Use Decision Tree model for real-time upselling to high spenders.
    - 📅 Focus marketing campaigns in the UK during December.
    """)
