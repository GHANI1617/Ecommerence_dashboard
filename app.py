import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="🛍️ E-Commerce Data Mining", layout="wide")
st.title("🛍️ E-Commerce Retail Data Mining Dashboard")

st.markdown("""
Welcome to the **Retail Data Mining** app! This interactive dashboard covers:
- 🔍 EDA (Exploratory Data Analysis)
- 🔗 Association Rule Mining
- 🎯 Spender Classification
- 🔵 Customer Segmentation (K-Means)
- 💡 Strategic Business Insights
""")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_excel("Online Retail.xlsx")
    df.dropna(subset=["CustomerID", "Description"], inplace=True)
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["Description"] = df["Description"].str.strip()
    df["Country"] = df["Country"].str.strip()
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df.drop_duplicates(inplace=True)
    return df

df = load_data()

# Sidebar
st.sidebar.title("🧭 Navigation")
section = st.sidebar.radio("Go to Section", [
    "📄 Dataset Overview", "📊 EDA", "🔗 Association Rules",
    "🎯 Spender Classification", "🔵 Clustering", "💡 Insights"
])

# 1. Dataset Overview
if section == "📄 Dataset Overview":
    st.subheader("📄 Dataset Overview")
    st.metric("Total Rows", df.shape[0])
    st.metric("Total Columns", df.shape[1])
    st.dataframe(df.sample(10))

    st.markdown("#### 🔍 Missing Values")
    st.write(df.isnull().sum())

    st.markdown("#### 🔎 Column Types")
    st.write(df.dtypes)

# 2. EDA
elif section == "📊 EDA":
    st.subheader("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🥇 Top 10 Products")
        top_products = df["Description"].value_counts().head(10)
        st.bar_chart(top_products)

    with col2:
        st.markdown("#### 🌍 Top 10 Countries by Revenue")
        top_countries = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_countries)

    st.markdown("#### 📦 Quantity Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["Quantity"], bins=30, kde=True, color="teal", ax=ax1)
    st.pyplot(fig1)

    st.markdown("#### 💵 Price Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df["UnitPrice"], bins=50, kde=True, color="darkorange", ax=ax2)
    st.pyplot(fig2)

    st.markdown("#### 📆 Monthly Sales Trend")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Month"] = df["InvoiceDate"].dt.to_period("M")
    monthly_sales = df.groupby("Month")["TotalPrice"].sum()
    fig3, ax3 = plt.subplots()
    monthly_sales.plot(kind="line", marker="o", ax=ax3, color="crimson")
    ax3.set_ylabel("Total Sales")
    ax3.set_title("Monthly Revenue Trend")
    st.pyplot(fig3)

# 3. Association Rule Mining
elif section == "🔗 Association Rules":
    st.subheader("🔗 Association Rule Mining")

    df_uk = df[df["Country"] == "United Kingdom"]
    top_items = df_uk["Description"].value_counts().head(50).index
    df_uk = df_uk[df_uk["Description"].isin(top_items)]
    basket = df_uk.groupby(["InvoiceNo", "Description"])["Quantity"].sum().unstack().fillna(0)
    basket_bool = (basket > 0).astype(bool)

    if basket_bool.empty or basket_bool.shape[1] < 2:
        st.warning("Not enough transactions for rule mining.")
    else:
        frequent_items = apriori(basket_bool, min_support=0.02, use_colnames=True)
        rules = association_rules(frequent_items, metric="lift", min_threshold=1)
        rules_sorted = rules.sort_values("lift", ascending=False)

        st.markdown("### 📋 Top Association Rules")
        st.dataframe(rules_sorted[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

# 4. Classification
elif section == "🎯 Spender Classification":
    st.subheader("🎯 High vs Low Spender Classification")

    df_class = df.copy()
    df_class["Spender"] = (df_class["TotalPrice"] > df_class["TotalPrice"].median()).astype(int)
    le = LabelEncoder()
    df_class["CountryEncoded"] = le.fit_transform(df_class["Country"])

    X = df_class[["Quantity", "UnitPrice", "CountryEncoded"]]
    y = df_class["Spender"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_name = st.selectbox("Choose Model", ["Decision Tree", "Naive Bayes", "KNN"])
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    else:
        model = KNeighborsClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.metric("Accuracy", round(accuracy_score(y_test, y_pred) * 100, 2))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.markdown("### 🔍 Try a Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        qty = st.slider("Quantity", 1, 100, 5)
    with col2:
        price = st.slider("Unit Price", 0.1, 100.0, 10.0)
    with col3:
        country = st.selectbox("Country", sorted(df["Country"].unique()))
    country_code = le.transform([country])[0]
    pred = model.predict([[qty, price, country_code]])[0]
    st.success(f"Prediction: {'💰 High Spender' if pred else '🧾 Low Spender'}")

# 5. Clustering
elif section == "🔵 Clustering":
    st.subheader("🔵 Customer Segmentation (K-Means)")

    customer_df = df.groupby("CustomerID").agg({"Quantity": "sum", "TotalPrice": "sum"}).reset_index()
    k = st.slider("Select Number of Clusters", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, random_state=0)
    customer_df["Cluster"] = kmeans.fit_predict(customer_df[["Quantity", "TotalPrice"]])

    fig, ax = plt.subplots()
    sns.scatterplot(data=customer_df, x="Quantity", y="TotalPrice", hue="Cluster", palette="Set2", ax=ax)
    ax.set_title("Customer Segments")
    st.pyplot(fig)

# 6. Insights
elif section == "💡 Insights":
    st.subheader("💡 Business Recommendations & Insights")

    st.markdown("""
    ### 🔎 Observations:
    - **UK** dominates sales by a large margin.
    - Product sales are heavily **skewed** towards a few top items.
    - High spenders can be predicted with >80% accuracy.
    - Clustering reveals distinct buyer groups (e.g. bulk vs retail buyers).

    ### 🚀 Strategic Recommendations:
    - 🎁 **Bundle offers** for top co-purchased items using association rules.
    - 🧠 Use **spender classification** to target premium customers for upsells.
    - 🌍 Expand to promising countries like **Netherlands** and **Germany**.
    - 🏷️ Target low-quantity, high-margin customers for flash deals.
    - 🤖 Consider automation for segment-specific campaigns using clusters.
    """)
