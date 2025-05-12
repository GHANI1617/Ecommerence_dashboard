import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# Page config
st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("Online Retail.xlsx")
    df.dropna(subset=["CustomerID", "Description"], inplace=True)
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["Description"] = df["Description"].str.strip()
    df["Country"] = df["Country"].str.strip()
    return df

df = load_data()

# Sidebar
st.sidebar.title("📊 Dashboard Navigation")
menu = st.sidebar.radio("Choose Section", [
    "Dataset Overview", "EDA", "Association Rules",
    "Spender Prediction", "Clustering", "Insights"
])

st.title("🛍️ E-Commerce Data Mining Dashboard")

# Dataset Overview
if menu == "Dataset Overview":
    st.subheader("📂 Dataset Sample & Info")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

# EDA
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

# Association Rules
elif menu == "Association Rules":
    st.subheader("🔗 Association Rule Mining (Optimized)")

    st.markdown("This analysis is limited to the top 50 most frequent products in the UK for performance.")

    # Limit to UK and top 50 products
    df_uk = df[df["Country"] == "United Kingdom"]
    top_items = df_uk["Description"].value_counts().head(50).index
    df_uk = df_uk[df_uk["Description"].isin(top_items)]

    # Build the basket
    basket = df_uk.groupby(["InvoiceNo", "Description"])["Quantity"].sum().unstack().fillna(0)
    basket_bool = (basket > 0).astype(bool)

    # Check if basket is empty
    if basket_bool.empty or basket_bool.shape[1] < 2:
        st.warning("Not enough transactions for association rule mining.")
    else:
        try:
            frequent_items = apriori(basket_bool, min_support=0.02, use_colnames=True)
            rules = association_rules(frequent_items, metric="lift", min_threshold=1)
            rules = rules.sort_values(by="lift", ascending=False)

            if rules.empty:
                st.warning("No strong association rules found with current settings.")
            else:
                st.write("**Top 10 Strong Rules**")
                st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

        except Exception as e:
            st.error(f"Error during mining: {e}")

# Spender Prediction
elif menu == "Spender Prediction":
    st.subheader("🎯 Classification: Spender Prediction")

    df_class = df.copy()
    df_class["Spender"] = (df_class["TotalPrice"] > df_class["TotalPrice"].median()).astype(int)
    le = LabelEncoder()
    df_class["CountryEncoded"] = le.fit_transform(df_class["Country"])

    X = df_class[["Quantity", "UnitPrice", "CountryEncoded"]]
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

    st.write("**Classification Report**")
    st.text(classification_report(y_test, y_pred))
    st.metric("Accuracy", round(accuracy_score(y_test, y_pred) * 100, 2))

    st.markdown("### 🔍 Try It Yourself")
    qty = st.slider("Quantity", 1, 100)
    price = st.slider("Unit Price", 0.1, 100.0)
    country = st.selectbox("Country", df["Country"].unique())
    country_code = le.transform([country])[0]
    pred = model.predict([[qty, price, country_code]])[0]
    st.success(f"Prediction: {'High Spender' if pred else 'Low Spender'}")

# Clustering
elif menu == "Clustering":
    st.subheader("🔵 Customer Segmentation (K-Means)")
    customer_data = df.groupby("CustomerID").agg({"Quantity": "sum", "TotalPrice": "sum"}).reset_index()

    k = st.slider("Select number of clusters", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, random_state=0)
    customer_data["Cluster"] = kmeans.fit_predict(customer_data[["Quantity", "TotalPrice"]])

    fig, ax = plt.subplots()
    sns.scatterplot(data=customer_data, x="Quantity", y="TotalPrice", hue="Cluster", palette="Set2", ax=ax)
    plt.title("Customer Segments")
    st.pyplot(fig)

# Insights
elif menu == "Insights":
    st.subheader("💡 Business Recommendations")

    st.markdown("""
    ### Key Insights:
    - 💰 **High spenders** can be identified and targeted with premium offers.
    - 🔄 Frequent co-purchased items can be bundled into combo deals.
    - 🌍 Top sales regions are **UK, Netherlands, and EIRE** — ideal for localized campaigns.
    - 🧠 Classification models offer >80% accuracy in predicting high spenders.

    ### Suggested Actions:
    - ✅ Launch a **customer loyalty program** for high-value clusters.
    - 📦 Create bundles using **frequent itemsets** from association rules.
    """)
