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

# Configure Streamlit page
st.set_page_config(page_title="E-Commerce Data Mining", layout="wide")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_excel("Online Retail.xlsx")
    df.dropna(subset=["CustomerID", "Description"], inplace=True)
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["Description"] = df["Description"].str.strip()
    df["Country"] = df["Country"].str.strip()
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Clean negative quantities and prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df.drop_duplicates(inplace=True)
    return df

df = load_data()

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("Go to section:", [
    "Dataset Overview", "EDA", "Association Rules",
    "Spender Prediction", "Customer Clustering", "Insights"
])

st.title("🛍️ E-Commerce Dashboard")
st.write("This app performs data mining on the Online Retail dataset, including EDA, Association Rule Mining, Classification, Clustering, and Business Insights.")

# Dataset Overview
if section == "Dataset Overview":
    st.subheader("📄 Data Sample")
    st.write(df.sample(10))
    st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    st.write("Columns:", df.columns.tolist())
    st.write("Column Data Types:")
    st.write(pd.DataFrame(df.dtypes, columns=["Data Type"]))

# EDA Section
elif section == "EDA":
    st.subheader("📊 Exploratory Data Analysis")
    
    # Top products
    st.markdown("### 🥇 Top 10 Products by Frequency")
    st.bar_chart(df["Description"].value_counts().head(10))

    # Top countries
    st.markdown("### 🌍 Top 10 Countries by Sales")
    top_countries = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_countries)

    # Quantity distribution
    st.markdown("### 🧮 Quantity Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["Quantity"], bins=30, kde=True, color="green", ax=ax1)
    st.pyplot(fig1)

    # Price distribution
    st.markdown("### 💵 Unit Price Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df["UnitPrice"], bins=50, kde=True, color="purple", ax=ax2)
    st.pyplot(fig2)

    # Monthly sales
    st.markdown("### 📆 Monthly Sales Trend")
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
    monthly_sales = df.groupby("InvoiceMonth")["TotalPrice"].sum()
    fig3, ax3 = plt.subplots()
    monthly_sales.plot(ax=ax3, color="orange", marker="o")
    ax3.set_ylabel("Total Sales")
    st.pyplot(fig3)

# Association Rules
elif section == "Association Rules":
    st.subheader("🔗 Association Rule Mining")

    # Filter UK transactions and top 50 frequent products
    df_uk = df[df["Country"] == "United Kingdom"]
    top_items = df_uk["Description"].value_counts().head(50).index
    df_uk = df_uk[df_uk["Description"].isin(top_items)]

    # Create basket
    basket = df_uk.groupby(["InvoiceNo", "Description"])["Quantity"].sum().unstack().fillna(0)
    basket_bool = (basket > 0)  # Boolean DataFrame

    if basket_bool.empty or basket_bool.shape[1] < 2:
        st.warning("Not enough transactions for rule mining.")
    else:
        # Apply Apriori and association rules
        frequent_items = apriori(basket_bool, min_support=0.02, use_colnames=True)
        rules = association_rules(frequent_items, metric="lift", min_threshold=1)

        # Show top rules
        st.markdown("### 📋 Top 10 Strong Association Rules")
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

        # Visualize top antecedents
        st.markdown("### 📈 Most Frequent Antecedents")
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        top_antecedents = rules['antecedents_str'].value_counts().head(10)

        fig, ax = plt.subplots()
        top_antecedents.plot(kind='barh', color=sns.color_palette('Dark2'), ax=ax)
        ax.set_title("Top 10 Antecedents in Association Rules")
        ax.set_xlabel("Frequency")
        st.pyplot(fig)


# Classification
elif section == "Spender Prediction":
    st.subheader("🎯 Classification: High vs Low Spender")

    df_class = df.copy()
    df_class["Spender"] = (df_class["TotalPrice"] > df_class["TotalPrice"].median()).astype(int)
    le = LabelEncoder()
    df_class["CountryEncoded"] = le.fit_transform(df_class["Country"])

    X = df_class[["Quantity", "UnitPrice", "CountryEncoded"]]
    y = df_class["Spender"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_choice = st.selectbox("Choose Model", ["Decision Tree", "Naive Bayes", "KNN"])
    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_choice == "Naive Bayes":
        model = GaussianNB()
    else:
        model = KNeighborsClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.text(classification_report(y_test, y_pred))
    st.metric("Accuracy", round(accuracy_score(y_test, y_pred) * 100, 2))

    st.markdown("### 🔍 Predict Your Own")
    qty = st.slider("Quantity", 1, 100)
    price = st.slider("Unit Price", 0.1, 100.0)
    country = st.selectbox("Country", df["Country"].unique())
    country_code = le.transform([country])[0]
    pred = model.predict([[qty, price, country_code]])[0]
    st.success(f"Prediction: {'High Spender' if pred else 'Low Spender'}")

# Clustering
elif section == "Customer Clustering":
    st.subheader("🔵 Customer Segmentation via K-Means")
    customer_data = df.groupby("CustomerID").agg({"Quantity": "sum", "TotalPrice": "sum"}).reset_index()
    k = st.slider("Select Number of Clusters", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, random_state=0)
    customer_data["Cluster"] = kmeans.fit_predict(customer_data[["Quantity", "TotalPrice"]])

    fig, ax = plt.subplots()
    sns.scatterplot(data=customer_data, x="Quantity", y="TotalPrice", hue="Cluster", palette="Set2", ax=ax)
    plt.title("Customer Clusters")
    st.pyplot(fig)

# Insights
elif section == "Insights":
    st.subheader("💡 Business Recommendations")
    st.markdown("""
    ### Key Findings:
    - Most customers purchase small quantities.
    - UK is the dominant market with highest sales.
    - High-spending customers are identifiable with >80% model accuracy.
    - Association rules show co-purchase patterns for bundling.

    ### Recommendations:
    - Target high-value segments with loyalty programs.
    - Use classification model to upsell in real-time.
    - Create product bundles using strong association rules.
    - Expand marketing to countries with growing spend like Netherlands and Germany.
    """)
