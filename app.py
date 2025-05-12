import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="📊 Data Mining Dashboard", layout="wide")
st.title("🧠 Interactive Data Mining App")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("📁 Upload a CSV or Excel File", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("🔍 Dataset Preview")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")
    st.write("Missing values:", df.isnull().sum())

    # Step 2: Data Cleaning and Preprocessing
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Step 3: EDA
    st.subheader("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        selected_num = st.selectbox("📈 Numeric Column for Histogram", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_num], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        selected_cat = st.selectbox("📊 Categorical Column for Bar Chart", cat_cols)
        fig2, ax2 = plt.subplots()
        df[selected_cat].value_counts().head(10).plot(kind='bar', ax=ax2)
        st.pyplot(fig2)

    # Step 4: Association Rule Mining
    if cat_cols:
        st.subheader("🔗 Association Rule Mining")
        ar_col = st.selectbox("Select Column for Basket (e.g., product)", cat_cols)
        basket = df.groupby(df.columns[0])[ar_col].value_counts().unstack().fillna(0)
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)

        try:
            frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))
        except:
            st.warning("Association rule mining failed—data may not be suitable.")

    # Step 5: Classification
    st.subheader("🤖 Classification (Decision Tree / Naive Bayes / KNN)")

    target = st.selectbox("🎯 Select Target Variable", num_cols + cat_cols)
    features = st.multiselect("🔢 Select Features", [col for col in df.columns if col != target])

    if features and target:
        X = df[features]
        y = df[target]

        # Encode categorical features
        for col in X.select_dtypes(include='object').columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_type = st.selectbox("Choose Model", ["Decision Tree", "Naive Bayes", "KNN"])
        if model_type == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_type == "Naive Bayes":
            model = GaussianNB()
        else:
            model = KNeighborsClassifier()

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.text("Classification Report:")
        st.text(classification_report(y_test, preds))
        st.metric("Accuracy", round(accuracy_score(y_test, preds) * 100, 2))

    # Step 6: Clustering
    st.subheader("🌀 K-Means Clustering")
    cluster_vars = st.multiselect("Select Variables for Clustering", num_cols)

    if len(cluster_vars) >= 2:
        k = st.slider("Number of Clusters", 2, 6, 3)
        kmeans = KMeans(n_clusters=k, random_state=0)
        data_for_cluster = df[cluster_vars]
        data_for_cluster_scaled = StandardScaler().fit_transform(data_for_cluster)

        clusters = kmeans.fit_predict(data_for_cluster_scaled)
        df["Cluster"] = clusters

        fig3, ax3 = plt.subplots()
        sns.scatterplot(x=data_for_cluster[cluster_vars[0]], y=data_for_cluster[cluster_vars[1]],
                        hue=clusters, palette="Set2", ax=ax3)
        st.pyplot(fig3)

    # Step 7: Recommendations
    st.subheader("💡 Business Recommendations")
    st.markdown("""
    - Analyze frequent patterns to bundle products.
    - Use classification to segment high-value customers.
    - Apply clustering to identify target customer groups.
    - Clean and scale data before modeling for better results.
    """)

else:
    st.info("👈 Upload a dataset to get started.")
