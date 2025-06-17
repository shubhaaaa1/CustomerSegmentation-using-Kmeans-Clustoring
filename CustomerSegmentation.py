import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation using KMeans Clustering")

# Upload data
uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.write(df.head())

    # Select features for clustering
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    st.subheader("🧮 Select Features for Clustering")
    selected_features = st.multiselect("Choose at least two numeric features:", numeric_columns, default=numeric_columns[:2])
    
    if len(selected_features) >= 2:
        data = df[selected_features].dropna()

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Select number of clusters
        st.subheader("🔢 Choose Number of Clusters")
        k = st.slider("Select number of clusters (K)", 2, 10, 3)

        # Apply KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        df['Cluster'] = clusters

        st.success("KMeans clustering applied successfully!")

        # Show cluster centers
        st.subheader("📍 Cluster Centers")
        centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=selected_features)
        st.write(centers)

        # Visualization (2D)
        if len(selected_features) == 2:
            st.subheader("📉 Cluster Visualization")
            fig, ax = plt.subplots()
            sns.scatterplot(x=selected_features[0], y=selected_features[1], hue=clusters, data=data, palette="tab10", ax=ax)
            ax.set_title("Customer Segmentation")
            st.pyplot(fig)
        else:
            st.info("Select exactly 2 features for 2D visualization.")

        # Cluster count
        st.subheader("📊 Cluster Distribution")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        st.bar_chart(cluster_counts)

    else:
        st.warning("Please select at least two features.")
else:
    st.info("Awaiting CSV file upload. Please upload to proceed.")

