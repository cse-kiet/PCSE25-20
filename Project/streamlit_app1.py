
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

#st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Mutual Funds Clustering App")

uploaded_file = st.file_uploader("Upload your mutual funds CSV file", type=["csv"])
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    df = raw_df.copy()

    features = [
        'ExpenseRatio', 'TurnoverRatio',
        'ALP3Y', 'ALP5Y', 'ALP10Y',
        'BET3Y', 'BET5Y', 'BET10Y',
        'R2_3YRS', 'R2_5YRS', 'R2_10YRS',
        'SD_3YRS', 'SD_5YRS', 'SD_10YRS',
        'SHP_3YRS', 'SHP_5YRS', 'SHP_10YRS'
    ]

    df = df[features].dropna()

    # Standardize and reduce dimensions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Sidebar - Clustering options
    st.sidebar.header("Clustering Options")
    cluster_method = st.sidebar.selectbox("Choose clustering method", ("KMeans", "Agglomerative", "DBSCAN"))

    if cluster_method in ("KMeans", "Agglomerative"):
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
    if cluster_method == "DBSCAN":
        eps = st.sidebar.slider("EPS", 0.1, 5.0, 0.5)
        min_samples = st.sidebar.slider("Min samples", 1, 20, 5)

    # Fit model
    if cluster_method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
    elif cluster_method == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
    elif cluster_method == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)

    # Plot results
    st.subheader("Cluster Visualization (PCA reduced)")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='tab10', s=60)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"{cluster_method} Clustering")
    plt.grid(True)
    st.pyplot()

    # Category analysis
    full_df = raw_df.copy()
    full_df = full_df.dropna(subset=features + ['SchemeCategory']).reset_index(drop=True)
    full_df['Cluster'] = labels

    cluster_label_map = (
        full_df.groupby('Cluster')['SchemeCategory']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
        .to_dict()
    )

    full_df['Predicted_Category'] = full_df['Cluster'].map(cluster_label_map)
    full_df['Misclustered'] = full_df['SchemeCategory'] != full_df['Predicted_Category']

    st.subheader("Cluster Category Assignment")
    for cluster_id, group in full_df.groupby('Cluster'):
        st.markdown(f"### Cluster {cluster_id} (Labeled as: **{cluster_label_map[cluster_id]}**)")
        st.dataframe(group[['SchemeName', 'SchemeCategory', 'Predicted_Category', 'Misclustered']])

    st.subheader("Misclustered Schemes")
    mis = full_df[full_df['Misclustered']]
    if not mis.empty:
        st.dataframe(mis[['SchemeName', 'SchemeCategory', 'Predicted_Category', 'Cluster']])
    else:
        st.success("All schemes were clustered into their true categories!")
