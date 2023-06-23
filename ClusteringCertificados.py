import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px

# Initialize LabelEncoder and StandardScaler
scaler = StandardScaler()

def load_data(file):
    df = pd.read_excel(file)

    # Fill missing values with mean of respective columns for numeric data
    df_numeric = df.select_dtypes(include=[np.number])
    df[df_numeric.columns] = df_numeric.fillna(df_numeric.mean())

    # Create a copy of df before transformations for dropdown options
    df_original = df.copy()

    # Normalize the data
    df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

    # Determine the optimal number of clusters using the Elbow method
    wcss = []  # Within-Cluster-Sum-of-Squares
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)

    n_clusters = wcss.index(min(wcss)) + 1
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    df['Cluster'] = kmeans.fit_predict(df)

    return df, kmeans, df_original

def app():
    st.title("Análisis del comportamiento del cliente Certificados")

    pin = st.text_input("Please input your PIN", type="password")

    if pin == "0000":

        # Load the data
        file = st.file_uploader("Please upload your data", type=['xlsx', 'xls'])
        if file is not None:
            df, kmeans, df_original = load_data(file)

            st.write("Please input the client's information below:")

            # Collect user input
            codigo_del_cliente = st.number_input('Código del Cliente', value=0)
            cantidad_de_certificados = st.number_input('Cantidad de certificados', value=0)
            monto_de_apertura = st.number_input('Monto de Apertura', value=0.0)
            duracion = st.number_input('Duracion', value=0)
            duracion_estimada = st.number_input('Duracion estimada', value=0)
            diferencia_duracion = st.number_input('Diferencia duracion', value=0)
            tasa_del_certificado = st.number_input('Tasa del Certificado', value=0.0)
            plazo = st.number_input('Plazo', value=0)

            # Create DataFrame from user input
            new_client = pd.DataFrame([[codigo_del_cliente, cantidad_de_certificados, monto_de_apertura, duracion, duracion_estimada, diferencia_duracion, tasa_del_certificado, plazo]], columns=df.columns[:-1])

            new_client = pd.DataFrame(scaler.transform(new_client), columns=new_client.columns)

            # Predict the cluster for the new client
            new_client_cluster = kmeans.predict(new_client)

            st.write(f"The new client belongs to cluster: {new_client_cluster[0]}")

            # Calculate cluster sizes
            cluster_sizes = df['Cluster'].value_counts(normalize=True)

            # Check if the new client belongs to a "weird" cluster
            if cluster_sizes[new_client_cluster[0]] < 0.10:
                st.write("Warning: This client is behaving in a potentially weirdway.")

            # Visualize the clusters
            fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color='Cluster')
            st.plotly_chart(fig)

        else:
            st.error("Incorrect PIN. Please try again.")

if __name__ == "__main__":
    app()
