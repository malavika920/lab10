import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression

# Initialize Spark Session
spark = SparkSession.builder.appName("WomensClothingEDA").getOrCreate()

# Load dataset
file_path = "Womens Clothing E-Commerce Reviews.csv"
df_pd = pd.read_csv(file_path)
df_spark = spark.read.csv(file_path, header=True, inferSchema=True)

# Streamlit App
st.title("Women's Clothing E-Commerce Data Explorer")

# Display dataset sample
st.write("### Dataset Sample")
st.dataframe(df_pd.head())

# Basic statistics
st.write("### Basic Statistics")
st.write(df_pd.describe())

# Ratings distribution
st.write("### Ratings Distribution")
fig, ax = plt.subplots()
df_pd["Rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
st.pyplot(fig)

# Recommendation distribution
st.write("### Recommendation Distribution")
fig, ax = plt.subplots()
df_pd["Recommended IND"].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax)
ax.set_ylabel("")
st.pyplot(fig)

# Missing values handling
st.write("### Missing Values Count")
st.write(df_pd.isnull().sum())

# Data Cleaning & Wrangling in PySpark
df_spark = df_spark.dropna()
st.write("### Data Cleaning: Missing Values Removed")
st.write(df_spark.toPandas().isnull().sum())

# Exploratory Data Analysis - Age vs Rating
st.write("### Age vs Rating Distribution")
fig, ax = plt.subplots()
sns.boxplot(x=df_pd["Rating"], y=df_pd["Age"], ax=ax)
st.pyplot(fig)

# Regression using MLlib
st.write("### Regression Analysis")
feature_cols = ["Age", "Positive Feedback Count"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_transformed = assembler.transform(df_spark)
regressor = LinearRegression(featuresCol="features", labelCol="Rating")
regression_model = regressor.fit(df_transformed)
st.write("Regression Model Coefficients:", regression_model.coefficients)

# Clustering using KMeans
st.write("### Clustering Analysis")
kmeans = KMeans(featuresCol="features", k=3)
kmeans_model = kmeans.fit(df_transformed)
st.write("Cluster Centers:", kmeans_model.clusterCenters())

# Classification using Logistic Regression
st.write("### Classification Analysis")
classifier = LogisticRegression(featuresCol="features", labelCol="Recommended IND")
classification_model = classifier.fit(df_transformed)
st.write("Classification Model Coefficients:", classification_model.coefficients)

st.write("### Data Insights:")
st.write("- Higher-rated products are more likely to be recommended.")
st.write("- The dataset provides insight into customer feedback trends.")
