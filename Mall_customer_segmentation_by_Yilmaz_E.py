# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('QT5Agg')
from matplotlib import pyplot
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

# To see all columns at once
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)


df = pd.read_csv('Mall_Customers.csv')

'''
CustomerID: Unique ID assigned to the customer.
Gender: Gender of the customer.
Age: Age of the customer.
Annual Income: Annual income of the customer.
Spending Score(1-100): Score assigned by the mall based on customer behaviour and spending nature.
'''

def check_data(dataframe,head=5):
    print(20*"-" + "Information".center(20) + 20*"-")
    print(dataframe.info())
    print(20*"-" + "Data Shape".center(20) + 20*"-")
    print(dataframe.shape)
    print("\n" + 20*"-" + "The First 5 Data".center(20) + 20*"-")
    print(dataframe.head())
    print("\n" + 20 * "-" + "The Last 5 Data".center(20) + 20 * "-")
    print(dataframe.tail())
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print(dataframe.isnull().sum())
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T)

check_data(df)

'''
All columns are int64 typed but Gender as object.
All is 200 non-null.
Shape is (200, 5).
All customer are adult, and most of the customers are middle-aged. Max of them is 70 and min of them is 18.
Mean of the annual income is 60.56, and maz is 137k while min is 15k.
Spending score differs from 1 to 99 with mean of 50.2.
'''

# CustomerID is irrelevant in this context so drop the column
df.drop('CustomerID', axis=1, inplace=True)
df.head()


# Separate numerical and categorical columns
numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()

print("Numerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)


# Examine numerical columns
print(df[numerical_columns].describe())


# Distribution plots - histogram - useful for understanding the distribution of each numerical column
df[numerical_columns].hist(bins=15, figsize=(10, 8))
plt.show()


# Distribution plots - density plot - helpful for visualizing the shape of distributions
for col in numerical_columns:
    plt.figure()
    sns.kdeplot(df[col], shade=True)
    plt.title(f'Distribution of {col}')
    plt.show()


# Box plots - great for spotting outliers and comparing distributions across variables
sns.boxplot(data=df[numerical_columns])
plt.xticks(rotation=90)
plt.show()
# None of the numerical columns have outliers except annual income with several observations.


# Correlation analysis - compute correlations between numerical columns to identify relationships
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
# Age and spending score relatively high negative correlation with -0.33


# Feature engineering - Assuming that a person's income should increase by age
df['Age_times_Income'] = df['Age'] * df['Annual Income (k$)']
df.head()


# Pair plots - examine pairwise relationships and scatter plot each pair
sns.pairplot(df[numerical_columns])
plt.show()
# As a striking observation, there is accumulation for middle-aged people with annual income.


# Outlier detection - identify extreme values using the Interquartile Range (IQR) method
Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[numerical_columns] < (Q1 - 1.5 * IQR)) | (df[numerical_columns] > (Q3 + 1.5 * IQR)))
# Filter to show only rows with at least one outlier in any numerical column
outlier_rows = df[outliers.any(axis=1)]
print(outlier_rows)
# There are 2 outliers for income while none for other columns.


# Standard deviation and variance - evaluate variability in each column
print(df[numerical_columns].std())
print(df[numerical_columns].var())
# Std shows that how much a column's mean deviated and variation shows the spread.
'''
Higher Variability in Income and Spending Score: The wide spread in Annual Income and
Spending Score indicates these features are key to identifying meaningful segments, as they
show distinct differences in financial characteristics and spending behavior.

Moderate Variability in Age: While age has some variability, it’s not as high as income or
spending score, so it may be a secondary characteristic for segmenting groups. However, age
still contributes to overall patterns, especially if certain age groups align with specific income
levels or spending habits.

Overall, each feature's variability highlights potential customer groupings, where income and 
spending behavior are likely the most influential in forming distinct segments.
'''


# Skewness and Kurtosis - check for skewness(asymmetry) and kurtosis(peakedness)
print(df[numerical_columns].skew())
print(df[numerical_columns].kurtosis())
'''
Interpretation of Skewness:
Age shows a slight right(positive) skew, meaning that there are several high values pulling the mean above the median,
It is going to be transformed to prevent this.
Annul income also shows the similar pattern and needs to be transformed.
Spending score almost asymmetrical and does not need any adjustments.

Interpretation of Kurtosis:
Age shows a flatter distribution with lighter tails(platykurtic) compared to a normal distribution meaning that there 
are fewer outliers.
Annual income is approximately normal.
Spending score shows similar characteristics as in Age.
'''

# After applying BoxCox transformation, it is observed that it makes worse.
df['Age_sqroot'] = np.sqrt(df['Age'])
numerical_columns = df.select_dtypes(include=['number']).columns.tolist()


# Examine categorical columns
# Frequency distribution - check the count of each category to understand the distribution of values
for col in categorical_columns:
    print(df[col].value_counts())
    print()  # For better readability
# There are 112 females and 88 males.


# Bar plots - visualize the frequency of each category
for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.show()


# GroupBy analysis - check for patterns by grouping categorical columns with numerical columns
for col in numerical_columns:
    print(df.groupby('Gender')[col].mean())
# Even though differences are relatively low, females spend more money than male despite they earn less.


# Binary encoding since there are two genders in this context
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Feature engineering - Grouping age variable
# Define a function to categorize ages
def categorize_age(age):
    if age < 30:
        return 'Young Adult'  # 18 to 29
    elif 30 <= age < 60:
        return 'Adult'  # 30 to 59
    else:
        return 'Senior'  # 60 and above

# Apply the function to the 'Age' column and create a new column 'Age_Group'
df['Age_Group'] = df['Age'].apply(categorize_age)

# Display the updated DataFrame
print(df[['Age', 'Age_Group']].head())


# Robust scaling because there are several outliers
# Select numerical columns
# Separate out the binary variable
numerical_columns = df.select_dtypes(include=['number']).columns.difference(['Gender'])

# Initialize the scaler
scaler = RobustScaler()

# Apply robust scaling only to numerical columns (excluding Gender)
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Display the scaled DataFrame
print(df[numerical_columns].head())


# Unsupervised learning models
# 1. K-means
# Apply KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=0)  # Specify n_clusters based on prior knowledge or experimentation
df['KMeans_Cluster'] = kmeans.fit_predict(df[numerical_columns])

# Optional: Dimensionality Reduction for better visualization (e.g., to 2D)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[numerical_columns])

# Plot KMeans clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df['KMeans_Cluster'], palette="viridis", s=60)
plt.title('KMeans Clustering')
plt.show()

df['KMeans_Cluster'] = kmeans.fit_predict(df[numerical_columns])

wss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df[numerical_columns])
    wss.append(kmeans.inertia_)  # Sum of squared distances to the nearest cluster center

# Plot the WSS for each k value
plt.plot(range(1, 11), wss, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WSS')
plt.title('Elbow Method for Optimal k')
plt.show()
# 4 as elbow seems best choice.


from sklearn.metrics import silhouette_score

# Calculate silhouette score for KMeans clusters
silhouette_avg = silhouette_score(df[numerical_columns], df['KMeans_Cluster'])
print(f'Silhouette Score for KMeans: {silhouette_avg}')
# silhouette score varies from -1 to 1 and shows how well-separated clusters and how closely the data points in each
# cluster resemble one another. Higher the better, and we have moderate as 0.39. So it can be optimized.


from sklearn.metrics import davies_bouldin_score

# Calculate Davies-Bouldin Index for KMeans clusters
db_index = davies_bouldin_score(df[numerical_columns], df['KMeans_Cluster'])
print(f'Davies-Bouldin Index for KMeans: {db_index}')
# 0 is absolute reference point but it is 0.94 here. Thus, there is a room for improvement as well.

from sklearn.metrics import calinski_harabasz_score

# Calculate Calinski-Harabasz Index for KMeans clusters
ch_index = calinski_harabasz_score(df[numerical_columns], df['KMeans_Cluster'])
print(f'Calinski-Harabasz Index for KMeans: {ch_index}')
# Also,  higher the better here. The score is 131.60 and it can be improved.


# 2. Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=4)  # Specify n_clusters based on experimentation
df['Agglomerative_Cluster'] = agg_clustering.fit_predict(df[numerical_columns])

# Plot Agglomerative Clustering
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df['Agglomerative_Cluster'], palette="viridis", s=60)
plt.title('Agglomerative Clustering')
plt.show()

from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

# Calculate Davies-Bouldin Index
db_index_agg = davies_bouldin_score(df[numerical_columns], df['Agglomerative_Cluster'])
print(f'Davies-Bouldin Index for Agglomerative Clustering: {db_index_agg}')
# 0.98

# Calculate Silhouette Score
silhouette_avg_agg = silhouette_score(df[numerical_columns], df['Agglomerative_Cluster'])
print(f'Silhouette Score for Agglomerative Clustering: {silhouette_avg_agg}')
# 0.36

# Calculate Calinski-Harabasz Index
ch_index_agg = calinski_harabasz_score(df[numerical_columns], df['Agglomerative_Cluster'])
print(f'Calinski-Harabasz Index for Agglomerative Clustering: {ch_index_agg}')
# 108.00


from sklearn.cluster import DBSCAN

# 3. DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Parameters need to be tuned
df['DBSCAN_Cluster'] = dbscan.fit_predict(df[numerical_columns])

# Calculate and print evaluation metrics for DBSCAN
db_index_dbscan = davies_bouldin_score(df[numerical_columns], df['DBSCAN_Cluster'])
silhouette_avg_dbscan = silhouette_score(df[numerical_columns], df['DBSCAN_Cluster'], metric='euclidean')
ch_index_dbscan = calinski_harabasz_score(df[numerical_columns], df['DBSCAN_Cluster'])

print(f'Davies-Bouldin Index for DBSCAN: {db_index_dbscan}')   # 2.82
print(f'Silhouette Score for DBSCAN: {silhouette_avg_dbscan}')   # 0.26
print(f'Calinski-Harabasz Index for DBSCAN: {ch_index_dbscan}')   # 52.47


from sklearn.mixture import GaussianMixture

# 4. GMM clustering
gmm = GaussianMixture(n_components=4)  # Specify number of clusters
df['GMM_Cluster'] = gmm.fit_predict(df[numerical_columns])

# Calculate and print evaluation metrics for GMM
db_index_gmm = davies_bouldin_score(df[numerical_columns], df['GMM_Cluster'])   # 1.50
silhouette_avg_gmm = silhouette_score(df[numerical_columns], df['GMM_Cluster'])   # 0.21
ch_index_gmm = calinski_harabasz_score(df[numerical_columns], df['GMM_Cluster'])   # 72.68

print(f'Davies-Bouldin Index for GMM: {db_index_gmm}')
print(f'Silhouette Score for GMM: {silhouette_avg_gmm}')
print(f'Calinski-Harabasz Index for GMM: {ch_index_gmm}')

#######################################################################################################################

'''
Overall. K-means clustering seems better choice compared the others.
'''
# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # Specify n_clusters based on experimentation
df['KMeans_Cluster'] = kmeans.fit_predict(df[numerical_columns])

# Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=4)  # Specify n_clusters based on experimentation
df['Agglomerative_Cluster'] = agg_clustering.fit_predict(df[numerical_columns])

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Parameters need to be tuned
df['DBSCAN_Cluster'] = dbscan.fit_predict(df[numerical_columns])

# Gaussian Mixture Models (GMM)
gmm = GaussianMixture(n_components=4, random_state=42)  # Specify number of clusters
df['GMM_Cluster'] = gmm.fit_predict(df[numerical_columns])

# Display the updated DataFrame with all original observations and clustering results
print(df.head())  # Show the first few rows of the updated DataFrame



