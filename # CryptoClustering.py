# CryptoClustering
# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %%
# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    "Resources/crypto_market_data.csv",
    index_col="coin_id")

# Display sample data
df_market_data.head(10)

# %%
# Generate summary statistics
df_market_data.describe()

# %%
df_market_data.info()

# %%
# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)

# %% [markdown]
# ---

# %% [markdown]
# ### Prepare the Data

# %%
# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
data_scaled = StandardScaler().fit_transform(df_market_data[['price_change_percentage_24h','price_change_percentage_7d','price_change_percentage_14d',
                                                        'price_change_percentage_30d','price_change_percentage_60d','price_change_percentage_200d',
                                                        'price_change_percentage_1y']])

# %%
# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(data_scaled, columns=['price_change_percentage_24h','price_change_percentage_7d','price_change_percentage_14d',
                                                        'price_change_percentage_30d','price_change_percentage_60d','price_change_percentage_200d',
                                                        'price_change_percentage_1y'])

# Copy the crypto names from the original data
df_market_data_reindex = df_market_data.rename_axis('coin_id').reset_index()
# Set the coinid column as index
df_market_data_scaled['coin_id'] = df_market_data_reindex['coin_id']
df_market_data_scaled.set_index('coin_id', inplace=True)

# Display sample data
df_market_data_scaled.head()

# %% [markdown]
# ---

# %% [markdown]
# ### Find the Best Value for k Using the Original Data.

# %%
# Create a list with the number of k-values from 1 to 11

k = list(range(1,11))

# %%
# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    k_model = KMeans(n_clusters=i, random_state=1)
    k_model.fit(df_market_data_scaled)
    inertia.append(k_model.inertia_)


# %%
# Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k":k, "inertia": inertia}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow = pd.DataFrame(elbow_data)

# %%
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
df_elbow.hvplot.line(
    x='k',
    y='inertia',
    title = 'Elbow Curve',
    xticks = k
)

# %% [markdown]
# #### Answer the following question: 
# 
# **Question:** What is the best value for `k`?
# 
# **Answer:** There is a distinct "elbow" at k = 4.  Four is the best starting point for identifying k.

# %% [markdown]
# ---

# %% [markdown]
# ### Cluster Cryptocurrencies with K-means Using the Original Data

# %%
# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4, random_state=1)

# %%
# Fit the K-Means model using the scaled data
model.fit(df_market_data_scaled)

# %%
# Predict the clusters to group the cryptocurrencies using the scaled data
kmeans_predictions = model.predict(df_market_data_scaled)

# Print the resulting array of cluster values.
kmeans_predictions

# %%
# Create a copy of the DataFrame
df_market_data_scaled_predictions = df_market_data_scaled.copy()

# %%
# Add a new column to the DataFrame with the predicted clusters
df_market_data_scaled_predictions['cluster_id']=kmeans_predictions

# Display sample data
df_market_data_scaled_predictions.head()

# %%
# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
df_market_data_scaled_predictions.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    by = 'cluster_id'
)

# %% [markdown]
# ---

# %% [markdown]
# ### Optimize Clusters with Principal Component Analysis.

# %%
# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)

# %%
# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
pca_data = pca.fit_transform(df_market_data_scaled)
# View the first five rows of the DataFrame. 
print(pca_data[:5])

# %%
# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
pca.explained_variance_ratio_

# %% [markdown]
# #### Answer the following question: 
# 
# **Question:** What is the total explained variance of the three principal components?
# 
# **Answer:** The values added together = 0.3719856 + 0.3470083 + 0.17603793 = 0.895    
# The total explained variance of the three principal components is 0.895 .

# %%
# Create a new DataFrame with the PCA data.
df_pca_data = pd.DataFrame(pca_data, columns=["PCA1", "PCA2", "PCA3"])
# Creating a DataFrame with the PCA data

# Copy the crypto names from the original data
df_pca_data['coin_id'] = df_market_data_reindex['coin_id']

# Set the coinid column as index
df_pca_data.set_index("coin_id",inplace=True)

# Display sample data
df_pca_data.head()

# %% [markdown]
# ---

# %% [markdown]
# ### Find the Best Value for k Using the PCA Data

# %%
# Create a list with the number of k-values from 1 to 11
k=list(range(1,11))

# %%
# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    pca_k_model = KMeans(n_clusters=i, random_state=1)
    pca_k_model.fit(df_pca_data)
    inertia.append(pca_k_model.inertia_)

# %%
# Create a dictionary with the data to plot the Elbow curve
pca_elbow_data={"k":k, "pca_inertia":inertia}
# Create a DataFrame with the data to plot the Elbow curve
df_pca_elbow = pd.DataFrame(pca_elbow_data)

# %%
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
df_pca_elbow.hvplot.line(
    x = "k",
    y = "pca_inertia",
    title = "PCA Elbow Curve",
    xticks = k
)

# %% [markdown]
# #### Answer the following questions: 
# 
# * **Question:** What is the best value for `k` when using the PCA data?
# 
#   * **Answer:**
# The best value for k using PCA data is 4.
# 
# * **Question:** Does it differ from the best k value found using the original data?
# 
#   * **Answer:** 
#   Both the original scaled data and the PCA data provide a clear "elbow" at k=4.  They both predict the same value.

# %% [markdown]
# ### Cluster Cryptocurrencies with K-means Using the PCA Data

# %%
# Initialize the K-Means model using the best value for k
k_model_pca = KMeans(n_clusters=4, random_state=1)

# %%
# Fit the K-Means model using the PCA data
k_model_pca.fit(df_pca_data)

# %%
# Predict the clusters to group the cryptocurrencies using the PCA data
kmeans_predictions_pca = k_model_pca.predict(df_pca_data)
# Print the resulting array of cluster values.
kmeans_predictions_pca

# %%
# Create a copy of the DataFrame with the PCA data
df_pca_data_predictions = df_pca_data.copy()

# Add a new column to the DataFrame with the predicted clusters
df_pca_data_predictions['cluster_id'] = kmeans_predictions_pca

# Display sample data
df_pca_data_predictions.head()

# %%
# Create a scatter plot using hvPlot by setting 
# `x="PC1"` and `y="PC2"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
df_pca_data_predictions.hvplot.scatter(
    x = "PCA1",
    y = "PCA2",
    by = "cluster_id"
)

# %% [markdown]
# ### Visualize and Compare the Results
# 
# In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.

# %%
# Composite plot to contrast the Elbow curves
df_elbow_composite = pd.merge(df_elbow, df_pca_elbow, on = "k")

df_elbow_composite.hvplot.line(
    x = "k",
    y = ["inertia", "pca_inertia"],
    title = "Composite Elbow Curve",
    xticks = k
)


# %%
# OR
df_elbow_composite.hvplot.line(
    x = "k",
    y = "inertia",
    title = "Elbow Curve",
    xticks = k
) + df_elbow_composite.hvplot.line(
    x = "k",
    y = "pca_inertia",
    title = "pca Elbow Curve",
    xticks = k
)

# %%
# Composite plot to contrast the clusters
df_market_data_scaled_predictions.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    by = 'cluster_id'
) + df_pca_data_predictions.hvplot.scatter(
    x = "PCA1",
    y = "PCA2",
    by = "cluster_id"
)

# %% [markdown]
# #### Answer the following question: 
# 
#   * **Question:** After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
# For these data, my analysis shows little advantage to using fewer features to cluster the data. The elbow curves for the scaled data vs. the pca data are very similar.  Both clearly show a distinct elbow with a k value of 4.  Using the pca data, there is greater separation for the clusters that conain single data points.  However, there is no clear visual impact on density of the clusters or separation between the two most populated clusters.  
# 
# This means that our analysis yields similar results with the data with fewer features.  The main advantage of performing the pca is to reduce the number of features in the data for data processing storage and speed.
#   


