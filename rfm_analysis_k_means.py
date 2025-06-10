#!/usr/bin/env python
# coding: utf-8

# # **IMPORT DATA**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


df = pd.read_csv('/content/drive/MyDrive/account_access (1).csv')
df


# In[ ]:


df = df.drop('customer_region', axis=1)
df


# # **DATA PREPARATION**

# **Missing Value**

# In[ ]:


missing_values = df.isnull().sum()
print("Jumlah Nilai Kosong pada Setiap Kolom:")
print(missing_values)


# In[ ]:


df_cleaned_missing = df.dropna()
df_cleaned_missing


# **Zero Values**

# In[ ]:


count_columns_with_zeros = (df_cleaned_missing == 0).sum()
print("Jumlah Kolom dengan Nilai 0:")
print(count_columns_with_zeros)

columns_with_zeros = count_columns_with_zeros[count_columns_with_zeros > 0].index
print("\nKolom-kolom dengan Nilai 0:")
columns_with_zeros


# In[ ]:


df_cleaned_zeros = df_cleaned_missing[(df_cleaned_missing[columns_with_zeros] != 0).all(axis=1)]
df_cleaned_zeros


# In[ ]:


zero_date = df_cleaned_zeros[df_cleaned_zeros['invoice_date'] == '0000-00-00']
zero_date


# In[ ]:


df_cleaned_zero_date = df_cleaned_zeros[df_cleaned_zeros['invoice_date'] != '0000-00-00']
df_cleaned_zero_date


# **Duplicate Data**

# In[ ]:


df_duplicate = df_cleaned_zero_date[df_cleaned_zero_date.duplicated(keep=False)]
df_duplicate


# In[ ]:


df_cleaned_duplicate = df_cleaned_zero_date.drop_duplicates()
dataset = df_cleaned_duplicate
dataset


# # **EXPLORATORY DATA ANALYSIS**

# In[ ]:


dataset['invoice_date'] = pd.to_datetime(dataset['invoice_date'])
dataset['YearMonth'] = dataset['invoice_date'].dt.to_period('M')

monthly_transaction_frequency = dataset.groupby('YearMonth').size()

# Plotting
plt.figure(figsize=(10, 6))
monthly_transaction_frequency.plot(kind='bar', color='skyblue')
plt.title('Monthly Transaction Frequency')
plt.xlabel('Year-Month')
plt.ylabel('Transaction Count')
plt.show()


# In[ ]:


dataset['TransactionValue'] = dataset['product_sales_total']

monthly_transaction_value = dataset.groupby('YearMonth')['TransactionValue'].sum()

# Plotting
plt.figure(figsize=(10, 6))
monthly_transaction_value.plot(kind='bar', color='orange')
plt.title('Monthly Transaction Value')
plt.xlabel('Year-Month')
plt.ylabel('Transaction Value')
plt.show()


# In[ ]:


first_purchase_month = dataset.groupby('customer_id')['YearMonth'].min()

new_customer_count = first_purchase_month.value_counts().sort_index()

cumulative_new_customers = new_customer_count.cumsum()

# Plotting
plt.figure(figsize=(10, 6))
new_customer_count.plot(kind='bar', color='green', alpha=0.7)
plt.title('Monthly New Customer Count')
plt.xlabel('Year-Month')
plt.ylabel('New Customer Count')
plt.show()


# In[ ]:


dataset['DayOfWeek'] = dataset['invoice_date'].dt.day_name()

# Daily order habit with days sorted
daily_order_habit = dataset.groupby('DayOfWeek').size().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Plotting the results
plt.figure(figsize=(10, 6))
daily_order_habit.plot(kind='bar', color='green', alpha=0.7)
plt.title('Daily Order Habit')
plt.xlabel('Day of Week')
plt.ylabel('Order Count')
plt.show()


# In[ ]:


# Most Popular Product by frequency order
popular_product_by_frequency = dataset['product_name'].value_counts()

# Most Popular Product by quantity order
popular_product_by_quantity = dataset.groupby('product_name')['product_qty'].sum().sort_values(ascending=False)

# Most Popular Product by total customer
popular_product_by_customer = dataset.groupby('product_name')['customer_id'].nunique().sort_values(ascending=False)

# Plotting
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 22))

# Most Popular Product by frequency order
axes[0].bar(popular_product_by_frequency.index, popular_product_by_frequency, color='blue', alpha=0.7)
axes[0].set_title('Most Popular Product by Frequency Order')
axes[0].set_xlabel('Product Name')
axes[0].set_ylabel('Frequency')
axes[0].tick_params(axis='x', rotation=90)

# Most Popular Product by quantity order
axes[1].bar(popular_product_by_quantity.index, popular_product_by_quantity, color='green', alpha=0.7)
axes[1].set_title('Most Popular Product by Quantity Order')
axes[1].set_xlabel('Product Name')
axes[1].set_ylabel('Quantity')
axes[1].tick_params(axis='x', rotation=90)

# Most Popular Product by total customer
axes[2].bar(popular_product_by_customer.index, popular_product_by_customer, color='orange', alpha=0.7)
axes[2].set_title('Most Popular Product by Total Customer')
axes[2].set_xlabel('Product Name')
axes[2].set_ylabel('Total Customer')
axes[2].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()


# In[ ]:


repeat_orders = dataset.groupby('customer_id')['invoice_number'].count()

# Filter customers with repeat orders
repeat_customers = repeat_orders[repeat_orders > 1]

# Calculate the average number of repeat orders per customer
average_repeat_orders = repeat_customers.mean()

# Display the number of repeat customers and average repeat orders per customer
print(f"Number of repeat customers: {len(repeat_customers)}")
print(f"Average repeat orders per customer: {average_repeat_orders}")


# # **RFM SEGMENTATION**

# ## **Recency Segmentation**

# In[ ]:


# 1. Recency segmentation
# Assuming today's date is 2023-12-16
today_date = pd.to_datetime('2023-12-16')
dataset['invoice_date'] = pd.to_datetime(dataset['invoice_date'])
recency_df = dataset.groupby('customer_id')['invoice_date'].max().reset_index()
recency_df['recency'] = (today_date - recency_df['invoice_date']).dt.days

# Define recency segmentation boundaries
recency_bins = [0, 30, 90, 180, 360, np.inf]
recency_labels = ['Active', 'Warm', 'Cold', 'Sleep', 'Inactive']
recency_df['recency_segment'] = pd.cut(recency_df['recency'], bins=recency_bins, labels=recency_labels)

plt.figure(figsize=(10, 6))
sns.countplot(data=recency_df, x='recency_segment', order=recency_labels, palette='viridis')
plt.title('Recency Segmentation')
plt.xlabel('Recency Segment')
plt.ylabel('Number of Customers')
plt.show()


# ## **Customer Value Segmentation**

# In[ ]:


# 2. Customer Value Segmentation
frequency_df = dataset.groupby('customer_id')['invoice_number'].nunique().reset_index(name='frequency')
monetary_df = dataset.groupby('customer_id')['product_sales_total'].sum().reset_index(name='monetary')

# Merge Recency, Frequency, and Monetary dataframes
rfm_df = pd.merge(recency_df, frequency_df, on='customer_id')
rfm_df = pd.merge(rfm_df, monetary_df, on='customer_id')


# In[ ]:


correlation_matrix = rfm_df[['recency', 'frequency', 'monetary']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('RFM Value Correlation Matrix')
plt.show()


# In[ ]:


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.boxplot(x=rfm_df['frequency'])
plt.title('Frequency Boxplot')

plt.subplot(1, 2, 2)
sns.scatterplot(x=rfm_df['frequency'], y=rfm_df['monetary'])
plt.title('Scatter Plot: Frequency vs Monetary')

plt.show()


# In[ ]:


print("Frequency values:")
print(rfm_df['frequency'].describe())

print("\nMonetary values:")
print(rfm_df['monetary'].describe())


# In[ ]:


# Assuming rfm_df is your DataFrame

# Remove values greater than specified thresholds
max_frequency = 120
max_monetary = 1.77750e+08

rfm_data = rfm_df[
    (rfm_df['frequency'] <= max_frequency) &
    (rfm_df['monetary'] <= max_monetary)
]

# Print Frequency and Monetary values after removing values greater than specified thresholds
print("Frequency values:")
print(rfm_data['frequency'].describe())

print("\nMonetary values:")
print(rfm_data['monetary'].describe())

# Plotting
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.boxplot(x=rfm_data['frequency'])
plt.title('Frequency Boxplot')

plt.subplot(1, 2, 2)
sns.scatterplot(x=rfm_data['frequency'], y=rfm_data['monetary'])
plt.title('Scatter Plot: Frequency vs Monetary')

plt.show()


# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler()
rfm_scaled = scaler.fit_transform(rfm_data[['recency', 'frequency', 'monetary']])
rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['recency', 'frequency', 'monetary'])

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
rfm_scaled_df['frequency'].hist()
plt.title('Frequency Histogram (Normalized)')

plt.subplot(1, 2, 2)
sns.boxplot(x=rfm_scaled_df['frequency'])
plt.title('Frequency Boxplot (Normalized)')

plt.show()


# ## **Clustering**

# In[ ]:


from sklearn.cluster import KMeans

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled_df)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=5, random_state=42)
rfm_data['cluster'] = kmeans.fit_predict(rfm_scaled_df)

# Frequency/Monetary Segmentation Profiling using bar plot and scatter heat plot
plt.figure(figsize=(14, 5))

cluster_colors = sns.color_palette('viridis', n_colors=len(rfm_data['cluster'].unique()))
cluster_labels = ['Champion', 'Loyal', 'Potential Loyal', 'At Risk', 'Hibernating']

plt.figure(figsize=(12, 4))

# Count plot with consistent colors and labels
plt.subplot(1, 2, 1)
sns.countplot(x=rfm_data['cluster'], palette=cluster_colors)
plt.title('Customer Segmentation by K-means Clustering')
plt.xticks(range(len(cluster_labels)), cluster_labels)  # Set x-axis ticks to cluster_labels

# Scatter plot with consistent colors and labels
plt.subplot(1, 2, 2)
scatter = sns.scatterplot(x=rfm_data['frequency'], y=rfm_data['monetary'], hue=rfm_data['cluster'], palette=cluster_colors, legend='full')
plt.title('Scatter Plot: Frequency vs Monetary (Clustered)')
plt.legend(title='Clusters', loc='upper right')  # Add legend with title and specify location

# To make the legend more readable, you can adjust its position and format
scatter.legend_.set_title('Clusters')  # Set legend title
scatter.legend_.set_bbox_to_anchor((1, 1))  # Adjust legend position

# Customize legend labels
for t, l in zip(scatter.legend_.texts, cluster_labels):
    t.set_text(l)

plt.tight_layout()
plt.show()


# ## **RFM Segmentation Result using heatmap plot**

# In[ ]:


# Assuming 'rfm_data' is your DataFrame

# Adjust colors and labels
cluster_colors = sns.color_palette('viridis', n_colors=len(rfm_data['cluster'].unique()))
cluster_labels = ['Champion', 'Loyal', 'Potential Loyal', 'At Risk', 'Hibernating']

plt.figure(figsize=(10, 5))

# Count plot with labels and legend
sns.countplot(x=rfm_data['recency_segment'], order=recency_labels, hue=rfm_data['cluster'], palette='viridis')
plt.title('Total Customer per Customer Value Segment')
plt.xlabel('Recency Segment')
plt.ylabel('Total Customers')
plt.legend(title='Clusters', loc='upper right')  # Add legend with title and specify location

# Customize legend labels
legend_labels = {cluster: label for cluster, label in zip(rfm_data['cluster'].unique(), cluster_labels)}
plt.legend(title='Clusters', loc='upper right', labels=legend_labels.values())

plt.show()


# In[ ]:


total_monetary_by_recency = rfm_data.groupby('recency_segment')['monetary'].sum().reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(x='recency_segment', y='monetary', data=total_monetary_by_recency, order=recency_labels, palette='viridis')
plt.title('Total Monetary by Recency Segment')
plt.show()


# In[ ]:


# Calculate total monetary by cluster
total_monetary_by_cluster = rfm_data.groupby('cluster')['monetary'].sum().reset_index()

plt.figure(figsize=(10, 5))

# Bar plot with labels
sns.barplot(x='cluster', y='monetary', data=total_monetary_by_cluster, palette='viridis')
plt.title('Total Monetary per Customer Value Segment')
plt.xlabel('Clusters')
plt.ylabel('Total Monetary')
plt.xticks(range(len(cluster_labels)), cluster_labels)  # Set x-axis ticks to cluster_labels

plt.show()


# # **EVALUASI**

# In[ ]:


# Get the inertia value
inertia_value = kmeans.inertia_
error_val = inertia_value/1000
print(f"Inertia: {error_val}")


# In[ ]:


from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(rfm_scaled_df, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg}")

