import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

ds = pd.read_csv('Project_1_data.csv')
ds = ds.fillna(0)
categorical_cols = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']
label_encoder = preprocessing.LabelEncoder()
for col in categorical_cols:
    ds[col] = ds[col].astype(str)
    ds[col] = label_encoder.fit_transform(ds[col])

kmeans = KMeans(n_clusters=7)
kmeans.fit(ds)
ds['Cluster'] = kmeans.labels_
cluster_analysis = ds.groupby('Cluster').mean()
print(cluster_analysis)

score = silhouette_score(ds, ds['Cluster'])
print('Silhouette Score: %.3f' % score)

mean_spending_score = ds.groupby('Cluster')['Spending_Score'].mean()
mean_spending_score.index = mean_spending_score.index + 1

plt.figure(figsize=(10, 5))
plt.bar(mean_spending_score.index, mean_spending_score.values)
plt.xlabel('Cluster')
plt.ylabel('Average Spending Score')
plt.title('Average Spending Score by Cluster')
plt.xticks(range(1, len(mean_spending_score.index) + 2), rotation=90)
plt.ylim([1.1, 1.6])

plt.show()