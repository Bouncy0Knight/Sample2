# Import module for data manipulation
import pandas as pd
# Import module for linear algebra
import numpy as np
# Import module for data visualization
# Import module for k-protoype cluster
from kmodes.kprototypes import KPrototypes
# Ignore warnings
import warnings
from element import Element
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore', category = FutureWarning)
# Format scientific notation from Pandas
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load the data
df = pd.read_csv('C:/Users/allur/PycharmProjects/pythonProject/TestingAlgorithm/rewrittenData')
# The dimension of data
print('Dimension data: {} rows and {} columns'.format(len(df), len(df.columns)))
# Print the first 5 rows
df.head()
# Inspect the data type
df.info()
# Inspect the categorical variables
df.select_dtypes('object').nunique()
# Inspect the numerical variables
df.describe()
# Check missing value
df.isna().sum()
# Get the position of categorical columns
catColumnsPos = [df.columns.get_loc(col) for col in list(df.select_dtypes('object').columns)]
print('Categorical columns           : {}'.format(list(df.select_dtypes('object').columns)))
print('Categorical columns position  : {}'.format(catColumnsPos))
# Convert dataframe to matrix
dfMatrix = df.to_numpy()
# Fit the cluster
kprototype = KPrototypes(n_jobs = -1, n_clusters = 5, init = 'Huang', random_state = 0)
kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)

# Add the cluster to the dataframe
df['Cluster Labels'] = kprototype.labels_
df['Segment'] = df['Cluster Labels'].map({0:'not_recom', 1:'recommend', 2:'very_recom', 3:'priority', 4:'spec_prior'})
# Order the cluster
df['Segment'] = df['Segment'].astype('category')
df['Segment'] = df['Segment'].cat.reorder_categories(['not_recom','recommend','very_recom','priority','spec_prior'])

results = []
for index, value in df['Cluster Labels'].items():
    print(f"value is {value} at row {index}")
    e = Element(value,index)
    results.append(e)
for index, value in df['rec_status'].items():
    print(f"value is {value} at row {index}")
    results[index].setActual(value,index)
    max_row = index

sum = 0
for i in results:
    print(i)
    sum += i.matchNumber()

# Convert categorical data to one-hot encoding
dfOneHot = pd.get_dummies(df)
# Convert dataframe to matrix
dfMatrix = dfOneHot.to_numpy()

# Fit the cluster
kprototype = KPrototypes(n_jobs=-1, n_clusters=5, init='Cao')
kprototype.fit_predict(dfMatrix, categorical=catColumnsPos)

# Evaluate clustering performance using silhouette score
silhouette_avg = silhouette_score(dfMatrix, kprototype.labels_)
print('Silhouette score: {:.3f}'.format(silhouette_avg))

# print("The total distance from the value is ", sum)
# print("The algorithm was off by an average of ", sum / (max_row+1), " for each entry")

