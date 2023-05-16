##IMPORTS
#####################################
# Import module for data manipulation
# Ignore warnings
import warnings
from sklearn.metrics import silhouette_score

from NursesAnalysis.element import Element

from kmodes.kmodes import KModes
import pandas as pd

nClusters = 5
warnings.filterwarnings('ignore', category = FutureWarning)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#DATA LOADING AND INSPECTION
###############################
#Loads in the data into df
df = pd.read_csv("C:/Users/allur/PycharmProjects/pythonProject/NursesAnalysis/nursery.data")

catColumnsPos = [df.columns.get_loc(col) for col in list(df.select_dtypes('object').columns)]
print(catColumnsPos) #consider changin catColumnPos here to not include last object?

##Manually reassigning catColumnPos: because the last column is the answer
catColumnsPos = [0,1,2,3,4,5,6,7]
print(catColumnsPos)

dfMatrix = df.to_numpy()


kmode = KModes(n_jobs = -1, n_clusters = nClusters, init = 'Huang', random_state = 0)
clusters = kmode.fit(dfMatrix, categorical = catColumnsPos)
labels = kmode.predict(dfMatrix)

# Add the cluster to the dataframe
df['Cluster Labels'] = kmode.labels_
df['Segment'] = df['Cluster Labels'].map({0:'not_recom', 1:'recommend', 2:'very_recom', 3:'priority', 4:'spec_prior'})
# Order the cluster
df['Segment'] = df['Segment'].astype('category')
df['Segment'] = df['Segment'].cat.reorder_categories(['not_recom','recommend','very_recom','priority','spec_prior'])


results = []
for index, value in df['Cluster Labels'].items():
    #print(f"value is {value} at row {index}")
    e = Element(value,index)
    results.append(e)
for index, value in df['rec_status'].items():
    #print(f"value is {value} at row {index}")
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
kmode = KModes(n_jobs=-1, n_clusters=5, init='Cao')
kmode.fit_predict(dfMatrix, categorical=catColumnsPos)

# Evaluate clustering performance using silhouette score
silhouette_avg = silhouette_score(dfMatrix, kmode.labels_)
print('Silhouette score: {:.3f}'.format(silhouette_avg))


print("The total distance from the value is ", sum)
print("The algorithm was off by an average of ", sum / 12960, " for each entry")

