##IMPORTS
#####################################
# Import module for data manipulation
# Ignore warnings
import warnings

from NursesAnalysis.element import Element

from kmodes.kprototypes import KPrototypes
import pandas as pd

nClusters = 5
warnings.filterwarnings('ignore', category = FutureWarning)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#DATA LOADING AND INSPECTION
###############################
#Loads in the data into df
#df = pd.read_csv("C:/Users/allur/PycharmProjects/pythonProject/nursery.data")
df = pd.read_csv("C:/Users/allur/PycharmProjects/pythonProject/TestingAlgorithm/rewrittenData")
catColumnsPos = [df.columns.get_loc(col) for col in list(df.select_dtypes('object').columns)]
print(catColumnsPos) #consider changin catColumnPos here to not include last object?

##Manually reassigning catColumnPos: because the last column is the answer
catColumnsPos = [0,1,2,4,5,6,7]
print(catColumnsPos)

dfMatrix = df.to_numpy()

df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype(str))
kprototype = KPrototypes(n_jobs = -1, n_clusters = nClusters, init = 'Huang', random_state = 0)
clusters = kprototype.fit(dfMatrix, categorical = catColumnsPos)
labels = kprototype.predict(dfMatrix)

# Add the cluster to the dataframe
df['Cluster Labels'] = kprototype.labels_
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

print("The total distance from the value is ", sum)
print("The algorithm was off by an average of ", sum / 12960, " for each entry")

