import json
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from Modules import text_extractor as te
from Modules import clustering as cl
os.environ["LOKY_MAX_CPU_COUNT"] = "1"


FT = 'Samplings/VectorizedMainSample.jsonl'


Vectors = te.vectors_extractor(FT)


print()
print()
x = []
y1 = []
y2 = []
y3 = []
Models = []
for i in range(19):
    Results = cl.KMeans(Vectors, (i + 2))
    x.append(i + 2)
    y1.append(Results[2])
    y2.append(Results[3])
    y3.append(Results[4])
    Models.append(Results[0])
    for j in range(29):
        Results = cl.KMeans(Vectors, (i + 2))
        y1[i] = y1[i] + Results[2]
        y2[i] = y2[i] + Results[3]
        y3[i] = y3[i] + Results[4]
    y1[i] = (y1[i] / 30)
    y2[i] = (y2[i] / 30)
    y3[i] = (y3[i] / 30)
    print(i + 2)
    print()    
plt.plot(x, y1)
plt.legend(['Silhouette'], fontsize = 13)
plt.show()
plt.plot(x, y2)
plt.legend(['CalinskiHarabasz'], fontsize = 13)
plt.show()
plt.plot(x, y3)
plt.legend(['DaviesBouldin'], fontsize = 13)
plt.show()


print()
print()
print()
BestSilhouette = [(i + 2) for i, j in (sorted(enumerate(y1), key = lambda x: x[1], reverse = True))][:3]
print(BestSilhouette)
print()
BestDaviesBouldin = [(i + 2) for i, j in (sorted(enumerate(y3), key = lambda x: x[1], reverse = False))][:3]
print(BestDaviesBouldin)
print()
print()
print()
N = {}
for i in range(len(BestSilhouette)):
    N[BestSilhouette[i]] = N.get(BestSilhouette[i], 0) + (len(BestSilhouette) - i)
for i in range(len(BestDaviesBouldin)):
    N[BestDaviesBouldin[i]] = N.get(BestDaviesBouldin[i], 0) + (len(BestDaviesBouldin) - i)
print(N)
print()
Result = max(N, key = N.get)
print(Result)
print()
print()
print()


Model = (cl.KMeans(Vectors, Result))[0]
print(Model)
print()
print()
Filename = "ClusteringModels/KMeans_Model" + str(Result) + ".pkl"
with open(Filename, 'wb') as f:
    pickle.dump(Model, f)
