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
for i in range(19):
    Results = cl.AgglomerativeClustering(Vectors, (i + 2))
    x.append(i + 2)
    y1.append(Results[2])
    y2.append(Results[3])
    y3.append(Results[4])
    for j in range(29):
        Results = cl.AgglomerativeClustering(Vectors, (i + 2))
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
