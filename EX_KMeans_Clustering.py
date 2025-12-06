import json
import pickle
import os
import numpy as np
from Modules import text_extractor as te
os.environ["LOKY_MAX_CPU_COUNT"] = "1"


Model = None
with open("ClusteringModels/KMeans_Model_9.pkl", 'rb') as f:
    Model = pickle.load(f)


FT = 'Samplings/MainSample.jsonl'
FV = 'Samplings/VectorizedMainSample.jsonl'


News = te.data_extractor(FT)
Vectors = te.vectors_extractor(FV)


for n in range(len(News)):
    News[n]['cluster'] = (str((Model.predict(np.array(Vectors[n]).reshape(1, -1))[0]) + 1))
    FilePath = ("ClusterizedSamplings/KMeansClusters/KMeans_Cluster_" + News[n]['cluster'] + ".jsonl")
    with open(FilePath, 'a', encoding = 'utf-8') as f:
        json_line = json.dumps(News[n], ensure_ascii = False)
        f.write(json_line + '\n')



               
