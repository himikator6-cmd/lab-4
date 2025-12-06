import json
import pickle
import numpy as np
from Modules import text_extractor as te
from Modules import text_preprocessing as tp


FT = 'Samplings/TokenizedMainSample.jsonl'


Model = None
with open("VectorizationModels/Fasttext_Skip_gram_200_10.pkl", 'rb') as f:
    Model = pickle.load(f)


TokenizedNews = te.data_extractor(FT)
DocVectors = []
for n in range(len(TokenizedNews)):
    Tokens = (TokenizedNews[n]['text']).split(' ')
    Vectors = []
    for t in range(len(Tokens)):
        if (Tokens[t] in Model.wv):
            Vectors.append(Model.wv[Tokens[t]])
    DocVector = (str(np.mean(Vectors, axis = 0)))
    DocVector = tp.clean_punctuation(DocVector)
    DocVectors.append(DocVector)


with open("Samplings/VectorizedMainSample.jsonl", 'w', encoding = 'utf-8') as f:
    for n in range(len(TokenizedNews)):
        TokenizedNews[n]['text'] = DocVectors[n]
        json_line = json.dumps(TokenizedNews[n], ensure_ascii = False)
        f.write(json_line + '\n')
    
