import pickle
from Modules import text_extractor as te
from Modules import text_to_vector as tv


Train = "Samplings/TokenizedMainSample.jsonl"


Traintext = te.news_extractor(Train)
TrainSentences = [Traintext[0].split(' ')]
for t in range(1, len(Traintext)):
    TrainSentences.append(Traintext[t].split(' '))


WSGModel = tv.W2V_Skip_gram(TrainSentences, 200, 10)
with open("VectorizationModels/W2V_Skip_gram_200_10.pkl", 'wb') as f:
    pickle.dump(WSGModel, f)


WCBOWModel = tv.W2V_CBOW(TrainSentences, 200, 10)
with open("VectorizationModels/W2V_CBOW_200_10.pkl", 'wb') as f:
    pickle.dump(WCBOWModel, f)


FSGModel = tv.Fasttext_Skip_gram(TrainSentences, 200, 10)
with open("VectorizationModels/Fasttext_Skip_gram_200_10.pkl", 'wb') as f:
    pickle.dump(FSGModel, f)


FCBOWModel = tv.Fasttext_CBOW(TrainSentences, 200, 10)
with open("VectorizationModels/Fasttext_CBOW_200_10.pkl", 'wb') as f:
    pickle.dump(FCBOWModel, f)


