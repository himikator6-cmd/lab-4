import json
from Modules import text_extractor as te
from Modules import text_preprocessing as tp


FT = 'Samplings/IndexatedMainSample.jsonl'


News = te.data_extractor(FT)
for n in range(len(News)):
    News[n]['text'] = ' '.join(tp.Space_Preprocessed_Tokenizer_Lemmatization(News[n]['text']))


with open("Samplings/TokenizedMainSample.jsonl", 'w', encoding = 'utf-8') as f:
    for n in News:
        json_line = json.dumps(n, ensure_ascii = False)
        f.write(json_line + '\n')
