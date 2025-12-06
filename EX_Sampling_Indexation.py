import json
from Modules import text_extractor as te


FT = 'Samplings/MainSample.jsonl'


News = te.data_extractor(FT)
for n in range(len(News)):
    News[n]['index'] = n


with open("Samplings/IndexatedMainSample.jsonl", 'w', encoding = 'utf-8') as f:
    for n in News:
        json_line = json.dumps(n, ensure_ascii = False)
        f.write(json_line + '\n')
