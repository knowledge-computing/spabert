import pandas as pd
import json
from request_wrapper import RequestWrapper
import time
import pdb 

start_idx = 17335
wikidata_sample30k_path = 'wikidata_sample30k/wikidata_30k.json'
out_path = 'wikidata_sample30k/wikidata_30k_neighbor.json'

#with open(out_path, 'w') as out_f:
#    pass 

sparql_wrapper = RequestWrapper(baseuri = 'https://query.wikidata.org/sparql')

df= pd.read_json(wikidata_sample30k_path)
df = df[start_idx:]

print('length of df:', len(df))

for index, record in df.iterrows():
    print(index)
    uri = record.results['item']['value']
    q_id = uri.split('/')[-1]
    response = sparql_wrapper.wikidata_nearby_query(str(q_id))
    time.sleep(1)
    with open(out_path, 'a') as out_f:
        out_f.write(json.dumps(response))
        out_f.write('\n')
    

