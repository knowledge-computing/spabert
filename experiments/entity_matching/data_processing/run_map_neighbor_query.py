from query_wrapper import QueryWrapper
from request_wrapper import RequestWrapper
import glob
import os
import json
import time
import random

DATASET_OPTIONS = ['OSM', 'OS', 'USGS', 'GB1900']
KB_OPTIONS = ['wikidata', 'linkedgeodata']

dataset = 'USGS'
kb = 'wikidata'
overwrite = False

assert dataset in DATASET_OPTIONS
assert kb in KB_OPTIONS

if dataset == 'OSM':
    raise NotImplementedError
    
elif dataset == 'OS':  
    raise NotImplementedError
    
elif dataset == 'USGS':

    candidate_file_paths = glob.glob('outputs/alignment_dir/wikidata_USGS*.json')
    candidate_file_paths = sorted(candidate_file_paths)
    
    out_dir = 'outputs/wikidata_neighbors/'

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

elif dataset == 'GB1900':

    raise NotImplementedError
    
else:
    raise NotImplementedError


if kb == 'linkedgeodata':
    sparql_wrapper = QueryWrapper(baseuri = 'http://linkedgeodata.org/sparql')
elif kb == 'wikidata':
    #sparql_wrapper = QueryWrapper(baseuri = 'https://query.wikidata.org/sparql')
    sparql_wrapper = RequestWrapper(baseuri = 'https://query.wikidata.org/sparql')
else:
    raise NotImplementedError

start_map = 6   # 6 
start_line = 4    # 4 

for candiate_file_path in candidate_file_paths[start_map:]:
    map_name = os.path.basename(candiate_file_path).split('wikidata_')[1]
    out_path = os.path.join(out_dir, 'wikidata_' + map_name)

    with open(candiate_file_path, 'r') as f:
        cand_data = f.readlines()

    with open(out_path, 'a') as out_f:
        for line in cand_data[start_line:]:
            line_dict = json.loads(line)
            ret_line_dict = dict()
            for key, value in line_dict.items(): # actually just one pair

                print(key)

                place_name = key 
                for cand_entity in value:
                    time.sleep(2)
                    q_id = cand_entity['item']['value'].split('/')[-1]
                    response = sparql_wrapper.wikidata_nearby_query(str(q_id))

                    if place_name in ret_line_dict:
                        ret_line_dict[place_name].append(response)
                    else:
                        ret_line_dict[place_name] = [response]

                    #time.sleep(random.random()*6)
                    


            out_f.write(json.dumps(ret_line_dict))
            out_f.write('\n')

    print('finished with ',candiate_file_path)
    break
                    
print('done')

'''
        {"Martin": [{"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q27001"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(18.921388888 49.063611111)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "city in northern Slovakia"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q281028"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-101.734166666 43.175)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "city in and county seat of Bennett County, South Dakota, United States"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q761390"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(1.3394 51.179)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "hamlet in Kent"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q2177502"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-100.115 47.826666666)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "city in North Dakota"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q2454021"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-85.64168 42.53698)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "village in Michigan, USA"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q2481111"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-93.21833 32.09917)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "village in Red River Parish, Louisiana, United States"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q2635473"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-82.75944 37.56778)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "city in Kentucky, USA"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q2679547"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-1.9041 50.9759)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "village in Hampshire, England, United Kingdom"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q2780056"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-88.851666666 36.341944444)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "city in Tennessee, USA"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q3261150"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-83.18556 34.48639)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "town in Franklin and Stephens Counties, Georgia, USA"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q6002227"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-101.709 41.2581)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "census-designated place in Nebraska, United States"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q6774807"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-82.1906 29.2936)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "unincorporated community in Marion County, Florida, United States"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q6774809"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-83.336666666 41.5575)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "unincorporated community in Ohio, United States"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q6774810"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(116.037 -32.071)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "suburb of Perth, Western Australia"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q9029707"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-81.476388888 33.069166666)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "unincorporated community in South Carolina, United States"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q11770660"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-0.325391 53.1245)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "village and civil parish in Lincolnshire, UK"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q14692833"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-93.5444 47.4894)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "unincorporated community in Itasca County, Minnesota"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q14714180"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-79.0889 39.2242)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "unincorporated community in Grant County, West Virginia"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q18496647"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(11.95941 46.34585)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "human settlement in Italy"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q20949553"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(22.851944444 41.954444444)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "mountain in Republic of Macedonia"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q24065096"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "<http://www.wikidata.org/entity/Q111> Point(290.75 -21.34)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "crater on Mars"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q26300074"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-0.97879914 51.74729077)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "Thame, South Oxfordshire, Oxfordshire, OX9"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q27988822"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-87.67361111 38.12444444)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "human settlement in Vanderburgh County, Indiana, United States"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q27995389"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-121.317 47.28)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "human settlement in Washington, United States of America"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q28345614"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-76.8 48.5)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "human settlement in Senneterre, Quebec, Canada"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q30626037"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-79.90972222 39.80638889)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "human settlement in United States of America"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q61038281"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-91.13 49.25)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "Meteorological Service of Canada's station for Martin (MSC ID: 6035000), Ontario, Canada"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q63526691"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(152.7011111 -29.881666666)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "Parish of Fitzroy County, New South Wales, Australia"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q63526695"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(148.1011111 -33.231666666)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "Parish of Ashburnham County, New South Wales, Australia"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q96149222"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(14.725573779 48.76768332)"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q96158116"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(15.638358708 49.930136157)"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q103777024"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(-3.125028822 58.694748091)"}, "itemDescription": {"xml:lang": "en", "type": "literal", "value": "Shipwreck off the Scottish Coast, imported from Canmore Nov 2020"}}, {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q107077206"}, "coordinates": {"datatype": "http://www.opengis.net/ont/geosparql#wktLiteral", "type": "literal", "value": "Point(11.688165 46.582982)"}}]}


if overwrite:
    # flush the file if it's been written
    with open(out_path, 'w') as f:
        f.write('')

for name in namelist:
    name = name.replace('"', '')
    name = name.strip("'")
    if len(name) == 0:
      continue
    print(name)
    mydict = dict()
    mydict[name] =  sparql_wrapper.wikidata_query(name)
    line = json.dumps(mydict)
    #print(line)
    with open(out_path, 'a') as f:
        f.write(line)
        f.write('\n')
    time.sleep(1)

print('done')


{"info": {"name": "10TH ST", "geometry": [4193.118085062303, -831.274950414831]}, 
"neighbor_info": 
{"name_list": ["BM 107", "PALM AVE", "WT", "Hidalgo Sch", "PO", "MAIN ST", "BRYANT CANAL", "BM 123", "Oakley Sch", "BRAWLEY", "Witter Sch", "BM 104", "Pistol Range", "Reid Sch", "STANLEY", "MUNICIPAL AIRPORT", "WESTERN AVE", "CANAL", "Riverview Cem", "BEST CANAL"], 
"geometry_list": [[4180.493095652702, -836.0635465095995], [4240.450935702045, -855.345637906981], [4136.084840542623, -917.7895986922882], [4150.386997979736, -948.7258091165079], [4056.955267048625, -847.1018277439381], [4008.112642182582, -849.089249977583], [4124.177447575567, -1004.0706369942257], [4145.382175508665, -626.1608201557082], [4398.137868976953, -764.1087236140554], [4221.1546492913285, -1062.5745271963772], [4015.203890157584, -985.0178210457995], [3989.2345421184878, -948.9340389243871], [4385.585449075614, -660.4590917125413], [3936.505159635338, -803.6822663422273], [3960.1233867112846, -686.7988766730389], [4409.714306709143, -600.6633389979504], [3871.2873706574037, -832.0785684368772], [4304.899727301024, -524.472390102557], [3955.640201659347, -578.5544271698675], [4075.8524354668034, -1183.5837385075774]]}}
'''