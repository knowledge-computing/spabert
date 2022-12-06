import json

json_file = 'entity_linking/outputs/wikidata_usgs_linking_descript.json'

with open(json_file, 'r') as f:
	data = f.readlines()

num_ambi = 0
for line in data:
	line_dict = json.loads(line)
	for key,value in line_dict.items():
		len_value = len(value)
		if len_value < 2:
			continue
		else:
			num_ambi += 1
			print(key)
print(num_ambi)