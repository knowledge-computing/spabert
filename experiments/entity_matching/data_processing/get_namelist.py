import json
import os

def get_name_list_osm(ref_paths):
    name_list = []

    for json_path in ref_paths:
        with open(json_path, 'r') as f:
            data = f.readlines()
        for line in data:
            record = json.loads(line)
            name = record['name']
            name_list.append(name)

    namelist = sorted(namelist)
    return name_list

# deprecated
def get_name_list_usgs_od(ref_paths):
    name_list = []

    for json_path in ref_paths:
        with open(json_path, 'r') as f:
            annot_dict = json.load(f)
        for key, place in annot_dict.items():
            place_name = ''
            for idx in range(1, len(place)+1):
                try:
                    place_name += place[str(idx)]['text_label']
                    place_name += ' ' # separate words with spaces

                except Exception as e:
                    print(place)
            place_name = place_name[:-1] # remove last space
            
            name_list.append(place_name)

    namelist = sorted(namelist)
    return name_list

def get_name_list_usgs_od_per_map(ref_paths):
    all_name_list_dict = dict()

    for json_path in ref_paths:
        map_name = os.path.basename(json_path).split('.json')[0]

        with open(json_path, 'r') as f:
            annot_dict = json.load(f)

        map_name_list = []
        for key, place in annot_dict.items():
            place_name = ''
            for idx in range(1, len(place)+1):
                try:
                    place_name += place[str(idx)]['text_label']
                    place_name += ' ' # separate words with spaces

                except Exception as e:
                    print(place)
            place_name = place_name[:-1] # remove last space
            
            map_name_list.append(place_name)
        all_name_list_dict[map_name] = sorted(map_name_list)

    return all_name_list_dict


def get_name_list_gb1900(ref_path):
    name_list = []

    with open(ref_path, 'r',encoding='utf-16') as f:
        data = f.readlines()


    for line in data[1:]: # skip the header
        try:
            line = line.split(',')
            text = line[1]
            lat = float(line[-3])
            lng = float(line[-2])
            semantic_type = line[-1]
            
            name_list.append(text)
        except:
            print(line)

    namelist = sorted(namelist)

    return name_list


if __name__ == '__main__':
    #name_list = get_name_list_usgs_od(['labGISReport-master/output/USGS-15-CA-brawley-e1957-s1957-p1961.json', 
        #'labGISReport-master/output/USGS-15-CA-capesanmartin-e1921-s1917.json'])
    name_list = get_name_list_gb1900('data/GB1900_gazetteer_abridged_july_2018/gb1900_abridged.csv')
