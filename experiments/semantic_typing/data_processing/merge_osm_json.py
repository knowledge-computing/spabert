import os
import json 
import math
import glob
import re
import pdb

'''
NO LONGER NEEDED

Process the california, london, and minnesota OSM data and prepare pseudo-sentence, spatial context

Load the raw output files genrated by sql
Unify the json by changing the structure of dictionary
Save the output into two files, one for training+ validation, and the other one for testing
'''

region_list = ['california','london','minnesota']

input_json_dir = '../data/sql_output/sub_files/'
output_json_dir = '../data/sql_output/'

for region_name in region_list:
    file_list = glob.glob(os.path.join(input_json_dir, 'spatialbert-osm-point-' + region_name + '*.json'))
    file_list = sorted(file_list)
    print('found %d files for region %s' % (len(file_list), region_name))

    
    num_test_files = int(math.ceil(len(file_list) * 0.2))
    num_train_val_files = len(file_list) - num_test_files

    print('%d files for train-val' % num_train_val_files)
    print('%d files for test-tes' % num_test_files)

    train_val_output_path = os.path.join(output_json_dir + 'osm-point-' + region_name + '_train_val.json')
    test_output_path = os.path.join(output_json_dir + 'osm-point-' + region_name + '_test.json')

    # refresh the file
    with open(train_val_output_path, 'w') as f:
        pass
    with open(test_output_path, 'w') as f:
        pass

    for idx in range(len(file_list)):

        if idx < num_train_val_files:
            output_path = train_val_output_path
        else:
            output_path = test_output_path

        file_path = file_list[idx]

        print(file_path)
        
        with open(file_path, 'r') as f:
            data = f.readlines()

        line = data[0]

        line = re.sub(r'\n', '', line)
        line = re.sub(r'\\n', '', line)
        line = re.sub(r'\\+', '', line)
        line = re.sub(r'\+', '', line)

        line_dict_list = json.loads(line)


        for line_dict in line_dict_list:
            
            line_dict = line_dict['json_build_object']

            if not line_dict['name'][0].isalpha(): # discard record if the first char is not enghlish etter
                continue

            neighbor_name_list = line_dict['neighbor_info'][0]['name_list']
            neighbor_geom_list = line_dict['neighbor_info'][0]['geometry_list']

            assert(len(neighbor_geom_list) == len(neighbor_geom_list))

            temp_dict = \
            {'info':{'name':line_dict['name'], 
                    'geometry':{'coordinates':line_dict['geometry']},
                    'class':line_dict['class']
                    }, 
            'neighbor_info':{'name_list': neighbor_name_list,
                            'geometry_list': neighbor_geom_list
                    }
            }

            with open(output_path, 'a') as f:
                json.dump(temp_dict, f)
                f.write('\n')

        

            
                        
