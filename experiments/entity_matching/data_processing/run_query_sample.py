from query_wrapper import QueryWrapper
from get_namelist import *
import glob
import os
import json
import time


#sparql_wrapper_linkedgeo = QueryWrapper(baseuri = 'http://linkedgeodata.org/sparql')

#print(sparql_wrapper_linkedgeo.linkedgeodata_query('Los Angeles'))


sparql_wrapper_wikidata = QueryWrapper(baseuri = 'https://query.wikidata.org/sparql')

#print(sparql_wrapper_wikidata.wikidata_query('Los Angeles'))

#time.sleep(3)

#print(sparql_wrapper_wikidata.wikidata_nearby_query('Q370771'))
print(sparql_wrapper_wikidata.wikidata_nearby_query('Q97625145'))

