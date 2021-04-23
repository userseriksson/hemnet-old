#How to run script,
#pwd:/Users/joeriksson/IdeaProjects/hemnet/sold_items/sold_items
#run: python analytics.py
# Script to flatten the json file in order to convert to pandas.

import pandas as pd
import json
from pandas.io.json import json_normalize
import re
dict ={}
# Load a json file from crwaler
with open('/Users/joeriksson/VSC/villa_hemnet/villa_hemnet/villa_hemnet/spiders/results.json', 'r') as f:
    data = json.load(f)

# Convert to data fram with index
df = pd.DataFrame.from_dict(data,orient='index')
df = df.replace('\n','', regex=True) # Remove new line


# Step 1
def flatten(x):
    d = {}
    # Each element of the dict
    for k,v in x.items():
        # Check value type
        if isinstance(v,list):
            # If list: iter sub dict
            for k_s, v_s in v[0].items():
                d["{}_{}".format(k, k_s)] = v_s
        else: d[k] = v
    return pd.Series(d)

# Unnest first and second dict in pandas
out_first = df.join(df["firs_att"].apply(flatten)).drop("firs_att", axis=1)
out_second = df.join(df["second_att"].apply(flatten)).drop("second_att", axis=1)
# Mergre data piounts
work_data=pd.merge(out_first, out_second[['Antal rum','Boarea','Biarea','Avgift/månad','Driftskostnad','Byggår']], left_index=True, right_index=True)
work_data=work_data.drop(['second_att'],axis=1)

#Complite data set to work on
print(work_data)
print(work_data.info())
work_data.to_csv('/Users/joeriksson/Desktop/python_data/vasa_kungsholmen_20210424.csv')

#store appartment data as  plk file.
work_data.to_pickle('vasa_kungsholmen.plk')  # where to save it, usually as a .pkl

#read an plk data
villa_work_data = pd.read_pickle('vasa_kungsholmen.plk') #to load 123.pkl back to the dataframe df
t=villa_work_data['info'].astype(str)
print(t)
df = pd.DataFrame(villa_work_data)

print(t)
print(villa_work_data)

print(villa_work_data.head())
