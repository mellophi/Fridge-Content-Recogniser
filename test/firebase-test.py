import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import json
import re
from datetime import datetime
import statistics

# def update_threshold(fetched_data_temp_child, ref_thresh):
#     list_temp_data_dict = list(fetched_data_temp_child.values())
#     avg_dict = {'Bittergourd':[], 'Potato':[], 'Onion':[]}
#     for d in list_temp_data_dict:
#         for key, value in d.items():
#             avg_dict[key].append(value)
    
#     for key, value in avg_dict.items():
#         if value:
#             ref_thresh.update({
#                 key:statistics.mean(value)
#             })

# def get_datetime():
#     date_pat = re.compile(
#             "^(\d{4}-\d{2}-\d{2})\s(\d{2}:\d{2}:\d{2})?")
#     match = date_pat.search(str(datetime.now()))
#     date, time = match.group(1), match.group(2)
#     time = time.replace(':','-')
#     return date, time


if __name__ == '__main__':
    cred = credentials.Certificate('firebase-sdk.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL':'https://firepy-5998e.firebaseio.com/'
    })
    # date, time = get_datetime()
    
#     # random shit
#     vegetable_name = 'Potato'
#     vegetable_weight = -0.5

#     #refences to the database 
#     ref_master = db.reference('/Master') 
    ref_temp = db.reference('/Temp') 
#     ref_temp_child = ref_temp.child(date) #Master/date reference
#     ref_thresh = db.reference('/Threshold')

#     # fetching all data
#     fetched_data_threshold = ref_thresh.get()
    fetched_data_temp = ref_temp.get()
#     fetched_data_master = ref_master.get()
#     fetched_data_temp_child = ref_temp_child.get()

#     # print(fetched_data_temp_child)

#     # deleted the values if it is not in the same date
#     for key, value in ref_temp.get().items():
#         if key != date:
#             update_threshold(fetched_data_temp_child, ref_thresh)
#             ref_temp.delete()


#     # adding old data to the history table (Temp)
#     ref_temp.child(date).push({
#         vegetable_name:float(fetched_data_master[vegetable_name])
#     })
    
#     # updating old data to the new data in master table(Master)
#     new_value = float(fetched_data_master[vegetable_name]) + vegetable_weight
#     ref_master.update({
#         vegetable_name:new_value
#     })

