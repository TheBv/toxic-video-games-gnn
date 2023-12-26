from pymongo import MongoClient
from bson.objectid import ObjectId
client_base = MongoClient(f'mongodb://localhost:27017')
mydb_base = client_base.get_database("toxic_games")

collection_base = mydb_base.get_collection("games")

client_to_update = MongoClient(f'mongodb://localhost:27017')
mydb_to_update = client_to_update.get_database("toxic_games")

annotators = mydb_to_update.get_collection("annotators")

train_data_client = MongoClient(f'mongodb://localhost:27017')
mydb_train_data= train_data_client.get_database("toxic_games")

train_data = mydb_train_data.get_collection("train_data")


validation_data_client = MongoClient(f'mongodb://localhost:27017')
mydb_validation_data= validation_data_client.get_database("toxic_games")

validation_data = mydb_validation_data.get_collection("validation_data")

object_ids = []

test = {}


collection_base.update_many({}, {"$set" :{"annotation": []}})

# Merge the annotations
for annotator in annotators.find({}):
   annotations = annotator["annotations"]
   for id in annotations.keys():
      item = collection_base.find_one({"_id": ObjectId(id)})
      if item == None:
         continue
      if annotations[id]["problem"]:
         collection_base.update_one({"_id": ObjectId(id)}, {"$set" :{"problem": True}})
         continue
      current_annotation = annotations[id]["gameAnnotation"]
      game_annotation = []
      if "annotation" in item:
         game_annotation = item["annotation"]
      game_annotation.append(current_annotation)
      collection_base.update_one({"_id": ObjectId(id)}, {"$set" :{"annotation": game_annotation}})
   

# Reset the sample types since due to a bug they were lost
for train in train_data.find({}):
   item = collection_base.find({"logid": train["logid"]})
   if item == None:
      continue
   collection_base.update_one({"logid": train["logid"]}, {"$set" :{"type": "train"}})


for validation in validation_data.find({}):
    item = collection_base.find({"logid": validation["logid"]})
    if item == None:
       continue
    collection_base.update_one({"logid": validation["logid"]}, {"$set" :{"type": "random"}})
