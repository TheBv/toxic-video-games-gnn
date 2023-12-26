import torch

from pymongo import MongoClient
from torch_geometric.data import Data
import argparse

def main(count, collection, name, time, threshold):
    source = []
    target = []
    features = []
    graphs = []
    nodes = []

    for idx, item in enumerate(collection.find({"detoxify-original": True, "detoxify-original-average":{ "$exists": True}, "lang" : "Language.ENGLISH"}).limit(count).allow_disk_use(True)):
        # Log the current progress
        if idx % 100 == 0:
            print(f"Processed {idx} matches")
        logdata = item["log"]["events"]

        # Iterate through events creating a node from each event
        # Then connect nodes if they are within a certain timespan and have the same victim/attacker
        for index, event in enumerate(logdata):
            nodes.append([1 if event["kill"] else 0,1 if event["chargeUsed"] else 0,1 if event["medicDrop"] else 0,1 if event["message"] else 0,1 if event["capture"] else 0])
            for index2, event2 in enumerate(logdata):
                if index == index2 or (event["second"] - event2["second"] > time or event["second"] - event2["second"] < 0):
                    continue
                if event["attacker"] == event2["attacker"]:
                    source.append(index)
                    target.append(index2)
                    features.append([1])
                elif event["victim"] and event["victim"] == event2["victim"]:
                    source.append(index)
                    target.append(index2)
                    features.append([0])

        players = list(set(target + source))
        if (len(players) == 0):
            source = []
            target = []
            nodes = []
            features = []
            continue

        toxic = 0
        if (len(item["log"]["chat"]) != 0 and item["detoxify-original-average"] > threshold):
            toxic = 1
        x = torch.tensor(nodes, dtype=torch.float32)
        graphs.append(Data(x=x, edge_attr=torch.tensor(features, dtype=torch.float32), edge_index=torch.tensor([source, target], dtype=torch.long),y=torch.tensor([toxic])))
        source = []
        nodes=[]
        target = []
        features = []
    torch.save(graphs, name)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-m", "--mongo", help="Mongodb connect string", default="mongodb://localhost:27017")
    arg_parser.add_argument("-d", "--database", help="Mongodb database string", default="BASchrottenbacher")
    arg_parser.add_argument("-c", "--collection", help="Collection to use", default="Detoxify10K")
    arg_parser.add_argument("-s", "--size", help="Optional size", type=int, default=0)
    arg_parser.add_argument("-n", "--name", help="Filepath of the graph created", type=str, default='Detoxify.graph')
    arg_parser.add_argument("-t", "--threshold", help="Optional toxicity threshold value", type=float, default=0.3)
    arg_parser.add_argument("--time", help="Optional value for the timespan connected events can be within", type=float, default=10)
    args = arg_parser.parse_args()

    count = args.size
    client = MongoClient(args.mongo)
    mydb = client.get_database(args.database)

    collection = mydb.get_collection(args.collection)
    print("Gathering data...")
    main(count, collection, args.name, args.time, args.threshold)
    print("Done.")