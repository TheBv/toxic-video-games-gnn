import time
from typing import Dict, List
from detoxify import Detoxify
import pymongo
from pymongo import UpdateOne
import numpy
import argparse


def predict_chats(model : Detoxify, chats: List[str]):
    return model.predict(chats)


def apply_prediction_to_elements(elements: Dict[str, dict], predictions: Dict[str, list], model_name: str):
    start = 0
    for key, value in elements.items():
        subset = get_subset_of_predictions(predictions, start, start + len(value["log"]["chat"]))
        elements[key] = apply_chats_to_element(value, subset, model_name)
        start += len(value["log"]["chat"])
    return elements


def get_subset_of_predictions(predictions: Dict[str, list], start, end):
    subset = {}
    for key, value in predictions.items():
        subset[key] = value[start:end]
    return subset


def get_single_of_predictions(predictions: Dict[str, list], index):
    subset = {}
    for key, value in predictions.items():
        try:
            subset[key] = value[index]
        except:
            print("")
    return subset


def apply_chats_to_element(element: dict, predictions: Dict[str, list], model_name: str) -> dict:
    for idx, chat in enumerate(element["log"]["chat"]):
        element["log"]["chat"][idx][f"detoxify-{model_name}"] = get_single_of_predictions(predictions, idx)
    return element


def update_database_bulk(collection, new_elements, model_name):
    operations = []
    for logid, log_data in new_elements.items():
        operations.extend([UpdateOne({"logid": logid}, {"$set": {"log.chat": log_data["log"]["chat"]}}),
                           UpdateOne({"logid": logid}, {"$set": {f"detoxify-{model_name}": True}})])
    collection.bulk_write(operations)


def predict_and_write_chats(model, chats, current_logs, collection, model_name):
    results = predict_chats(model, chats)
    new_elements = apply_prediction_to_elements(current_logs, results, model_name)
    update_database_bulk(collection, new_elements, model_name)

def main(col, model, batch_size: int, model_name: str):
    chats = []
    current_logs = {}
    element: dict
    start_time = time.time()
    chat_amounts = []
    times = []
    for element in col.find({"success": True, f"detoxify-{model_name}": {"$exists": False}, "lang" : "Language.ENGLISH"}):

        new_chats = list(map(lambda x: x["message"], element["log"]["chat"]))

        if len(chats) + len(new_chats) > batch_size:
            predict_and_write_chats(model, chats, current_logs, col, model_name)

            # INFO
            chat_amounts.append(len(chats))
            times.append(time.time() - start_time)
            start_time = time.time()

            if len(chat_amounts) > 10:
                chat_amounts.pop(0)
            if len(times) > 10:
                times.pop(0)

            print(f"Currently processing ~{round(numpy.average(chat_amounts)/numpy.average(times),2)} chats/per second")
            # RESET
            chats = []
            current_logs = {}

        current_logs[element["logid"]] = element
        chats.extend(new_chats)
    # Apply the remaining chats
    predict_and_write_chats(model, chats, current_logs, col, model_name)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-m", "--mongo", help="Mongodb connect string", default="mongodb://localhost:27017")#"mongodb://user:lht&0nnyo2&P%40%3BvR@lehre.texttechnologylab.org:27021/?authSource=BASchrottenbacher")
    arg_parser.add_argument("-d", "--database", help="Mongodb database string", default="BASchrottenbacher")
    arg_parser.add_argument("-c", "--collection", help="Collection to use", default="toxicity_new")
    arg_parser.add_argument("--model", help="Name of the detoxify model to use", default="original", dest="model")
    arg_parser.add_argument("--device", help="Torch device to use", default="cuda", dest="device")
    arg_parser.add_argument("-n", "--name", help="Name of the detoxify model to use", default="original")
    arg_parser.add_argument("-b", "--batch", help="Batch size", type=int, default=200)
    args = arg_parser.parse_args()

    myclient = pymongo.MongoClient(args.mongo)

    col = myclient[args.database][args.collection]

    model = Detoxify(args.model, device=args.device)

    main(col, model, args.batch, args.model)
