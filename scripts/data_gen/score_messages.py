import pymongo
from pymongo import UpdateOne
import argparse



def main(col, batch_size: int, model_name: str):
    chats = []
    element: dict
    operations = []
    for element in col.find({"success": True, f"detoxify-{model_name}": True, f"detoxify-{model_name}-average": {"$exists": False}, "lang" : "Language.ENGLISH"}):
        chats = []
        for chat in element["log"]["chat"]:
            if chat["steamid"] != "Console":
                chats.append(chat)
        score = 0
        for chat in element["log"]["chat"]:
            if chat[f"detoxify-{model_name}"]["toxicity"] >= 0.8:
                score += 1
            if chat[f"detoxify-{model_name}"]["severe_toxicity"] >= 0.8:
                score += 5
        average = -1
        if len(chats) > 0:
            average = score / len(chats)
        operations.extend([UpdateOne({"logid": element["logid"]}, {"$set": {f"detoxify-{model_name}-average": average}})])
        if len(operations) > batch_size:
            col.bulk_write(operations)
            operations = []
    # write the remaining operations
    col.bulk_write(operations)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-m", "--mongo", help="Mongodb connect string", default="mongodb://localhost:27017")#"mongodb://user:lht&0nnyo2&P%40%3BvR@lehre.texttechnologylab.org:27021/?authSource=BASchrottenbacher")
    arg_parser.add_argument("-d", "--database", help="Mongodb database string", default="toxic_games")
    arg_parser.add_argument("-c", "--collection", help="Collection to use", default="games")
    arg_parser.add_argument("--model", help="Name of the detoxify model to use", default="original", dest="model")
    arg_parser.add_argument("-b", "--batch", help="Batch size", type=int, default=200)
    args = arg_parser.parse_args()

    myclient = pymongo.MongoClient(args.mongo)

    col = myclient[args.database][args.collection]

    main(col, args.batch, args.model)
