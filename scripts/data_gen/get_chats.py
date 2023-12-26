import time
import pymongo
from pymongo import errors
import aiohttp
import asyncio
import numpy
import argparse
from lingua import Language, LanguageDetectorBuilder


languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH, Language.RUSSIAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

def prepare_output_success(parsed_log, id):
    return {
        "logid": id,
        "success": True,
        "log": parsed_log,
        "lang": str(detect_language(parsed_log))
    }


def prepare_output_failed(error, id):
    return {
        "logid": id,
        "success": False,
        "reason": error["reason"]
    }

def detect_language(parsed_log):
    chats = ""
    for chat in parsed_log["chat"]:
        if chat["steamid"] != "Console":
            chats = chats + chat["message"] + "."
    return detector.detect_language_of(chats)
    

async def insert_many(col, responses):
    try:
        col.insert_many(responses, ordered=False)
    except errors.BulkWriteError as ex:
        if ex.code == 65:
            #for error in ex.details['writeErrors']:
            #    if error['code'] == 11000:              
            #        logid = error['op']['logid']
            #        if isinstance(logid, int) and len(responses) > 1:
            #            await insert_many(col,
            #                remove_response_by_logid(responses, logid))

            return
        else:
            print("Error inserting responses: ", ex.details)


def remove_response_by_logid(responses, logid: int):
    ret_responses = []
    for response in responses:
        if response['logid'] == logid:
            continue
        ret_responses.append(response)
    return ret_responses


async def get_response(session, api, logid: int):
    try:
        async with session.get(f"{api}{logid}") as response:
            if response.content_type == 'application/json':
                json = await response.json()
                if not response.ok:
                    return prepare_output_failed(json, logid)
                else:
                    return prepare_output_success(json, logid)
            else:
                return prepare_output_failed({"reason": await response.text()}, logid)
    except Exception as ex:
        return prepare_output_failed({"reason": "DISCONNECT_ERROR"}, logid)


async def main(batch_size: int, col, start, end, api):
    batch_times = []
    inserting = None
    async with aiohttp.ClientSession() as session:
        for idx, i in enumerate(numpy.array_split(numpy.arange(start, end, -1), int((start-end) / batch_size))):
            batch_time = time.time()
            responses = await asyncio.gather(*[get_response(session, api, int(logid)) for logid in i])
            if (inserting):
                await inserting
            inserting = asyncio.get_event_loop().create_task(insert_many(col, responses))
            batch_times.append(time.time() - batch_time)
            if len(batch_times) > 50:
                batch_times.pop(0)
            print(f"{start - i[len(i)-1]}/{start - end} | {round((start - i[len(i)-1])*100/(start-end), 2)}% | ETA: {round((numpy.average(batch_times) * ((start -end) /batch_size) - (idx +1)) / (60 * 60), 2)} hours")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-s", "--start", help="Starting logid", type=int, default=3_041_101)
    arg_parser.add_argument("-e", "--end", help="Ending logid", type=int, default=3_000_000)
    arg_parser.add_argument("-m", "--mongo", help="Mongodb connect string")
    arg_parser.add_argument("-d", "--database", help="Mongodb database string", default="BASchrottenbacher")
    arg_parser.add_argument("-c", "--collection", help="Collection to use", default="toxicity_new")
    arg_parser.add_argument("-a", "--api", help="Url to the api endpoint", default="http://localhost:8001/id/")
    arg_parser.add_argument("-b", "--batch", help="Batch size", type=int, default=100)
    args = arg_parser.parse_args()

    myclient = pymongo.MongoClient(args.mongo)
        #"mongodb://user:lht&0nnyo2&P%40%3BvR@lehre.texttechnologylab.org:27021/?authSource=BASchrottenbacher")

    col = myclient[args.database][args.collection]
    asyncio.get_event_loop().run_until_complete(main(args.batch, col, args.start, args.end, args.api))
