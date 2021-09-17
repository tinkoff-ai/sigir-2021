import argparse
import csv
import json
import os

from loguru import logger
from tqdm import tqdm


def read_sessions_from_training_file(training_file: str, K: int = None):
    user_sessions = []
    current_session_id = None
    current_session = []
    prods = set()
    urls = set()
    with open(training_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):

            # print(row['product_action'], row)
            # if idx > 20:
            #     break

            # if a max number of items is specified, just return at the K with what you have
            if K and idx >= K:
                break
            # just append "detail" events in the order we see them
            # row will contain: session_id_hash, product_action, product_sku_hash
            _session_id_hash = row["session_id_hash"]
            # when a new session begins, store the old one and start again
            if (
                current_session_id
                and current_session
                and _session_id_hash != current_session_id
                and len(current_session) > 1
            ):
                user_sessions.append(current_session)
                # reset session
                current_session = []
            # check for the right type and append
            if row["event_type"] in ["detail", "add", "purchase"]:
                prods.add(row["product_sku_hash"])
                current_session.append(row["product_sku_hash"])

            elif row["product_sku_hash"] == "":

                current_session.append(row["hashed_url"])
                urls.add(row["hashed_url"])

            # update the current session id
            current_session_id = _session_id_hash

    # print how many sessions we have...
    print("# total sessions: {}".format(len(user_sessions)))
    # print first one to check
    print("First session is: {}".format(user_sessions[0]))
    # assert user_sessions[0][0] == 'd5157f8bc52965390fa21ad5842a8502bc3eb8b0930f3f8eafbc503f4012f69c'
    # assert user_sessions[0][-1] == '63b567f4cef976d1411aecc4240984e46ebe8e08e327f2be786beb7ee83216d0'

    return user_sessions, prods, urls


def get_statistic(sessions, skus, urls, max_shift=10, max_k=40):

    predictor = dict()

    last_url = None
    shift = 1

    for session in sessions:

        for el in session:
            # print(el in skus, last_url, )
            if el in skus and last_url is not None and shift < max_shift:
                if last_url in predictor.keys():

                    if el in predictor[last_url].keys():
                        predictor[last_url][el] += 1 / shift
                    else:
                        predictor[last_url][el] = 1 / shift
                else:
                    predictor[last_url] = dict()
                    predictor[last_url][el] = 1 / shift

                shift += 1

            elif el in urls:
                last_url = el
                shift = 1

    final_predictor = {}

    for el in predictor.keys():
        final_predictor[el] = [
            x[0] for x in sorted(predictor[el].items(), key=lambda x: x[1], reverse=True)
        ][:max_k]

    logger.info(f"num of keys in predictor is {len(final_predictor)}")
    return final_predictor


def get_sessions_from_test(PATH):

    with open(PATH) as json_file:
        # read the test cases from the provided file
        test_queries = json.load(json_file)
    # loop over the records and predict the next event

    # all_skus = list(model.index_to_key)
    test_s = []

    skus = set()
    for idx, t in tqdm(enumerate(test_queries)):

        items = [
            x["product_sku_hash"] if x["product_sku_hash"] is not None else x["hashed_url"]
            for x in t["query"]
        ]
        if len(items) >= 2:
            test_s.append(items)

        for x in t["query"]:
            if x["product_sku_hash"] != "" and x["product_sku_hash"] is not None:
                skus.add(x["product_sku_hash"])

    return test_s, skus


def get_all_sessions_and_skus(train_session_file, test_file1, test_file2, top_k_sessions):
    sessions = []
    skus = set()

    new_sessions, new_skus = get_sessions_from_test(test_file1)

    sessions.extend(new_sessions)
    for sku in new_skus:
        skus.add(sku)
    logger.info(f"num of sessions after the first test file is {len(sessions)}")
    logger.info(f"num of skus after the first test file is {len(skus)}")

    if test_file2 is not None:
        new_sessions, new_skus = get_sessions_from_test(test_file2)

        sessions.extend(new_sessions)
        for sku in new_skus:
            skus.add(sku)
        logger.info(f"num of sessions after the second test file is {len(sessions)}")
        logger.info(f"num of skus after the first test file is {len(skus)}")
    else:
        logger.info("the second test file is not provided")

    new_sesisons, new_skus, urls = read_sessions_from_training_file(
        train_session_file, K=top_k_sessions
    )

    sessions.extend(new_sesisons)
    for sku in new_skus:
        skus.add(sku)

    return sessions, skus, urls


def main(args):

    sessions, skus, urls = get_all_sessions_and_skus(
        train_session_file=args.train_session_file,
        test_file1=args.test_file1,
        test_file2=args.test_file2,
        top_k_sessions=args.top_k_sessions,
    )

    logger.info(f"num of sessions after the second test file is {len(sessions)}")
    logger.info(f"num of skus after the first test file is {len(skus)}")
    logger.info(f"num of urls after the second test file is {len(urls)}")

    final_predictor = get_statistic(sessions=sessions, skus=skus, urls=urls)

    # dump to file
    with open(args.predictor_path, "w") as outfile:
        json.dump(final_predictor, outfile, indent=2)


def createParser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_session_file", type=str, default="../../sigir_data/train/browsing_train.csv",
    )
    parser.add_argument("--test_file1", type=str, default="../../sigir_data/rec_test_phase_1.json")
    parser.add_argument(
        "--test_file2", type=str, default="../../sigir_data/local_test_phase_2.json"
    )
    parser.add_argument("--predictor_path", type=str, default="search_predictor.json")
    parser.add_argument("--top_k_sessions", type=int)

    return parser


if __name__ == "__main__":

    parser = createParser()
    args, _ = parser.parse_known_args()

    if args.top_k_sessions == -1:
        args.top_k_sessions = None
    main(args)
