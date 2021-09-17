import argparse
from datetime import datetime
import json
import os
import time

import boto3
from create_url_sku_dict import get_all_sessions_and_skus, get_statistic
import gensim
from loguru import logger
from tqdm import tqdm

from settings import (
    AWS_ACCESS_KEY,
    AWS_SECRET_KEY,
    BUCKET_NAME,
    EMAIL,
    most_popular_sku,
    PARTICIPANT_ID,
    SUBMISSION_TEST_SAMPLE,
    TRAIN_SESSION_SAMPLE,
)


def train_product_2_vec_model(
    sessions: list,
    min_c: int = 3,
    size: int = 64,
    window: int = 30,
    iterations: int = 15,
    sg=0,
    workers=10,
    ns_exponent: float = 0.75,
):
    """
    Train CBOW to get product embeddings. We start with sensible defaults from the literature - please
    check https://arxiv.org/abs/2007.14906 for practical tips on how to optimize prod2vec.
    :param sessions: list of lists, as user sessions are list of interactions
    :param min_c: minimum frequency of an event for it to be calculated for product embeddings
    :param size: output dimension
    :param window: window parameter for gensim word2vec
    :param iterations: number of training iterations
    :param ns_exponent: ns_exponent parameter for gensim word2vec
    :return: trained product embedding model
    """
    model = gensim.models.Word2Vec(
        sentences=sessions,
        min_count=min_c,
        vector_size=size,
        window=window,
        epochs=iterations,
        sg=sg,
        ns_exponent=ns_exponent,
    )

    print("# products in the space: {}".format(len(model.wv.index_to_key)))

    return model.wv


def make_preds(file, model_cbow, url2sku, skus, k=20):
    cnt_preds = 0
    with open(file) as json_file:
        # read the test cases from the provided file
        test_queries = json.load(json_file)

    all_skus = list(model_cbow.index_to_key)
    blend_count = 0
    my_predictions = []

    counts = 0
    m_value = []
    cnt_preds = 0
    add_count = 0
    add_super = 0
    left = 0

    all_urls = set()

    for idx, t in tqdm(enumerate(test_queries)):

        # copy the test case
        _pred = dict(t)
        removed = [
            _["product_sku_hash"]
            for _ in t["query"]
            if _["product_sku_hash"] and _["product_action"] == "remove"
        ]

        _products_in_session = [
            _["product_sku_hash"]
            for _ in t["query"]
            if _["product_sku_hash"] and _["product_sku_hash"] not in removed
        ]

        urls = [x["hashed_url"] for x in t["query"]]

        skus = [_["product_skus_hash"] for _ in t["query"] if _["product_skus_hash"]]

        joined_skus = []

        sku_in_search = []
        for el in t["query"]:
            if el["is_search"] and el["product_skus_hash"]:
                sku_in_search.extend(
                    list(set(el["product_skus_hash"]) - set(el["clicked_skus_hash"] or []))
                )

        recs_added = []

        if len(skus):
            for el in skus:
                joined_skus.extend(el)
            joined_skus = list(set(joined_skus))

        recs = most_popular_sku

        if _products_in_session and _products_in_session[-1] in all_skus:

            recs = model_cbow.similar_by_word(_products_in_session[-1], topn=200)
            recs = [y[0] for y in recs if (y[0] in skus) and y[0] not in _products_in_session][:20]

            """ Blending part """
            if urls[-1] in url2sku.keys():
                recs_url = [x for x in url2sku[urls[-1]] if x not in skus][:50]
                recs = [x for x in recs if x in recs_url] + [x for x in recs if x not in recs_url]

                blend_count += 1

            if len(recs_added):
                recs = recs_added + recs
                add_count += 1

        elif urls[-1] in url2sku.keys() and urls[-1] is not None:
            """If session has no info about interactions with sku"""

            recs = most_popular_sku

            all_urls.add(urls[-1])

            counts += 1
            recs = [x for x in url2sku[urls[-1]] if x not in skus][:50]
            m_value.append(len(recs[:20]))
            cnt_preds += 1

            if len(recs_added):
                recs = recs_added + recs
                add_count += 1

        else:
            recs = most_popular_sku
            left += 1
            if len(recs_added):
                recs = recs_added + recs
                add_super += 1

        _pred["label"] = recs[:k]

        assert isinstance(_pred["label"], list)

        my_predictions.append(_pred)

    logger.info(
        "Predictions made in {} out of {} total test cases".format(cnt_preds, len(test_queries))
    )
    logger.info(f"no preds in {left} cases")

    return my_predictions


def upload_submission(
    my_predictions,
    local_file: str,
    task: str,
    email,
    bucket_name,
    participant_id,
    aws_access_key,
    aws_secret_key,
):
    """
    Thanks to Alex Egg for catching the bug!
    :param local_file: local path, may be only the file name or a full path
    :param task: rec or cart
    :return:
    """

    local_prediction_file = "{}_{}.json".format(email.replace("@", "_"), round(time.time() * 1000))

    # dump to file
    with open(local_prediction_file, "w") as outfile:
        json.dump(my_predictions, outfile, indent=2)

    print("Starting submission at {}...\n".format(datetime.utcnow()))
    # instantiate boto3 client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name="us-west-2",
    )
    s3_file_name = os.path.basename(local_file)
    # prepare s3 path according to the spec
    # it needs to be like e.g. "rec/id/*.json"
    s3_file_path = "{}/{}/{}".format(task, participant_id, s3_file_name)
    # upload file
    s3_client.upload_file(local_file, bucket_name, s3_file_path)
    # say bye
    print("\nAll done at {}: see you, space cowboy!".format(datetime.utcnow()))

    os.remove(local_prediction_file)

    return local_prediction_file


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

    x_data = [x for x in sessions if len(x) > 1]

    model_cbow = train_product_2_vec_model(x_data, sg=0)
    url2sku = get_statistic(sessions=sessions, skus=skus, urls=urls)

    predictions = make_preds(file=args.file, model_cbow=model_cbow, url2sku=url2sku, skus=skus)

    logger.info("Predictions are ready to be uploaded")

    if args.upload_submission:
        upload_submission(
            my_predictions=predictions,
            email=args.email,
            bucket_name=args.bucket_name,
            participant_id=args.participant_id,
            aws_access_key=args.aws_access_key,
            aws_secret_key=args.aws_access_key,
        )
        logger.info("The file is uploaded")
    else:
        logger.info("The file is not uploaded")


def createParser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_session_file", type=str, default=TRAIN_SESSION_SAMPLE)
    parser.add_argument("--test_file1", type=str, default=SUBMISSION_TEST_SAMPLE)
    parser.add_argument("--test_file2", type=str, default=None)
    parser.add_argument("--predictor_path", type=str, default="search_predictor.json")
    parser.add_argument("--top_k_sessions", type=int)
    parser.add_argument("--email", type=str, default=EMAIL)
    parser.add_argument("--bucket_name", type=str, default=BUCKET_NAME)
    parser.add_argument("--participant_id", type=str, default=PARTICIPANT_ID)
    parser.add_argument("--aws_access_key", type=str, default=AWS_ACCESS_KEY)
    parser.add_argument("--aws_secret_key", type=str, default=AWS_SECRET_KEY)
    parser.add_argument("--upload_submission", type=int, default=0)

    parser.add_argument("--file", type=str, default=SUBMISSION_TEST_SAMPLE)

    return parser


if __name__ == "__main__":

    parser = createParser()
    args, _ = parser.parse_known_args()

    if args.top_k_sessions == -1:
        args.top_k_sessions = None

    main(args)
