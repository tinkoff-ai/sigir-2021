import csv
import json

from annoy import AnnoyIndex
from metrics import f1_at_k, mrr_at_k
import pandas as pd
from tqdm import tqdm

from settings import most_popular_sku, TEST_FILE_1, TEST_FILE_2, TRAIN_SESSION_FILE


def categorize_df(df, column_in, column_out):
    try:
        categories = list(sorted(df[column_in].dropna().unique()))
    except:
        categories = df[column_in].dropna().unique()
    c2i = {c: i for i, c in enumerate(categories)}
    c2i["nan"] = 0
    df[column_out] = df[column_in].apply(lambda x: c2i.get(x, 0))
    return df


def read_item_content(file):
    df = pd.read_csv(file)

    df = categorize_df(df, column_in="description_vector", column_out="description_cat")
    df = categorize_df(df, column_in="category_hash", column_out="category_cat")
    df = categorize_df(df, column_in="image_vector", column_out="image_cat")
    df = categorize_df(df, column_in="price_bucket", column_out="price_cat")

    del df["description_vector"]
    del df["category_hash"]
    del df["image_vector"]
    del df["price_bucket"]

    return df.set_index("product_sku_hash").T.to_dict()


def add_content_to_session(sessions, item2content):
    sessions = list(
        map(
            lambda s: [
                i
                for ii in map(
                    lambda i: [
                        i,
                        # f"description_cat_{item2content.get(i, {}).get('description_cat', 0)}",
                        f"category_cat_{item2content.get(i, {}).get('category_cat', 0)}",
                        # f"image_cat_{item2content.get(i, {}).get('image_cat', 0)}",
                        f"price_cat_{item2content.get(i, {}).get('price_cat', 0)}",
                    ],
                    s,
                )
                for i in ii
            ],
            sessions,
        )
    )
    return sessions


def get_sessions_from_df(df, skus=None, only_sku=True):
    user_sessions = []
    current_session_id = None
    current_session = []
    skus = skus or []

    for idx, row in enumerate(df):
        # just append "detail" events in the order we see them
        # row will contain: session_id_hash, product_action, product_sku_hash
        _session_id_hash = row["session_id_hash"]

        # if idx > 200000:
        #    break
        # when a new session begins, store the old one and start again
        if current_session_id and current_session and _session_id_hash != current_session_id:
            # @TODO: NEW THRESHOLD HERE!
            if len(current_session) > 3:
                user_sessions.append(current_session)
            # reset session
            current_session = []
        # check for the right type and append
        if row["product_action"] != "remove":
            skus.append(row["product_sku_hash"])
            if only_sku:
                product_sku_hash = row["product_sku_hash"]  # or row["hashed_url"]
            else:
                product_sku_hash = row["product_sku_hash"] or row["hashed_url"]
            if product_sku_hash is not None and product_sku_hash != "":
                current_session.append(product_sku_hash)
            # current_session.append(row["product_sku_hash"])
        # update the current session id
        current_session_id = _session_id_hash

    return user_sessions, skus


def read_sessions_from_training_file(training_file: str, skus=None, only_sku=True):
    with open(training_file) as csvfile:
        reader = csv.DictReader(csvfile)
        user_sessions, skus = get_sessions_from_df(reader, skus, only_sku=only_sku)

    # print how many sessions we have...
    print("# total sessions: {}".format(len(user_sessions)))
    # print first one to check
    print("First session is: {}".format(user_sessions[0]))
    skus = list(filter(lambda x: x is not None, set(skus)))
    return user_sessions, skus


def read_sessions_from_test_file(test_file: str, skus=None, only_sku=True):
    with open(test_file) as f:
        data = json.load(f)
        data = [item for sublist in data for item in sublist["query"]]

    user_sessions, skus = get_sessions_from_df(data, skus, only_sku=only_sku)

    # print how many sessions we have...
    print("# total sessions: {}".format(len(user_sessions)))
    # print first one to check
    print("First session is: {}".format(user_sessions[0]))
    skus = list(filter(lambda x: x is not None, set(skus)))
    return user_sessions, skus


def make_predictions(model, skus, test_file: str, only_sku=True):
    cnt_preds = 0
    my_predictions = []
    # get all possible SKUs in the model, as a back-up choice
    all_skus = list(model.index_to_key)
    print("Same SKUS.. {}".format(all_skus[:2]))
    with open(test_file) as json_file:
        # read the test cases from the provided file
        test_queries = json.load(json_file)
    # loop over the records and predict the next event
    for t in tqdm(test_queries):
        # this is our prediction, which defaults to a random SKU
        # next_sku = choice(all_skus)
        next_sku = most_popular_sku
        # copy the test case
        _pred = dict(t)
        _products_in_session = [_["product_sku_hash"] for _ in t["query"] if _["product_sku_hash"]]

        # @TODO: new section
        _links_in_session = [_["hashed_url"] for _ in t["query"] if _["hashed_url"]]
        if len(_products_in_session) > 0:
            _last_sku_like = _products_in_session[-1]
        elif not only_sku and len(_links_in_session) > 0:
            _last_sku_like = _links_in_session[-1]
        else:
            _last_sku_like = None

        if _last_sku_like is not None and _last_sku_like in all_skus:
            knn_sku_score = model.similar_by_word(_last_sku_like, topn=100)
            knn_sku = [sku for sku, score in knn_sku_score if sku in skus][:20]
            # knn_sku = list(filter(lambda x: x in skus, knn_sku))[:20]
            cnt_preds += 1
            _pred["label"] = knn_sku
        else:
            _pred["label"] = most_popular_sku

        # @TODO: old section

        # # get last product in the query session and check it is in the model space
        # if _products_in_session and _products_in_session[-1] in all_skus:
        #     # get first product from knn
        #     # next_sku = prod2vec_model.similar_by_word(_products_in_session[-1], topn=1)[0][0]
        #     knn_sku_score = prod2vec_model.similar_by_word(_products_in_session[-1], topn=50)
        #     knn_sku = [sku for sku, score in knn_sku_score]
        #     cnt_preds += 1
        #     _pred["label"] = knn_sku
        # else:
        #     _pred["label"] = most_popular_sku

        assert isinstance(_pred["label"], list)
        # append the label - which needs to be a list

        # append prediction to the final list
        my_predictions.append(_pred)
    # print(next_sku)

    # check for consistency
    assert len(my_predictions) == len(test_queries)
    # print out some "coverage"
    print("Predictions made in {} out of {} total test cases".format(cnt_preds, len(test_queries)))

    return my_predictions


def evaluate_locally(func, args_for_prediction):

    preds = func(**args_for_prediction)

    predictions = [_["label"] for _ in preds]
    ground_truth = [_["gt"] for _ in preds]

    next_item_mrr = mrr_at_k(predictions, ground_truth, 20)
    subsequent_items_f1 = f1_at_k(predictions, ground_truth, 20)

    res = {"mrr_next_item": next_item_mrr, "f1_all_items": subsequent_items_f1}

    return res


class CustomW2V:
    def __init__(self, w2v_model):

        points = []
        self.matrix = w2v_model.vectors
        self.encoder = {k: i for i, k in enumerate(w2v_model.index_to_key)}
        self.decoder = {i: k for i, k in enumerate(w2v_model.index_to_key)}

        self.index_to_key = w2v_model.index_to_key

        for idx, el in enumerate(w2v_model.index_to_key):
            points.append((el, self.matrix[idx]))

        embedding_size = w2v_model.vectors.shape[1]
        f = embedding_size
        t = AnnoyIndex(f)  # Length of item vector that will be indexed
        for idx, point in enumerate(points):
            t.add_item(idx, point[1])

        t.build(10)  # 10 trees
        self.index = t

    def similar_by_word(self, word, topn=20):

        if word in self.encoder:
            nbs = self.index.get_nns_by_item(self.encoder[word], topn)
            recs = [(self.decoder[x], -1) for x in nbs]
            return recs
        else:
            return [(x, -1) for x in most_popular_sku]
