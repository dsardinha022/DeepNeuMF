import random

import numpy
import numpy as np
from .constants import (COLUMN_DICT)
import pandas as pd
from time import time

IS_VAEX = False
try:
    import vaex as vx
    IS_VAEX = True
    print("Vaex is installed! Will eval() using Vaex")
except ImportError:
    print("Vaex is not installed. Evaluation will be done on pandas")
""
def get_ranking_metrics(true_rating, pred_rating, at_k=None, show_timing=False, is_local=False):
    '''
    Evalutaion Metrics are sourced from below repos
    https://github.com/microsoft/recommenders/blob/main/recommenders/evaluation/python_evaluation.py
    https://github.com/hexiangnan/neural_collaborative_filtering/blob/master/evaluate.py
    :param true_ranks:
    :param pred_ranks:
    :return:
    '''

    eval_precision_customer, eval_recall_customer, eval_map_customer, eval_ndcg_customer, eval_mrr_customer = get_customer_at_k_metrics(true_rating, pred_rating, at_k=at_k, show_timing=show_timing)


    return [eval_precision_customer, eval_recall_customer, eval_map_customer, eval_ndcg_customer, eval_mrr_customer]

def intersect_2d_numpy(arr_1, arr_2):
  return np.array([x for x in set(tuple(x) for x in arr_1) & set(tuple(x) for x in arr_2)])

def get_customer_at_k_metrics(true_rating, pred_rating, at_k=None, show_timing=False):
    time_start = time()
    eval_precision, eval_recall, eval_map, eval_ndcg = 0.0, 0.0, 0.0, 0.0


    # if(IS_VAEX):
    #     time_start_vaex = time()
    #     print("Self-Imp VAEX")  # Improve speed later using CPython and custom methods
    #     vaex_rating = vx.from_pandas(true_rating)
    #     vaex_preds = vx.from_pandas(pred_rating)
    #     comm_users = set(vaex_rating[COLUMN_DICT["column_user"]].unique()).intersection(set(vaex_rating[COLUMN_DICT["column_user"]].unique()))
    #     comm_users = list(comm_users)
    #     vaex_rating_comm = vaex_rating[vaex_rating[COLUMN_DICT["column_user"]].isin(comm_users)]
    #     true_rating_comm = vaex_preds[vaex_preds[COLUMN_DICT["column_user"]].isin(comm_users)]
    #     vaex_num_users = len(comm_users)
    #     print(vaex_rating_comm.shape)
    #     if(show_timing):
    #         print("Eval using VAEX was {}".format(time()-time_start_vaex))
    #
    #
    #
    #
    #
    # # Look into other libraries than Numpy for improved performance.
    # # Dask, Vaex or cuDF -> cuDF has NVIDIA GPU Support
    # #print(pred_rating.dtypes)
    # print("Self-Imp NUMPY") #Improve speed later using CPython and custom methods
    # time_start_numpy = time()
    # true_rating_np = true_rating[[COLUMN_DICT["column_user"],COLUMN_DICT["column_item"],COLUMN_DICT["column_rating"]]].to_numpy(dtype=object)
    # pred_rating_np = pred_rating[[COLUMN_DICT["column_user"],COLUMN_DICT["column_item"],COLUMN_DICT["column_prediction"]]].to_numpy(dtype=object)
    # #print(true_rating_np)
    #
    #
    # true_users_unique = np.unique(true_rating_np[:, 0])
    # pred_users_unique = np.unique(pred_rating_np[:, 0])
    # users_intersection = np.intersect1d(true_users_unique, pred_users_unique)
    # common_users_size = len(users_intersection)
    # true_rating_mask = np.isin(true_rating_np[:,0], users_intersection)
    # pred_rating_mask = np.isin(pred_rating_np[:, 0], users_intersection)
    #
    # # Common users from both arrays have been applied
    # true_rating_np = true_rating_np[true_rating_mask]
    # pred_rating_np = pred_rating_np[pred_rating_mask]
    #
    # #true_rating_np = true_rating_np[np.lexsort((-true_rating_np[:,2], true_rating_np[:,0]))]
    # # true_rating_np  = true_rating_np[-true_rating_np[:,2].argsort()]
    # # true_rating_np = true_rating_np[true_rating_np[:, 0].argsort()]
    # #print(true_rating_np)
    #
    # top_k_from_predictions = pred_rating_np[np.lexsort((-pred_rating_np[:,2], pred_rating_np[:,0]))]
    # split_indices = np.where(np.diff(top_k_from_predictions[:, 0]) != 0)[0] + 1
    # k_groups = np.split(top_k_from_predictions, split_indices)
    #
    # lambda_func_cumcount = lambda x: (np.arange(1,len(x)+1))
    # hxz = [g[:15] for g in k_groups]
    # lpr = [lambda_func_cumcount(m) for m in hxz]
    # k_rank = np.concatenate([np.column_stack((arr1, arr2)) for arr1, arr2 in zip(hxz, lpr)])
    #
    # #hits_arr = numpy.intersect1d(k_rank[:, [0,1]], )
    # # hits_arr_intersect = intersect_2d_numpy(k_rank[:, :2], true_rating_np[:, :2]) # [:, :2]  grabs the first two arrays
    # # mask = np.all(np.isin(k_rank[:, :2], hits_arr_intersect), axis=1)
    # # hits_arr = k_rank[mask]
    # print(k_rank)
    # print(k_rank.shape)
    #
    # mask_one = np.isin(k_rank[:, :2], true_rating_np[:, :2])
    # print(mask_one.shape)
    # mask_two = np.isin(true_rating_np[:, :2], k_rank[:, :2])
    # print(mask_two.shape)
    # temp_k_rank = k_rank[:, :2]
    # new_k_rabk = temp_k_rank[mask_one]
    # new_k_rabk = np.unique(new_k_rabk)
    # print(new_k_rabk.shape)
    # new_k_rabk = new_k_rabk.reshape(-1, 2) # Reshape 1d vector to 2d
    # print(new_k_rabk)
    # print(new_k_rabk.shape)
    #
    # temp_true_common = true_rating_np[:, :2]
    # new_true_common = temp_true_common[mask_two]
    #
    # #new_true_common = true_rating_np[mask_two,[0,1,2]]
    # #hits_arr = np.c_[new_k_rabk, new_true_common]
    #
    #
    #
    # print(new_true_common)
    # print(new_true_common.shape)
    # if (show_timing):
    #     print("Eval using NUMPY was {}".format(time() - time_start_numpy))





    ''' 
    Copyright Microsoft
    MIT License: https://github.com/microsoft/recommenders/blob/main/recommenders/evaluation/python_evaluation.py
    
    
    '''
    time_start_mcrsft = time()
    print("MICROSOFT NUMPY USERS INTERSECTION")
    common_u = set(true_rating[COLUMN_DICT["column_user"]]).intersection(
        set(pred_rating[COLUMN_DICT["column_user"]]))
    #print(common_u)

    #common =
    true_rating_common = true_rating[true_rating[COLUMN_DICT["column_user"]].isin(common_u)]
    pred_rating_common = pred_rating[pred_rating[COLUMN_DICT["column_user"]].isin(common_u)]
    num_users = len(common_u)


    # Below is logic for MRR
    true_copy = true_rating_common.copy()
    true_copy = (
        true_copy.sort_values([COLUMN_DICT["column_user"], COLUMN_DICT["column_rating"]],
                                       ascending=[True, False])
        .groupby(str(COLUMN_DICT["column_user"]), as_index=False)
        .head(at_k)  # PASS VARIABLE
        .reset_index(drop=True)
    )
    true_copy = true_copy[true_copy[COLUMN_DICT["column_rating"]] > 5] #take relevance docs with rating higher than 5
    true_copy = true_copy.groupby(COLUMN_DICT["column_user"]).apply(lambda x: x.max()).reset_index(drop=True)

    #print("true copy rank before min : {}".format(true_copy))

    true_copy = true_copy[[COLUMN_DICT["column_user"], COLUMN_DICT["column_item"]]]
    true_copy["rel"] = 1
    # Above is logic for MRR




    top_k_items = (
        pred_rating_common.sort_values([COLUMN_DICT["column_user"], COLUMN_DICT["column_prediction"]],
                                       ascending=[True, False])
        .groupby(str(COLUMN_DICT["column_user"]), as_index=False)
        .head(at_k)  # PASS VARIABLE
        .reset_index(drop=True)
    )

    top_k_items["rank"] = top_k_items.groupby(str(COLUMN_DICT["column_user"]), sort=False).cumcount() + 1

    # MRR Calculation
    # Evaluates on the first query
    # Source logic calculate mrr via pandas: https://softwaredoug.com/blog/2021/04/21/compute-mrr-using-pandas.html
    top_k_ranks = top_k_items[[COLUMN_DICT["column_user"], COLUMN_DICT["column_item"], "rank"]]
    mrr_rank = top_k_ranks.merge(true_copy, how="left", on=[COLUMN_DICT["column_user"], COLUMN_DICT["column_item"]]).fillna(0)
    mrr_rank = mrr_rank[mrr_rank["rel"] == 1]


    mrr_sorted = mrr_rank.groupby([COLUMN_DICT["column_user"], 'rel'])['rank'].min() #group by user and relevance -> get smallest rank each
    if not mrr_sorted.empty:
        mrr_dropped = mrr_sorted.loc[:, 1] #drop all revs of zero if any
        reciprocal_ranks = 1 / (mrr_dropped)
        eval_mrr = reciprocal_ranks.mean()
    else:
        eval_mrr = 0







    hits = pd.merge(top_k_items, true_rating_common, on=[COLUMN_DICT["column_user"], COLUMN_DICT["column_item"]])[
        [COLUMN_DICT["column_user"], COLUMN_DICT["column_item"], "rank"]
    ]
    hit_count = pd.merge(
        hits.groupby(str(COLUMN_DICT["column_user"]), as_index=False)[COLUMN_DICT["column_user"]].agg(
            {"hit": "count"}),
        true_rating_common.groupby(COLUMN_DICT["column_user"], as_index=False)[COLUMN_DICT["column_user"]].agg(
            {"actual": "count"}
        ),
        on=COLUMN_DICT["column_user"],
    )

    # map @ k
    hit_sorted = hits.copy()
    hit_sorted["rr"] = (
                               hit_sorted.groupby(COLUMN_DICT["column_user"]).cumcount() + 1
                       ) / hit_sorted["rank"]

    hit_sum = hit_sorted.groupby(COLUMN_DICT["column_user"]).agg({"rr": "sum"}).reset_index()

    merger = pd.merge(hit_sum, hit_count, on=COLUMN_DICT["column_user"])
    # map @ k ^^


    #NDCG@K
    dcg = hits.merge(pred_rating_common, on=[COLUMN_DICT["column_user"], COLUMN_DICT["column_item"]]).merge(
        true_rating_common, on=[COLUMN_DICT["column_user"], COLUMN_DICT["column_item"]], how="outer",
        suffixes=("_left", None)
    )

    dcg["rel"] = 1
    discfun = np.log

    # Calculate the actual discounted gain for each record
    dcg["dcg"] = dcg["rel"] / discfun(1 + dcg["rank"])

    # Calculate the ideal discounted gain for each record
    df_idcg = dcg.sort_values([COLUMN_DICT["column_user"], COLUMN_DICT["column_rating"]], ascending=False)
    df_idcg["irank"] = df_idcg.groupby(COLUMN_DICT["column_item"], as_index=False, sort=False)[
        COLUMN_DICT["column_rating"]
    ].rank("first", ascending=False)
    df_idcg["idcg"] = df_idcg["rel"] / discfun(1 + df_idcg["irank"])

    # Calculate the actual DCG for each user
    users = dcg.groupby(COLUMN_DICT["column_item"], as_index=False, sort=False).agg({"dcg": "sum"})

    # Calculate the ideal DCG for each user
    df_user = users.merge(
        df_idcg.groupby(COLUMN_DICT["column_item"], as_index=False, sort=False)
        .head(at_k)
        .groupby(COLUMN_DICT["column_item"], as_index=False, sort=False)
        .agg({"idcg": "sum"}),
        on=COLUMN_DICT["column_item"],
    )

    # DCG over IDCG is the normalized DCG
    df_user["ndcg"] = df_user["dcg"] / df_user["idcg"]

    if (show_timing):
        print("Eval using MICROSOFT was {}".format(time() - time_start_mcrsft))

    if hits.shape[0] != 0:
        eval_precision = (hit_count["hit"] / at_k).sum() / num_users
        eval_recall = (hit_count["hit"] / hit_count["actual"]).sum() / num_users
        eval_map = (merger["rr"] / merger["actual"]).sum() / num_users
        eval_ndcg = df_user["ndcg"].mean()

    print("Eval for customers took {}", str(time()-time_start))
    return eval_precision, eval_recall, eval_map, eval_ndcg, eval_mrr