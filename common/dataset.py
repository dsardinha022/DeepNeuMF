# Davina Sardinha
# Copyright (c) Desco Industries. All rights reserved.
# Licensed under the MIT License.
import csv
import itertools
import os
import re

import numpy
import tensorflow as tf
import time
import scipy.sparse as sp
import pandas as pd

import numpy as np
import random
import logging
from collections import OrderedDict, Counter
from numpy import count_nonzero
from .constants import (COLUMN_DICT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFile:


    def __init__(self, path, isBinary, training=None, file_dir=None):

        logger.info("Loading file {} to dataset ...".format(path))

        self.path = path
        self.isBinary = isBinary
        self.training = training
        self.file_dir = file_dir

        #self.data, self.userToId, self.itemToId, self.rawItemDescData, self.training_data_keywords = self.load_file(self.path, self.training)

        self.data, self.userToId, self.itemToId = self.load_file(self.path, self.training)
        self.idToUser = {value: key for (key, value) in self.userToId.items()}
        self.idToItem = {value: key for (key, value) in self.itemToId.items()}

        self.num_users = len(self.userToId.values())
        self.num_items = len(self.itemToId.values())

        self.idToKeyword = None  # Passed on later
        self.num_keywords = None  # Passed on later
        self.keyword_groundtruth = None

        logger.info("Unique Users {}".format(self.num_users))
        logger.info("Unique Items {}".format(self.num_items))

    #REDO BELOW AS THIS TAKES ALOT OF TIME - MORE DATAFRAME FRIENDLY
    def convert_to_month_day(self, timestamp_val):
        #print(timestamp_val)
        # Convert timestamp to datetime object
        dt = pd.to_datetime(int(timestamp_val), unit='s')
        # Format datetime object as month and day string
        md = dt.strftime('%m%d')
        md = int(md)
        # Return the string
        return md

    def load_file(self, filename, training):
        '''
        Dictionaries convert userId and itemID into keys
        This will help construct matrix with indexes outside number of unique items and user
        '''

        if training is not None:
            user_dict = training.userToId
            item_dict = training.itemToId
        else:
            user_dict, item_dict = OrderedDict(), OrderedDict()
        raw_data = []

        raw_item_description = []

        with open(filename, "r", encoding="UTF8") as f:
            csv_reader = csv.DictReader(f, delimiter=',')
            for line in csv_reader:
                u, i, r = int(line[COLUMN_DICT["column_user"]]), int(line[COLUMN_DICT["column_item"]]), \
                    float(line[COLUMN_DICT["column_rating"]])

                # Store training values
                if training is None:
                    if u not in user_dict.keys():
                        user_dict[u] = len(user_dict)

                    if i not in item_dict.keys():
                        item_dict[i] = len(item_dict)

                    if self.isBinary and (r > 0):
                        r = 1


                # Reassign values to dictionary ids (Regardless of training or test)
                u = user_dict[u]
                i = item_dict[i]
                #snapshot = [u, i, b, r, t]
                snapshot = [u, i, r]
                raw_data.append(snapshot)

        raw_dataset = pd.DataFrame(raw_data, columns=[COLUMN_DICT["column_user"], COLUMN_DICT["column_item"],
                                                       COLUMN_DICT["column_rating"]])



        # Comment below if not using TextVectorizor Preprocessing layer
        #raw_dataset[COLUMN_DICT["column_time"]] = raw_dataset[COLUMN_DICT["column_time"]].apply(lambda x: self.convert_to_month_day(x))


        logger.info(raw_dataset.head())

        return raw_dataset, user_dict, item_dict



    def get_user_to_item_data(self, userID):
        items = self.data.loc[self.data[COLUMN_DICT['column_user']] == userID]
        items = items[COLUMN_DICT['column_item']].drop_duplicates()
        return items

    def get_item_to_keywords_data(self, items, raw_item_description, training=None):
        item_to_keywords = []
        word_counts = Counter()
        item_descs = []
        item_descs_set = []


        # Split word by Commas

        #print(len(raw_item_description))

        # Massage some specific measurements " x " x " -> "x"x"
        # & " x "  -> "x"
        for phrase in raw_item_description:

            pattern1 = r'(\S)\s*x\s*(\S)'

            match1 = re.match(pattern1, phrase, flags=re.IGNORECASE)

            if match1:
                phrase = re.sub(pattern1, r'\1\2', phrase, flags=re.IGNORECASE)

            phrase = phrase.lower()

            unparse_desc = phrase.replace(",", " ")
            unparse_desc = unparse_desc.strip()
            item_descs.append(unparse_desc)

            phraseSplit = phrase.split(",")
            phraseSplit = [x.replace(" ", "")for x in phraseSplit]
            item_descs_set += phraseSplit


        #print(len(item_descs))

        item_descs_set = set(item_descs_set)

        training_keyword_df = pd.DataFrame(columns=["itemID", "itemDesc"])
        #print(len(item_descs))


        #Below is temporary
        if training is not None:
            file_name = "train_keywords.csv"
            file_path = os.path.join("./", file_name)
            if os.path.isfile(file_path):
                # If the file exists, load it into a pandas DataFrame
                try:
                    df = pd.read_csv(file_path)  # Modify this line if your file is in a different format
                    training_keyword_df = df
                    print(f"File '{file_name}' successfully loaded into pandas DataFrame.")
                    #return df
                except Exception as e:
                    print(f"Error loading file '{file_name}' into pandas DataFrame: {str(e)}")
                    #return None
        else:
            file_name = "test_keywords.csv"
            file_path = os.path.join("./", file_name)
            if os.path.isfile(file_path):
                # If the file exists, load it into a pandas DataFrame
                try:
                    df = pd.read_csv(file_path)  # Modify this line if your file is in a different format
                    training_keyword_df = df
                    print(f"File '{file_name}' successfully loaded into pandas DataFrame.")
                    #return df
                except Exception as e:
                    print(f"Error loading file '{file_name}' into pandas DataFrame: {str(e)}")
                    #return None

        raw_item_description = [x.lower() for x in raw_item_description]
        temp_df = pd.Series(raw_item_description, name="itemDesc")
        item_descs = pd.DataFrame({"itemID": items, "itemDesc": item_descs})


        if training_keyword_df.empty:
            training_list = []
            for phrase in item_descs_set:
                if phrase != "" and phrase != " " and phrase != "custom" and len(phrase) < 20 and '*' not in phrase\
                        and '(' not in phrase and ')' not in phrase:
                    item_indexes = temp_df[temp_df.str.contains(phrase)].index
                    item_values = items.iloc[item_indexes].tolist()
                    if len(item_values) != 0:

                        items_cleaned = set(item_values)
                        items_cleaned = list(items_cleaned)

                        items_cleaned = items_cleaned[:2]

                        items_cleaned = ' '.join(str(item) for item in items_cleaned)
                        training_pair = pd.DataFrame({"itemID":[items_cleaned],"itemDesc":[phrase]})
                        training_list.append(training_pair)

            training_keyword_df = pd.concat(training_list)

        if training is not None:
            training_keyword_df.to_csv("./train_keywords.csv", index=False)
        else:
            training_keyword_df.to_csv("./test_keywords.csv", index=False)
        print(training_keyword_df.shape)

        return item_descs, training_keyword_df

    def get_item_to_keywords(self, itemID, is_training=False, is_negative=False):
        if is_training:
            if not is_negative:
                item_desc_entry = self.data.loc[self.data[COLUMN_DICT["column_item"]] == itemID][:1]
                item_desc_input = item_desc_entry[COLUMN_DICT["column_itemdesc"]]
                item_desc_input = item_desc_input.values.tolist()  # THIS IS PROBLEWM FOR EXTRA LOGIC IONN TRAINING

                # item_desc_two_input = item_desc_entry["keywordtwo"]
                # item_desc_two_input = item_desc_two_input.values.tolist() #THIS IS PROBLEWM FOR EXTRA LOGIC IONN TRAINING
            else:
                # logger.info(len(itemID))
                negative_samples_size = len(set(itemID)) * 4  # generate negative samples porpotional to training length

                # Redo get negative -> IF BELOW DOESN"T WORK; Keep As This form
                # Get all items have seen
                # get vocab layer get set of all keywords found in description
                # subtract from main set of keywords
                # randomly select from negative pool same size as item list * num factor
                # return sample

                poss_negative_items = list(set(self.data[COLUMN_DICT['column_item']].to_list()) - set(itemID))

                negative_items_raw = self.data.loc[self.data[COLUMN_DICT["column_item"]].isin(poss_negative_items)]

                # negative_items_bundle = negative_items_raw.drop_duplicates(subset=[COLUMN_DICT["column_item"]])
                negative_items = negative_items_raw[COLUMN_DICT["column_itemdesc"]]

                negative_item_size = len(poss_negative_items)
                negative_required_samples = min(negative_samples_size, negative_item_size)
                negative_samples = random.sample(population=negative_items.values.tolist(), k=negative_required_samples)

                item_desc_input = negative_samples

                # logger.info(item_desc_input)
                # item_desc_input = list(item_desc_input)

        else:
            # Bloew is without TextVector
            all_combs = list(itertools.product(set(itemID), set(self.keywordToItem.values())))
            item_desc_input = pd.DataFrame(data=all_combs, columns=["itemID", "itemDesc"])

        return [item_desc_input]


class Dataset_Handler:
    """ Handles DataFile for Training and Test files"""

    def __init__(self,
                 train_file,
                 test_file=None,
                 file_dir=None,
                 negative_n=4, #default is 4
                 convert_binary=True,
                 seed=None):
        self.train_file = train_file
        self.test_file = test_file
        self.negative_n = negative_n
        self.file_dir = file_dir
        self.convert_binary = convert_binary
        self.seed = seed
        self.test_dataset = None

        self.train_dataset = DataFile(self.train_file, self.convert_binary, file_dir=self.file_dir)
        if self.test_file is not None:
            self.test_dataset = DataFile(self.test_file, self.convert_binary, file_dir=self.file_dir, training=self.train_dataset)

        if self.test_file is not None:
            self.test_full_dataset = np.concatenate((self.train_dataset.data, self.test_dataset.data), axis=0)

            # pivot_df = pd.DataFrame(self.test_full_dataset, columns=["userID", "itemID", "rating"])
            #             # pivot_df = pivot_df.drop_duplicates()
            #             # print("TEST FULL DS: {}".format(pivot_df))  # DEBUG ONLY
            #             # pivot_full = pivot_df.pivot(index="userID", columns="itemID", values='rating')
            #             # pivot_fill = pivot_full.fillna(0)
            #             # sparsity = 1.0 - count_nonzero(pivot_fill.to_numpy()) / pivot_fill.to_numpy().size
            #             # print("TEST Full Sparsity DS: {}".format(sparsity))
            #             # print("TEST Full Users: {}".format(pivot_df["userID"].nunique()))
            #             # print("TEST Full Items: {}".format(pivot_df["itemID"].nunique()))

        self.all_user_item_pairs = np.array(np.meshgrid(np.array(list(self.train_dataset.userToId.values())), np.array(
            list(self.train_dataset.itemToId.values())))).T.reshape(-1, 2).T
        logger.info(self.all_user_item_pairs.shape)

        self.all_item_keyword_pairs = None
        random.seed(self.seed)

    def create_all_item_keyword_pairs(self, items, keywords):
        self.all_item_keyword_pairs = np.array(np.meshgrid(np.array(list(items)), np.array(list(keywords)))).T.reshape(
            -1, 2).T
        logger.info("All Item -Keyword Pairs: {}", self.all_user_item_pairs.shape)

    def get_current_training_lists(self, is_local, start_run=False):

        user_input, item_input, labels = [], [], []

        # logger.info("U: {}", self.train_dataset.userToId.items())
        for user, key in self.train_dataset.userToId.items():
            users_to_items = self.train_dataset.get_user_to_item_data(
                key)  # create method to fetch all pairs of user U to items list
            users_to_items_set = set(users_to_items.to_list())  # Needed for negative sampler loop -original


            for item in users_to_items_set:
                # Append to lists
                user_input.append(key)
                item_input.append(item)
                labels.append(1)

            if not is_local:
                negative_samples_size = len(
                    set(users_to_items)) * self.negative_n  # generate negative samples porpotional to training length

                negative_items = list(
                    set(self.train_dataset.data[COLUMN_DICT['column_item']].to_list()) - set(users_to_items))

                negative_item_size = len(negative_items)
                negative_required_samples = min(negative_samples_size,
                                                negative_item_size)  # foundation is four negative samples OR remaining list of unseen items
                negative_samples = random.sample(population=negative_items, k=negative_required_samples)

                user_input += negative_required_samples * [key]
                item_input += negative_samples

                labels += negative_required_samples * [0]


        return user_input, item_input, labels

    def convert_predictions_to_ids(self, predictions, model_name):
        logger.info(predictions)

        predictions[COLUMN_DICT["column_user"]] = predictions[COLUMN_DICT["column_user"]].map(
            self.train_dataset.idToUser)
        predictions[COLUMN_DICT["column_item"]] = predictions[COLUMN_DICT["column_item"]].map(
             self.train_dataset.idToItem)

        return predictions
