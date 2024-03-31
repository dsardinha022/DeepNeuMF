import ast
import itertools

import numpy
import numpy as np
import pandas
import pandas as pd
import tensorflow as tf
from time import time
import os
import logging
import matplotlib.pyplot as plt
from .common.dataset import DataFile, Dataset_Handler

from .common.model_evaluation import get_ranking_metrics
from recommenders.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k,
                                                       precision_at_k,
                                                       recall_at_k, get_top_k_items)

from .common.constants import (COLUMN_DICT)


from recommenders.datasets.python_splitters import python_chrono_split


logger = logging.getLogger(__name__)

# MOVE TO CONSTANTS
METRICS_NAMES = ['precision_k', 'recall_k', 'map_k', 'ndcg_k', 'mrr']
# MOVE TO CONSTANTS ^^^

CUSTOMER_PRECISION_K, CUSTOMER_NDCG_K, CUSTOMER_MAP_K, CUSTOMER_RECALL_k, CUSTOMER_MRR = dict(), dict(), dict(), dict(), dict()
KEYWORD_PRECISION_K, KEYWORD_NDCG_K, KEYWORD_MAP_K, KEYWORD_RECALL_k = [], [], [], []
TRAINING_LOSS = dict()
TRAINING_ACC = dict()
VALID_ACC = dict()
VALID_LOSS = dict()
IS_LOCAL = False  # Mainly Handles Negative Sampling for Binary cross entropy


class DeepNeuMF:

    def __init__(
            self,
            source_path=None,
            n_factors=1,
            input_keywords=3,
            seed=None,
            layer_one=16,
            layer_two=8,
            layer_three=4,
            layer_four=None,
            batch_size=32,
            learning_rate=0.001,
            epochs=30,
            verbose=1,
            optimizer_name="adam",
            model_type="neumf",
            training_ratio=0.75,
            top_k=15,
            regualizer_factor=0.01,
            display_debug_messages=False,
            display_timing_messages=True
    ):
        """Constructor
               Args:
        """
        self.checkpoint_directory_name = None
        self.source_path = source_path
        self.n_factors = n_factors
        self.n_input_keywords = input_keywords
        self.seed = seed
        self.layer_one = layer_one
        self.layer_two = layer_two
        self.layer_three = layer_three
        self.layer_four = layer_four
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = epochs
        self.verbose = verbose
        self.optimizer_name = optimizer_name
        self.model_type = model_type
        self.training_ratio = training_ratio
        self.top_k = top_k
        self.regualizer_factor = regualizer_factor
        self.display_debug_messages = display_debug_messages
        self.display_timing_messages = display_timing_messages

        self.train_name = "DeepNeuMF-Keras-EXP-" + self.optimizer_name + "-" + str(
            self.training_ratio) + "-" + str(self.n_epochs) + "-[" + str(self.layer_one) + "," + str(
            self.layer_two) + "," + str(self.layer_three) + "," + str(self.layer_four) + "]-BTCH" + str(
            self.batch_size) + "-LRN" + str(self.learning_rate) + "-" + "{}".format(time())

        self.directory_name = "checkpoints/" + self.train_name + "/"

        if self.source_path is not None:
            source_df = pandas.read_csv(self.source_path)
            training_set, validation_set = self.data_preprocessing(source_df,
                                                                   "collaborative")  # Reformat and remove testr_set

            if not os.path.exists(self.directory_name):
                os.makedirs(self.directory_name)

            train_file = self.directory_name + "train_experimental.csv"
            validation_file = self.directory_name + "validation_experimental.csv"

            training_set.to_csv(train_file)
            validation_set.to_csv(validation_file)

            self.dataset = Dataset_Handler(train_file, test_file=validation_file, file_dir=self.directory_name)

            self.vocab_layer = None
            self.label_one_hot = None
            self.n_keywords = None

            self.n_users = self.dataset.train_dataset.num_users
            self.n_items = self.dataset.train_dataset.num_items

        #gpus = tf.config.list_physical_devices('GPU')
        # print(gpus)
        # if gpus:
        #     try:
        #         # Currently, memory growth needs to be the same across GPUs
        #         for gpu in gpus:
        #             tf.config.experimental.set_memory_growth(gpu, True)
        #             # from core.yolov4 import YOLO, decode, compute_loss, decode_train
        #         logical_gpus = tf.config.list_logical_devices('GPU')
        #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        #     except RuntimeError as e:
        #         # Memory growth must be set before GPUs have been initialized
        #         print(e)
        model = self.create_model()
        callbacks = self.create_callbacks()
        self.binacc_m = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
        self.tf_cs = tf.keras.metrics.CosineSimilarity()

        self.fit(model, self.dataset, callbacks)
        self.graph_metrics_through_epochs("Validation Results")
        # print("init has finished...")

    def create_model(
            self
    ):
        """
        Generic Model Layout
        :return: model
        """
        """Model Layers"""

        all_layers = [self.layer_one, self.layer_two, self.layer_three, self.layer_four]

        user_input = tf.keras.layers.Input(shape=[1], dtype=tf.int32, name="userID")
        item_input = tf.keras.layers.Input(shape=[1], dtype=tf.int32, name="itemID")


        user_gmf_embedding = tf.keras.layers.Embedding(
            self.n_users,
            self.n_factors,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(0, 0.01, self.seed),
            input_length=1
        )(user_input)
        reduced_sum_gmf_p = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(user_gmf_embedding)
        flat_gmf_p = tf.keras.layers.Flatten()(reduced_sum_gmf_p)

        item_gmf_embedding = tf.keras.layers.Embedding(
            self.n_items,
            self.n_factors,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=self.seed),
            input_length=1
        )(item_input)
        reduced_sum_gmf_q = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(item_gmf_embedding)
        flat_gmf_q = tf.keras.layers.Flatten()(reduced_sum_gmf_q)

        gmf_vector = tf.keras.layers.Multiply()([flat_gmf_p, flat_gmf_q])

        user_mlp_embedding = tf.keras.layers.Embedding(
            self.n_users,
            int(self.layer_one / 2),
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(0, 0.01, self.seed),
            input_length=1
        )(user_input)
        reduced_sum_mlp_p = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(user_mlp_embedding)
        flat_mlp_p = tf.keras.layers.Flatten()(reduced_sum_mlp_p)

        item_mlp_embedding = tf.keras.layers.Embedding(
            self.n_items,
            int(self.layer_one / 2),
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=self.seed),
            input_length=1
        )(item_input)
        reduced_sum_mlp_q = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(item_mlp_embedding)
        flat_mlp_q = tf.keras.layers.Flatten()(reduced_sum_mlp_q)

        mlp_vector = tf.keras.layers.Concatenate(axis=1)([flat_mlp_p, flat_mlp_q])

        final_dmf_vector = None
        if self.model_type == "dmf" or self.model_type == "dnmf":
            user_dmf_vector = tf.keras.layers.Embedding(
                self.n_users,
                self.layer_one,
                embeddings_initializer=tf.keras.initializers.TruncatedNormal(0, 0.01, self.seed),
                input_length=1
            )(user_input)

            item_dmf_vector = tf.keras.layers.Embedding(
                self.n_items,
                self.layer_one,
                embeddings_initializer=tf.keras.initializers.TruncatedNormal(0, 0.01, self.seed),
                input_length=1
            )(item_input)


            concat_list = []
            for indx in range(len(all_layers)):
                if all_layers[indx] is not None:
                    if all_layers[indx] != 0:
                        user_dmf_vector = tf.keras.layers.Dense(
                            all_layers[indx],
                            activation="relu",
                            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                scale=1.0,
                                mode="fan_out",
                                distribution="truncated_normal",
                                seed=self.seed,
                            ),

                        )(user_dmf_vector)

                        item_dmf_vector = tf.keras.layers.Dense(
                            all_layers[indx],
                            activation="relu",
                            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                scale=1.0,
                                mode="fan_out",
                                distribution="truncated_normal",
                                seed=self.seed,
                            ),

                        )(item_dmf_vector)

                        concat_layer = tf.keras.layers.Concatenate(axis=1)([user_dmf_vector, item_dmf_vector])
                        concat_layer = tf.keras.layers.Dense(
                            all_layers[indx],
                            activation="relu",

                        )(concat_layer)
                        concat_list.append(concat_layer)

            final_dmf_vector = tf.keras.layers.Concatenate(axis=2)(concat_list)
            final_dmf_vector = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(final_dmf_vector)
            final_dmf_vector = tf.keras.layers.Flatten()(final_dmf_vector)

        for indx in range(len(all_layers)):
            if all_layers[indx] is not None:
                if all_layers[indx] != 0:
                    mlp_vector = tf.keras.layers.Dense(
                            all_layers[indx],
                            activation="relu"
                    )(mlp_vector)


        if self.model_type == "gmf":
            final_vector = gmf_vector
        elif self.model_type == "mlp":
            final_vector = mlp_vector
        elif self.model_type == "neumf":
            final_vector = tf.keras.layers.Concatenate(axis=1)([gmf_vector, mlp_vector])
        elif self.model_type == "dmf":
            final_vector = final_dmf_vector
        elif self.model_type == "ndmf":
            final_vector = tf.keras.layers.Concatenate(axis=1)([final_dmf_vector, gmf_vector])


        predict_layer = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            use_biasF=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_avg",
                distribution="uniform",
                seed=self.seed,
            ),
            name="prediction"

        )(final_vector)

        optimizer = None
        if self.optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)  # same as compat.v1.train.AdamOptimizer
        elif self.optimizer_name == "adamadelta":
            optimizer = tf.keras.optimizers.Adadelta(self.learning_rate)
        elif self.optimizer_name == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
        elif self.optimizer_name == "graddesc":
            optimizer = tf.keras.optimizers.SGD(self.learning_rate)
        elif self.optimizer_name == "proxadagrad":
            optimizer = tf.compat.v1.train.ProximalAdagradOptimizer(self.learning_rate)

        if self.optimizer_name == "graddesc" or self.optimizer_name == "rmsprop" or self.optimizer_name == "adam" or self.optimizer_name == "adamadelta":
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                optimizer)  # enable mix precision -> only on Keras optimizers

        inputs = [user_input, item_input]

        model_name = self.model_type.lower() + "_model"
        model = tf.keras.Model(inputs=inputs, outputs=predict_layer, name=model_name)
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.CosineSimilarity(),
                               tf.keras.metrics.BinaryAccuracy()])

        print(model.summary())
        print("model has been created.")
        return model

    def create_callbacks(self, ):
        callbacks = []
        summary_callback = tf.keras.callbacks.TensorBoard(
            self.directory_name, profile_batch=0)
        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=1)

        callbacks += [summary_callback, earlystop_callback]
        return callbacks

    def save(self, ):
        print("Saved was performed correctly.")

    def load(self, gmf_path=None):
        """Load model parameters for further use.
        """
        model = None
        if gmf_path != None:
            gmf_path = os.path.join(self.directory_name, gmf_path)
            print(gmf_path)
            model = self.create_model()
            model.load_weights(gmf_path)

        print("model has been loaded.")
        return model

    def data_preprocessing(self, input_data, type):
        """
        Splits Data
        """
        training, validation = [], []
        if self.training_ratio == 0.8:
            training, validation = python_chrono_split(input_data, [0.80, 0.2])
        elif self.training_ratio == 0.75:
            training, validation = python_chrono_split(input_data, [0.75, 0.25])
        elif self.training_ratio == 0.5:
            training, validation = python_chrono_split(input_data, [0.5, 0.5])
        elif self.training_ratio == 0.25:
            (training, validation) = python_chrono_split(input_data, [0.25, 0.75])

        assert len(training) != 0, "Training data can not be empty."

        if type == "collaborative":
            # For validation set keep only users or items that used in training set
            validation = validation[validation["userID"].isin(training["userID"].unique())]
            validation = validation[validation["itemID"].isin(training["itemID"].unique())]

        return training, validation

    def fit(self, model, dataset, callbacks):
        time_start = time()
        for epoch in range(self.n_epochs):
            timeOne = time()

            start_run = False
            if epoch == 0:
                start_run = True
            user_input, item_input, labels = dataset.get_current_training_lists(
                IS_LOCAL, start_run)

            history = model.fit([numpy.array(user_input), numpy.array(item_input)],
                            numpy.array(labels), batch_size=self.batch_size, epochs=1,
                            callbacks=callbacks, verbose=self.verbose)

            timeTwo = time()
            if epoch % self.verbose == 0:
                self.evaluate_model(epoch, model, dataset)

                TRAINING_LOSS[epoch] = history.history['loss'][0]
                print('Epoch [%d] Loss %.4f [%.1f s]'
                      % (epoch, history.history['loss'][0], timeTwo - timeOne))
        time_end = time()
        if self.display_timing_messages:
            print("Model training completed with total time {} seconds".format(time_end - time_start))

    def evaluate_model(self, epoch, model, dataset):
        epoch_all_predictions = self.all_predictions(model, dataset, epoch)

        time_start = time()
        eval_customer = []

        merged = pd.merge(dataset.train_dataset.data, epoch_all_predictions, on=["userID", "itemID"], how="outer")
        epoch_all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

        eval_precision = precision_at_k(dataset.test_dataset.data, epoch_all_predictions,
                                        col_prediction='prediction', k=self.top_k)
        eval_customer.append(eval_precision)
        eval_recall = recall_at_k(dataset.test_dataset.data, epoch_all_predictions, col_prediction='prediction',
                                  k=self.top_k)
        eval_customer.append(eval_recall)
        eval_map = map_at_k(dataset.test_dataset.data, epoch_all_predictions, col_prediction='prediction', k=self.top_k)
        eval_customer.append(eval_map)
        eval_ndcg = ndcg_at_k(dataset.test_dataset.data, epoch_all_predictions, col_prediction='prediction',
                              k=self.top_k)
        eval_customer.append(eval_ndcg)
        eval_customer.append(0)  # mrr

        time_end = time()

        if self.display_timing_messages:
            print('Evaluating model at Epoch %d was %d seconds.' % (epoch, time_end - time_start))

        # Save results to graph -> Standard user x item
        CUSTOMER_PRECISION_K[epoch] = eval_customer[0]
        CUSTOMER_RECALL_k[epoch] = eval_customer[1]
        CUSTOMER_MAP_K[epoch] = eval_customer[2]
        CUSTOMER_NDCG_K[epoch] = eval_customer[3]
        CUSTOMER_MRR[epoch] = eval_customer[4]

        # Save last of predictions
        if epoch == self.n_epochs - 1:
            file_name = self.directory_name + "deepneumfmodel_allpredictions_experimental.csv"
            epoch_all_predictions = dataset.convert_predictions_to_ids(
                epoch_all_predictions, self.model_type)  # Convert dictionary values back to corresponding ids
            epoch_all_predictions.to_csv(file_name, index=False)

    def predict(self, user, item, misc, model):
        print(user)

        inputs = []
        predict_inputs = [numpy.array(user), numpy.array(item)]

        inputs = predict_inputs
        with tf.device('/CPU:0'):
            # print("BEFORE PRED {}".format(len(item_desc_input_frame)))
            prediction = model.predict(inputs, batch_size=1000, verbose=0)  # batch_size=1000
            print("PRED {}".format(prediction))
        return list(prediction.reshape(-1))  # for collaborative purposes
        # return list(prediction)

    def all_predictions(self, model, dataset, epoch):
        # Get all possible prediction combination pairs from training data
        time_one = time()
        cartesian_user_items = dataset.all_user_item_pairs

        # if self.model_type == "research_neumf":
        #
        #     #pred_entry = dataset.all_user_item_timestamp_pairs
        #     pred_entry = dataset.all_user_item_pairs
        #
        #     temp_column = np.array([[101]] * pred_entry.shape[1]) # create a 2D array with the same number of rows as a and one column with the value 7
        #     temp_column = temp_column.T
        #
        #     pred_entry = np.append(pred_entry, temp_column, axis=0)
        #     #print(temp2[2])
        #
        #
        #     predictions = list(self.predict(pred_entry[0], pred_entry[1], pred_entry[2], model))
        #
        #     all_predictions = pd.DataFrame(data={'userID': pred_entry[0], 'itemID': pred_entry[1], 'timestamp': pred_entry[2]})
        #     all_predictions["prediction"] = predictions
        #
        #     print(all_predictions)
        #
        # else:
        predictions = list(self.predict(cartesian_user_items[0], cartesian_user_items[1], None, model))
        print(len(cartesian_user_items[0]))
        print(len(predictions))
        all_predictions = pd.DataFrame(
            data={"userID": cartesian_user_items[0], "itemID": cartesian_user_items[1],
                  "prediction": predictions})

        time_two = time()

        if self.display_timing_messages:
            print("Calculation for all predictions was {} seconds.".format(time_two - time_one))
        return all_predictions


    def graph_metrics_through_epochs(self, evaluation_name):
        print("Generating graphs . . .")
        figure, axis = plt.subplots(2, 1)
        axis[0].plot(TRAINING_LOSS.keys(), TRAINING_LOSS.values(), label="Training Loss", color='orange')

        axis[0].set_title("Loss over Epochs")
        axis[0].legend(loc="upper right")

        for metric_name, metric_values in zip(METRICS_NAMES, [CUSTOMER_PRECISION_K, CUSTOMER_RECALL_k, CUSTOMER_MAP_K,
                                                              CUSTOMER_NDCG_K]):
            axis[1].plot(metric_values.keys(), metric_values.values(), label='{}@K'.format(metric_name))
            # axis[1].label()
            # axis[1].plot(metric_values, label=)
        axis[1].set_title("Customer Metrics over Epochs")
        axis[1].legend(loc="upper left")

        plt.tight_layout()  # Prevents squished graphs
        # plt.subplots_adjust(wspace=1, hspace=0.125)
        plt.xlabel("Epochs")

        f2 = figure.add_axes([0.1, 1, 1, 0.6])
        f2.set_title("MRR Each Iteration")
        f2.plot(CUSTOMER_MRR.keys(), CUSTOMER_MRR.values(), label='MRR Eval')
        # f2.label()
        # f2.plot(CUSTOMER_MRR, label='MRR Eval')

        plt.savefig(os.path.join(self.directory_name, evaluation_name + "_graph.png"))
        plt.show()
        print("Graphs complete!")

        print("Eval NDCG@{}: {}".format(self.top_k, CUSTOMER_NDCG_K[self.n_epochs - 1]))
        print("Eval Precision@{}: {}".format(self.top_k, CUSTOMER_PRECISION_K[self.n_epochs - 1]))
        print("Eval Recall@{}: {}".format(self.top_k, CUSTOMER_RECALL_k[self.n_epochs - 1]))
        print("Eval Map@{}: {}".format(self.top_k, CUSTOMER_MAP_K[self.n_epochs - 1]))

        print("All NDCG@{}: {}".format(self.top_k, CUSTOMER_NDCG_K))
        print("All Precision@{}: {}".format(self.top_k, CUSTOMER_PRECISION_K))
        print("All Recall@{}: {}".format(self.top_k, CUSTOMER_RECALL_k))
        print("All Map@{}: {}".format(self.top_k, CUSTOMER_MAP_K))
