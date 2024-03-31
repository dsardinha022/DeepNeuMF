import argparse
import json

import numpy
import numpy as np
import pandas
import pandas as pd
import tensorflow as tf
from time import time
import os
import logging
import matplotlib.pyplot as plt
from common.dataset import DataFile, Dataset_Handler
from common.model_evaluation import get_ranking_metrics



from recommenders.datasets.python_splitters import python_chrono_split

logger = logging.getLogger(__name__)

from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
import mlflow


#MOVE TO CONSTANTS
METRICS_NAMES = ['precision_k', 'recall_k', 'map_k', 'ndcg_k']
#MOVE TO CONSTANTS ^^^

CUSTOMER_PRECISION_K, CUSTOMER_NDCG_K, CUSTOMER_MAP_K, CUSTOMER_RECALL_k = [], [], [], []
TRAINING_LOSS = []
IS_LOCAL = False


class DeepNeuMF:

    def __init__(
            self,
            source_path=None,
            n_factors=1,
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
        self.metric_logger = None

        if self.source_path is not None:
            source_df = pandas.read_csv(self.source_path)
            training_set, validation_set = self.data_preprocessing(source_df)  # Reformat and remove testr_set

            if not os.path.exists(self.directory_name):
                os.makedirs(self.directory_name)

            train_file = self.directory_name + "train_experimental.csv"
            validation_file = self.directory_name + "validation_experimental.csv"

            training_set.to_csv(train_file)
            validation_set.to_csv(validation_file)

            dataset = Dataset_Handler(train_file, test_file=validation_file)
            self.n_users = dataset.train_dataset.num_users
            self.n_items = dataset.train_dataset.num_items
            self.n_brand = dataset.train_dataset.num_brands
            self.n_keywords = dataset.train_dataset.num_keywords

        gpus = tf.config.list_physical_devices('GPU')
        print(gpus)
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    # from core.yolov4 import YOLO, decode, compute_loss, decode_train
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        model = self.create_model()
        callbacks = self.create_callbacks()

        # AzureMl needs this to search for our active experiment run
        metric_logger_client = MlflowClient()
        if metric_logger_client is not None:
            self.metric_logger = metric_logger_client
            mlflow.start_run()
            active_run = mlflow.active_run()
            self.az_run_id = active_run.data.tags['mlflow.rootRunId']
            logger.info("AZURE RUN: {}".format(self.az_run_id))

        self.fit(model, dataset, callbacks)
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
        item_desc_input = tf.keras.layers.Input(shape=[1], dtype=tf.int32, name="itemDesc")
        brand_input = tf.keras.layers.Input(shape=[1], dtype=tf.int32, name="brand")

        user_gmf_embedding = tf.keras.layers.Embedding(
            self.n_users,
            self.n_factors,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(0, 0.01, self.seed),
            embeddings_regularizer=tf.keras.regularizers.l2(self.regualizer_factor),
            input_length=1
        )(user_input)
        reduced_sum_gmf_p = tf.keras.layers.Flatten()(user_gmf_embedding)
        item_embedding = tf.keras.layers.Embedding(
            self.n_items,
            self.n_factors,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=self.seed),
            embeddings_regularizer=tf.keras.regularizers.l2(self.regualizer_factor),
            input_length=1
        )(item_input)
        reduced_sum_gmf_q = tf.keras.layers.Flatten()(item_embedding)
        gmf_vector = tf.keras.layers.Multiply()([reduced_sum_gmf_p, reduced_sum_gmf_q])

        user_mlp_embedding = tf.keras.layers.Embedding(
            self.n_users,
            int(self.layer_one / 2),
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(0, 0.01, self.seed),
            embeddings_regularizer=tf.keras.regularizers.l2(self.regualizer_factor),
            input_length=1
        )(user_input)
        reduced_sum_mlp_p = tf.keras.layers.Flatten()(user_mlp_embedding)
        item_mlp_embedding = tf.keras.layers.Embedding(
            self.n_items,
            int(self.layer_one / 2),
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=self.seed),
            embeddings_regularizer=tf.keras.regularizers.l2(self.regualizer_factor),
            input_length=1
        )(item_input)
        reduced_sum_gmf_q = tf.keras.layers.Flatten()(item_mlp_embedding)
        mlp_vector = tf.keras.layers.Concatenate()([reduced_sum_mlp_p, reduced_sum_gmf_q])
        for indx in range(len(all_layers)):
            if all_layers[indx] is not None:
                if all_layers[indx] != 0:
                    mlp_vector = tf.keras.layers.Dense(
                        all_layers[indx],
                        activation="relu",
                        kernel_initializer=tf.keras.initializers.VarianceScaling(
                            scale=1.0,
                            mode="fan_avg",
                            distribution="uniform",
                            seed=self.seed,
                        ),
                    )(mlp_vector)

        # Below is Research
        item_desc_mlp_embedding = tf.keras.layers.Embedding(
            self.n_keywords,
            int(self.layer_one / 2),
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=self.seed),
            embeddings_regularizer=tf.keras.regularizers.l2(self.regualizer_factor),
            input_length=1
        )(item_desc_input)
        reduced_sum_item_desc_p = tf.keras.layers.Flatten()(item_desc_mlp_embedding)
        bmlp_vector = tf.keras.layers.Concatenate()([reduced_sum_item_desc_p, reduced_sum_gmf_q])

        for indx in range(len(all_layers)):
            if all_layers[indx] is not None:
               if all_layers[indx] != 0:
                    bmlp_vector = tf.keras.layers.Dense(
                        all_layers[indx],
                        activation="relu",
                        kernel_initializer=tf.keras.initializers.VarianceScaling(
                            scale=1.0,
                            mode="fan_avg",
                            distribution="uniform",
                            seed=self.seed,
                        ),
                    )(bmlp_vector)

        if self.model_type == "gmf":
            final_vector = gmf_vector
        elif self.model_type == "mlp":
            final_vector = mlp_vector
        elif self.model_type == "neumf":
            final_vector = tf.keras.layers.Concatenate()([gmf_vector, mlp_vector])
        elif self.model_type == "research_neumf":
            #final_vector = bmlp_vector
            final_vector = tf.keras.layers.Concatenate()([gmf_vector, bmlp_vector])

        predict_layer = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_avg",
                distribution="uniform",
                seed=self.seed,
            ),
            name="prediction"
            # kernel_initializer='lecun_uniform'

        )(final_vector)

        optimizer = None
        if self.optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)  # same as compat.v1.train.AdamOptimizer
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                optimizer)  # enable mix precision -> only on Keras optimizers
        elif self.optimizer_name == "adamadelta":
            optimizer = tf.compat.v1.train.AdadeltaOptimizer(self.learning_rate)
        elif self.optimizer_name == "rmsprop":
            optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate)
        elif self.optimizer_name == "graddesc":
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer_name == "proxadagrad":
            optimizer = tf.compat.v1.train.ProximalAdagradOptimizer(self.learning_rate)

        inputs = []
        if self.model_type == "gmf" or self.model_type == "mlp" or self.model_type == "neumf":
            inputs = [user_input, item_input]
        elif self.model_type == "research_neumf":
            inputs = [user_input, item_input, item_desc_input]

        model_name = self.model_type.lower() + "_model"
        model = tf.keras.Model(inputs=inputs, outputs=predict_layer, name=model_name)
        model.compile(optimizer=optimizer, loss="binary_crossentropy")

        print(model.summary())
        print("model has been created.")
        return model

    def create_callbacks(self, ):
        callbacks = []
        summary_callback = tf.keras.callbacks.TensorBoard(
            self.directory_name, profile_batch=0)

        callbacks += [summary_callback]
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

    def data_preprocessing(self, input_data):
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

        # For validation set keep only users or items that used in training set
        validation = validation[validation["userID"].isin(training["userID"].unique())]
        validation = validation[validation["itemID"].isin(training["itemID"].unique())]

        return training, validation

    def fit(self, model, dataset, callbacks):
        # Following original paper's code with using Keras to splice batch sets
        time_start = time()
        for epoch in range(self.n_epochs):
            timeOne = time()
            user_input, item_input, item_desc_input, brand_input, labels = dataset.get_current_training_lists(IS_LOCAL)
            #logger.info(user_input)
            #logger.info(item_input)
            item_desc_input = [item for sublist in item_desc_input for item in
                         (sublist if isinstance(sublist, list) else [sublist])]
            logger.info(item_desc_input[:250])
            #logger.info(labels[:100])
            # Original setup for working model is below
            if self.model_type == "gmf" or self.model_type == "mlp" or self.model_type == "neumf":
                history = model.fit([numpy.array(user_input), numpy.array(item_input)],
                                    numpy.array(labels), batch_size=self.batch_size, epochs=1,
                                    callbacks=callbacks, verbose=self.verbose)
            elif self.model_type == "research_neumf":
                history = model.fit([numpy.array(user_input), numpy.array(item_input), numpy.array(item_desc_input)],
                                    [numpy.array(labels)], batch_size=self.batch_size, epochs=1,
                                    verbose=self.verbose)

            # history = model.fit([numpy.array(user_input), numpy.array(item_input), numpy.array(labels)], batch_size=self.batch_size, epochs=1,callbacks=callbacks, verbose=0, shuffle=True)

            timeTwo = time()
            if epoch % self.verbose == 0:
                self.evaluate_model(epoch, model, dataset)
                TRAINING_LOSS.append(history.history['loss'][0])
                print('Epoch [%d] Loss %.4f [%.1f s]'
                      % (epoch, history.history['loss'][0], timeTwo - timeOne))
        time_end = time()

        if self.display_timing_messages:
            print("Model training completed with total time {} seconds".format(time_end - time_start))

    def evaluate_model(self, epoch, model, dataset):
        epoch_all_predictions = self.all_predictions(model, dataset)

        time_start = time()
        print()
        eval_customer, eval_item = get_ranking_metrics(dataset.test_dataset.data, epoch_all_predictions, self.top_k, self.display_timing_messages, IS_LOCAL)

        time_end = time()

        if self.display_timing_messages:
            print('Evaluating model at Epoch %d was %d seconds.' % (epoch, time_end - time_start))

        # Save results to graph
        CUSTOMER_PRECISION_K.append(eval_customer[0])
        CUSTOMER_RECALL_k.append(eval_customer[1])
        CUSTOMER_MAP_K.append(eval_customer[2])
        CUSTOMER_NDCG_K.append(eval_customer[3])


        # Save last of predictions
        if epoch == self.n_epochs - 1:
            file_name = self.directory_name + "DeepNeuMFmodel_allpredictions_experimental.csv"
            epoch_all_predictions = dataset.convert_predictions_to_ids(
                epoch_all_predictions)  # Convert dictionary values back to corresponding ids
            epoch_all_predictions.to_csv(file_name, index=False)

    def predict(self, user, item, misc, model):
        print(user)
        print(misc)
        inputs = []
        if self.model_type == "gmf" or self.model_type == "mlp" or self.model_type == "neumf":
            inputs = [numpy.array(user), numpy.array(item)]
        elif self.model_type == "research_neumf":
            inputs = [numpy.array(user), numpy.array(item), numpy.array(misc)]
        with tf.device('/CPU:0'):
            prediction = model.predict(inputs, batch_size=100, verbose=0)
        return list(prediction.reshape(-1))

    def all_predictions(self, model, dataset):
        # Get all possible prediction combination pairs from training data
        time_one = time()
        cartesian_user_items = dataset.all_user_item_pairs


        # RESEARCH
        #Below takes sixty seconds to complete !!!! REDO
        #prediction_brands = dataset.train_dataset.get_brand_from_items(list(cartesian_user_items[1]))
        # RESEARCH
        # pred_item_desc = []
        # print(cartesian_user_items[1])
        # item_temp = cartesian_user_items[1]
        # for item in item_temp:
        #     logger.info(item)
        item_desc = dataset.train_dataset.get_item_to_keywords(cartesian_user_items[1])
        #     pred_item_desc.append(item_desc)


        #predictions = list(self.predict(cartesian_user_items[0], cartesian_user_items[1], prediction_brands, model))
        predictions = list(self.predict(cartesian_user_items[0], cartesian_user_items[1], item_desc, model))

        # all_predictions = pd.DataFrame(
        #     data={"userID": cartesian_user_items[0], "itemID": cartesian_user_items[1], "prediction": predictions})

        all_predictions = pd.DataFrame(
            data={"userID": cartesian_user_items[0], "itemID": cartesian_user_items[1],"itemDesc": item_desc, "prediction": predictions})
        time_two = time()

        if self.display_timing_messages:
            print("Calculation for all predictions was {} seconds.".format(time_two - time_one))
        return all_predictions

    def graph_metrics_through_epochs(self, evaluation_name):



        print("Generating graphs . . .")
        figure, axis = plt.subplots(2, 1)
        axis[0].plot(TRAINING_LOSS, label="Training Loss")
        axis[0].set_title("Training Loss over Epochs")
        axis[0].legend(loc="upper right")

        for metric_name, metric_values in zip(METRICS_NAMES, [CUSTOMER_PRECISION_K, CUSTOMER_RECALL_k, CUSTOMER_MAP_K,
                                                              CUSTOMER_NDCG_K]):
            axis[1].plot(metric_values, label='{}@15'.format(metric_name))
            if self.metric_logger is not None:
                metrics = [Metric(key='{}@15'.format(metric_name), value=val, timestamp=int(time() * 1000), step=0) for val in
                           metric_values]
                self.metric_logger.log_batch(self.az_run_id, metrics=metrics)

        axis[1].set_title("Customer Metrics over Epochs")
        axis[1].legend(loc="upper left")

        if self.metric_logger is not None:
            self.metric_logger.log_figure(self.az_run_id, figure, evaluation_name + "_graph.png")

        plt.tight_layout() #Prevents squished graphs
        #plt.subplots_adjust(wspace=1, hspace=0.125)
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(self.directory_name, evaluation_name + "_graph.png"))
        plt.show()
        print("Graphs complete!")

## DO NOT REMOVE BELOW
def get_command_args():
    parser = argparse.ArgumentParser(description="Running keras model. . .")
    parser.add_argument('--source_path', type=str, default='Data/',
                        help='Dataset file path')
    parser.add_argument('--n_factors', type=int, default=4,
                        help='Number of N-factors to build model with')
    parser.add_argument('--seed', type=int, default=37,
                        help='Set randomness for model')
    parser.add_argument('--layer_one', type=int, default=16,
                        help='First Layer of MLP')
    parser.add_argument('--layer_two', type=int, default=8,
                        help='Second Layer of MLP')
    parser.add_argument('--layer_three', type=int, default=4,
                        help='Third Layer of MLP')
    parser.add_argument('--layer_four', type=int, default=0,
                        help='Last Layer of MLP (Optional)')
    parser.add_argument('--batch_size', type=int, default='128',
                        help="Size per batch")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for model (Default -> 0.001)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of Epochs')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbose')
    parser.add_argument('--optimizer_name', type=str, default='proxadagrad',
                        help='Set the optimizer: adam, proxadagrad')
    parser.add_argument('--model_type', type=str, default="neumf",
                        help='Type of model to run')
    parser.add_argument('--training_ratio', type=float, default=0.75,
                        help='Split ration for training and validation sets')
    parser.add_argument('--top_k', type=int, default=15,
                        help='Top Number of K Samples for Model Evaluation')
    parser.add_argument('--regualizer_factor', type=float, default=0.01,
                        help='Regualizing factor for normalizing values')
    parser.add_argument('--display_debug_messages', type=bool, default=1,
                        help='Display Debugging Messages')
    parser.add_argument('--display_timing_messages', type=bool, default=1,
                        help='Display Logic Timing Messages')
    return parser.parse_args()

# Initialized Model for training in AzureML
parser = get_command_args()
model = DeepNeuMF(
            source_path=parser.source_path,
            n_factors=parser.n_factors,
            seed=parser.seed,
            layer_one=parser.layer_one,
            layer_two=parser.layer_two,
            layer_three=parser.layer_three,
            layer_four=parser.layer_four,
            batch_size=parser.batch_size,
            learning_rate=parser.learning_rate,
            epochs=parser.epochs,
            verbose=parser.verbose,
            optimizer_name=parser.optimizer_name,
            model_type=parser.model_type,
            training_ratio=parser.training_ratio,
            top_k=parser.top_k,
            regualizer_factor=parser.regualizer_factor,
            display_debug_messages=parser.display_debug_messages,
            display_timing_messages=parser.display_timing_messages
            )


