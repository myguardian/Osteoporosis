import logging
import math
import os
import shutil
import sys
from collections import Counter
from glob import glob

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import cycle
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from numpy.random import seed
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import shap
import seaborn as sns

print(tf.__version__)


def set_directory():
    # detect the current working directory and add the sub directory
    main_path = os.getcwd()
    absolute_path = main_path + "Output/Classifier_results"
    try:
        os.mkdir(absolute_path)
    except OSError:
        logging.info("Creation of the directory %s failed. Folder already exists." % absolute_path)
    else:
        logging.info("Successfully created the directory %s " % absolute_path)


def build_and_compile_model(normalizer, target):
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]
    model = keras.Sequential([
        normalizer,
        Dense(32, activation='relu', input_shape=(11,)),
        # Dropout(0.10),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=METRICS)

    return model


def normalize_data(data):
    normalizer = tf.keras.layers.Normalization()


def encode_cat_data(data, features):
    dataset = data.copy()

    for feature in features:
        cat_one_hot = pd.get_dummies(dataset[feature], prefix=f'{feature}', drop_first=False)
        dataset = dataset.drop(feature, axis=1)
        dataset = dataset.join(cat_one_hot)

    return dataset


def plot_metrics(history):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'prc', 'precision', 'recall', ]
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])
        plt.legend()
    plt.savefig(f'Output/Classifier_results/DNN_Metrics.png')
    plt.clf()


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(f'Output/Classifier_results/conf_mat.png')
    plt.clf()


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(fp, tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def write_results_to_file(file, results):
    with open(file, 'w+') as f:
        result_names = ['Loss', 'True Positives', 'False Positives', 'True Negatives', 'False Negatives',
                        'Accuracy', 'Precision', 'Recall', 'AUC', 'PRC']
        index = 0
        for result_name in result_names:
            f.write(f'{result_name}: {results[index]}\n')
            index = index + 1


def create_shap_sample(tr_data, num_of_instances):
    sample = shap.utils.sample(tr_data, num_of_instances, random_state=120)
    return sample


def create_explainer(dnn_model, sample):
    model_explainer = shap.Explainer(dnn_model.predict, sample)
    return model_explainer


def plot_summary(model_explainer, shap_data, feature_names):
    shap_values = model_explainer(shap_data)
    shap.summary_plot(shap_values, shap_data, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f'Output/Classifier_results/shap_summary.png', )
    plt.clf()


if __name__ == "__main__":
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    try:
        # Get the data from the argument
        file_name = sys.argv[1]
        # file_name = 'Clean_Data_Main.csv'
        logging.info(f'Loading Data {file_name}\n')

        # Create the directory where the CSV files and images are going to be saved
        set_directory()

        # Get the Data
        data = pd.read_csv(file_name)

        # One-hot Encode the Data
        data = encode_cat_data(data, ['parentbreak', 'alcohol',
                                      'arthritis', 'diabetes', 'heartdisease',
                                      'oralster', 'smoke', 'bmdtest_10yr_caroc'])

        # Split the Data to Features and target
        X = data[['PatientAge', 'PatientGender', 'bmdtest_height', 'bmdtest_weight', 'parentbreak_1.0', 'alcohol_1.0',
                  'arthritis_1.0', 'diabetes_1.0', 'heartdisease_1.0', 'oralster_1.0', 'smoke_1.0']]
        y = data['bmdtest_10yr_caroc_2.0']

        # Normalize the Data
        norm = tf.keras.layers.Normalization(axis=-1)
        norm.adapt(X)

        # Split the data into Training and Test sets
        train_data, test_data, train_targets, test_targets = train_test_split(X, y, test_size=0.20, random_state=20)

        # This is just for Debugging purposes
        print(Counter(train_targets))

        np.random.seed(10)
        tf.random.set_seed(1)
        # Build the Model
        model = build_and_compile_model(norm, y)

        # Train the Model
        history = model.fit(train_data, train_targets, batch_size=32, epochs=100, verbose=1, validation_split=0.2)

        # Evaluate the model
        score = model.evaluate(test_data, test_targets, verbose=0)
        train_predictions = model.predict(train_data)
        test_predictions = model.predict(test_data)

        # Plot and visualize the data
        plot_metrics(history)

        # Plot a Confusion Matrix
        plot_cm(test_targets, test_predictions)

        # Plot the ROC Curves
        plot_roc('DNN_32_16_Train', train_targets, train_predictions, color=colors[0])
        plot_roc('DNN_32_16_Test', test_targets, test_predictions, color=colors[1], linestyle='--')
        plt.legend(loc='lower right')
        plt.savefig(f'Output/Classifier_results/ROC.png')
        plt.clf()

        filename = "Output/Classifier_results/DNN_Classifier_results.txt"
        write_results_to_file(filename, score)

        data_sample = create_shap_sample(train_data, int(len(train_data) * 0.20))
        print(data_sample.head())

        explainer = create_explainer(model, data_sample)
        plot_summary(explainer, data_sample,
                     ['PatientAge', 'PatientGender', 'bmdtest_height', 'bmdtest_weight', 'parentbreak_1.0',
                      'alcohol_1.0', 'arthritis_1.0', 'diabetes_1.0', 'heartdisease_1.0', 'oralster_1.0',
                      'smoke_1.0'])




    except ValueError as e:
        logging.error(e)
        logging.error('Unable to load the CSV File')
