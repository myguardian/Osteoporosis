import os
import shutil
import sys

import logging


def set_directory():
    # detect the current working directory and add the sub directory
    main_path = os.getcwd()
    absolute_path = main_path + "/models_results"
    model1_path = absolute_path + "/scikit_model_1"
    model2_path = absolute_path + "/scikit_model_2"
    model3_path = absolute_path + "/scikit_model_3"
    model4_path = absolute_path + "/scikit_model_4"
    model1_waterfall = model1_path + "/waterfalls"
    model2_waterfall = model2_path + "/waterfalls"
    model3_waterfall = model3_path + "/waterfalls"
    model4_waterfall = model4_path + "/waterfalls"
    try:
        if not os.path.exists(absolute_path):
            os.mkdir(absolute_path)
        if not os.path.exists(model1_path):
            os.mkdir(model1_path)
        if not os.path.exists(model2_path):
            os.mkdir(model2_path)
        if not os.path.exists(model3_path):
            os.mkdir(model3_path)
        if not os.path.exists(model4_path):
            os.mkdir(model4_path)
        if not os.path.exists(model1_waterfall):
            os.mkdir(model1_waterfall)
        if not os.path.exists(model2_waterfall):
            os.mkdir(model2_waterfall)
        if not os.path.exists(model3_waterfall):
            os.mkdir(model3_waterfall)
        if not os.path.exists(model4_waterfall):
            os.mkdir(model4_waterfall)

    except OSError:
        logging.info("Creation of the directory %s failed. Folder already exists." % absolute_path)
        # try:
    else:
        logging.info("Successfully created the directory %s " % absolute_path)


def move_results():
    cwd = os.getcwd()
    dst_dir = cwd + "/models_results"
    try:
        shutil.move(os.path.join(cwd, 'SciKit_Model_Results.txt'),
                    os.path.join(dst_dir, 'SciKit_Model_Results.txt'))

        print('Model results have been moved successfully.')
    except Exception as er:
        logging.error(er)
        logging.error("There was an error when moving the files")


def get_object_type(obj):
    return type(obj)


if __name__ == "__main__":
    try:
        current_dir = os.getcwd()
        script_dir = current_dir + '/SciKit_Scripts'
        # Get the data from the argument
        file_name = sys.argv[1]
        logging.info(f'Loading Data {file_name}\n')
        set_directory()
        scripts = ['/BR.py', '/LR.py', '/RR.py', '/RFR.py']

        for script in scripts:
            os.system("python " + '"' + script_dir + '"' + script + f" {file_name}")

        move_results()

        print('All Scripts have completed. Closing Program.')

    except ValueError as e:
        logging.error(e)
        logging.error('Unable to load the CSV File')
