import os
import shutil
import sys

import logging


def set_directory(temp_path):
    # detect the current working directory and add the sub directory
    main_path = os.getcwd()
    absolute_path = main_path + temp_path
    try:
        os.mkdir(absolute_path)
    except OSError:
        logging.error("Creation of the directory %s failed. Folder already exists." % absolute_path)
    else:
        logging.info("Successfully created the directory %s " % absolute_path)


def move_results():
    cwd = os.getcwd()
    dst_dir = cwd + "/Output/models_results"
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

        set_directory('/Output')
        set_directory('/Output/models_results')

        for i in range(1, 6):
            set_directory(f'/Output/models_results/scikit_model_{i}')
            set_directory(f'/Output/models_results/scikit_model_{i}/waterfalls')

        scripts = ['/BR.py', '/LR.py', '/RR.py', '/RFR.py', '/CB.py']

        for script in scripts:
            os.system("python " + '"' + script_dir + '"' + script + f" {file_name}")

        move_results()

        print('All Scripts have completed. Closing Program.')

    except ValueError as e:
        logging.error(e)
        logging.error('Unable to load the CSV File')
