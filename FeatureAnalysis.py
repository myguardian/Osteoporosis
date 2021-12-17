import logging
import sys
import numpy as np
import statistics
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


# In[17]:


def get_analysis(path):
    # replace with actual csv
    data = pd.read_csv(path)
    
    number_of_rows = len(data.index)
    print('Number of Rows: ' + str(number_of_rows) + '\n\n') 
    print('Number of Missing Values per Column: ' + str(number_of_rows) + '\n\n')
    print(data.isnull().sum())  
    print('\n\n')

    data = data.dropna()    

    features = list(data.columns.values)

    # In[23]:

    y = data['bmdtest_tscore_fn']
    for feature in features:
        x = data[feature]

        plt.title(feature + " Correlation with BMD Test Score")
        plt.xlabel(feature)
        plt.ylabel('bmdtest_tscore_fn')
        plt.scatter(x, y)
        plt.savefig(str(sys.argv[2]) + '\\' + f'{feature}')
        
        print(feature + " Analysis\n")
        print('Number of Empty: ' + str(len(data.index)))  

        if pd.to_numeric(x, errors='coerce').notnull().all():
            print("Standard Deviation of the sample is % s " % (statistics.stdev(x)))
            print("Mean of the sample is % s \n\n" % (statistics.mean(x)))
            print('Max: ' + str(x.max()))
            print('Min: ' + str(x.min()))
        else:
            counter = Counter(x)
            print(counter)
            print('\n\n')


if __name__ == "__main__":
    try:
        file_name = sys.argv[1]
        get_analysis(file_name)
    except ValueError as e:
        logging.error(e)
    






