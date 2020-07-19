import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

predictions_1_path = r"results/probabilies-predictions.csv"
# predictions_2_path = r"results/pump2-predictions.csv"
predictions_2_path = r"results/merged-predictions.csv"
predictions_3_path = r"results/pump3-predictions.csv"
# test = r"preprocessed_test.csv"

test1 = pd.read_csv(predictions_1_path)
test2 = pd.read_csv(predictions_2_path)
test3 = pd.read_csv(predictions_3_path)

test1.columns = ['idd', 'status_group']
test2.columns = ['idd', 'status_group']
test3.columns = ['idd', 'status_group']


weight1 = 0.8079
weight2 = 0.8064
# weight2 = 0.7318
weight3 = 0.8079
ids = test1.idd

vals_to_replace = {'functional' : 2, 'functional needs repair':1,
                   'non functional' : 0}

test1.status_group = test1.status_group.replace(vals_to_replace)
# test2.status_group = test2.status_group.replace(vals_to_replace)
test3.status_group = test3.status_group.replace(vals_to_replace)

test1 = test1.status_group
# test2 = test2.status_group
test3 = test3.status_group



merged_group = []

for index in range(len(test1)):
    cevap = [0]*3
    cevap[test1[index]] += weight1
    # cevap[test2[index]] += weight2
    cevap[test3[index]] += weight3
    # print(max(cevap))
    max_index = cevap.index(max(cevap))
    # if test1[index] != test3[index]:
    #     print(index ,test1[index] , test3[index])
    merged_group.append(max_index)
    # print(cevap)

vals_to_replace2 = {2: 'functional', 1: 'functional needs repair',
                   0: 'non functional'}

data = {'ID': ids, 'status_group': merged_group}

submit = pd.DataFrame(data=data)

submit.status_group = submit.status_group.replace(vals_to_replace2)
# submit.to_csv('results/merged-predictions.csv', index=False)

# test = pd.read_csv(test)
# print()
