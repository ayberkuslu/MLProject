import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

train = r"train.csv"
test = r"test.csv"

train = pd.read_csv(train)
test = pd.read_csv(test)
train_labels = pd.read_csv("labels.csv")
# print(train_labels)

# for row in train_labels:
  # print(train_labels[row])
train = pd.concat([train, train_labels], axis=0)


# Get dummy columns for the categorical columns and shuffle the data.

dummy_cols = ['funder', 'installer', 'basin', 'public_meeting', 'scheme_management', 'permit',
              'construction_year', 'extraction_type_class', 'payment_type', 'water_quality',
              'quantity', 'source_type', 'source_class', 'waterpoint_type',
             'waterpoint_type_group']

train = pd.get_dummies(train, columns = dummy_cols)

train = train.sample(frac=1).reset_index(drop=True)

test = pd.get_dummies(test, columns = dummy_cols)


# Let's split the train set into train and validation sets. Also remove the target.

target = train.status_group
# target = "status_group"
features = train.drop('status_group', axis=1)


print(target)
X_train, X_val, y_train, y_val = train_test_split(features, target, train_size=0.8)
# print(X_val)

# print(features)

# Both the train and test set are ready for modelling. I'll use a gradient boosting algorithm.
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

def model(X_train, X_val, y_train, y_val, test):
    if __name__ == '__main__':
        param_grid = {'learning_rate': [0.075, 0.7],
                      'max_depth': [13, 14],
                      'min_samples_leaf': [15, 16],
                      'max_features': [1.0],
                      'n_estimators': [100, 200]}

        estimator = GridSearchCV(estimator=GradientBoostingClassifier(),
                                 param_grid=param_grid,
                                 n_jobs=-1)

        estimator.fit(X_train, y_train)

        best_params = estimator.best_params_

        print(best_params)

        validation_accuracy = estimator.score(X_val, y_val)
        print('Validation accuracy: ', validation_accuracy)



model(X_train, X_val, y_train, y_val, test)
#{'min_samples_leaf': 16, 'n_estimators': 100, 'learning_rate': 0.075, 'max_features': 1.0, 'max_depth': 14}
#Validation accuracy:  0.796043771044


# Get data necessary for submission.

submit_loc = r"pump_submit.csv"
test_id = pd.read_csv(submit_loc)
test_id.columns = ['idd', 'status_group']
test_id = test_id.idd


def model_for_submission(features, target, test):
    if __name__ == '__main__':

         best_params = {'learning_rate': [0.075],
                        'max_depth': [14],
                        'min_samples_leaf': [16],
                        'max_features': [1.0],
                        'n_estimators': [100]}

         estimator = GridSearchCV(estimator=GradientBoostingClassifier(),
                                 param_grid=best_params,
                                 n_jobs=-1)

         estimator.fit(features, target)

         predictions = estimator.predict(test)

         data = {'ID': test_id, 'status_group': predictions}

         submit = pd.DataFrame(data=data)

         vals_to_replace = {2:'functional', 1:'functional needs repair',
                           0:'non functional'}

         submit.status_group = submit.status_group.replace(vals_to_replace)

         submit.to_csv('pump_predictions.csv', index=False)


model_for_submission(features, target, test)

