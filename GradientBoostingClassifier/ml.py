import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Get dummy columns for the categorical columns and shuffle the data.


#
#


train = r"preprocessed_train.csv"
test = r"preprocessed_test.csv"

train = pd.read_csv(train)
test = pd.read_csv(test)

# Get dummy columns for the categorical columns and shuffle the data.

dummy_cols = ['funder', 'installer', 'basin', 'public_meeting', 'scheme_management', 'permit',
              'construction_year', 'extraction_type_class', 'payment_type', 'water_quality',
              'quantity', 'source_type', 'source_class', 'waterpoint_type',
              'waterpoint_type_group']

train = pd.get_dummies(train, columns=dummy_cols)

train = train.sample(frac=1).reset_index(drop=True)

test = pd.get_dummies(test, columns=dummy_cols)

target = train.status_group
features = train.drop('status_group', axis=1)

# %80 e %20 lik train ve test setleri

print(target)
X_train, X_val, y_train, y_val = train_test_split(features, target, train_size=0.8)


def model(X_train, X_val, y_train, y_val, test):
    if __name__ == '__main__':
        param_grid = {'learning_rate': [0.075, 0.7],
                      'max_depth': [13, 14],
                      'min_samples_leaf': [15, 16],
                      'max_features': [1.0],
                      'n_estimators': [100, 200]}

        estimator = GridSearchCV(estimator=GradientBoostingClassifier(),
                                 param_grid=param_grid,
                                 n_jobs=-1,
                                 verbose=15)

        estimator.fit(X_train, y_train)

        best_params = estimator.best_params_

        print(best_params)

        validation_accuracy = estimator.score(X_val, y_val)
        print('Validation accuracy: ', validation_accuracy)


## model for training

# model(X_train, X_val, y_train, y_val, test)


# Submit icin dosya oku.

submit_loc = r"SubmissionFormat.csv"
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
                                 n_jobs=-1,
                                 verbose=15)

        estimator.fit(features, target)

        predictions = estimator.predict(test)

        data = {'ID': test_id, 'status_group': predictions}

        submit = pd.DataFrame(data=data)

        vals_to_replace = {2: 'functional', 1: 'functional needs repair',
                           0: 'non functional'}

        submit.status_group = submit.status_group.replace(vals_to_replace)

        submit.to_csv('ml-result.csv', index=False)


model_for_submission(features, target, test)
