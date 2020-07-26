import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# colab surumuyle uyumlu degil, pc de calistir.

train = r"preprocessed_train.csv"
test = r"preprocessed_test.csv"

train = pd.read_csv(train)
test = pd.read_csv(test)
print()


dummy_cols = ['funder', 'installer', 'basin', 'public_meeting', 'scheme_management', 'permit',
              'construction_year', 'extraction_type_class', 'payment_type', 'water_quality',
              'quantity', 'source_type', 'source_class', 'waterpoint_type',
              'waterpoint_type_group']

train = pd.get_dummies(train, columns=dummy_cols)

train = train.sample(frac=1).reset_index(drop=True)

test = pd.get_dummies(test, columns=dummy_cols)

# Let's split the train set into train and validation sets. Also remove the target.

target = train.status_group
features = train.drop('status_group', axis=1)

X_train, X_val, y_train, y_val = train_test_split(features, target, train_size=0.8)


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
                                 n_jobs=-1, verbose=15)

        estimator.fit(X_train, y_train)

        best_params = estimator.best_params_

        print(best_params)

        validation_accuracy = estimator.score(X_val, y_val)
        print('Validation accuracy: ', validation_accuracy)


print("started")
model(X_train, X_val, y_train, y_val, test)
# {'min_samples_leaf': 16, 'n_estimators': 100, 'learning_rate': 0.075, 'max_features': 1.0, 'max_depth': 14}
# Validation accuracy:  0.796043771044


# Get data necessary for submission.

submit_loc = r"results/pump_submit.csv"
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

        vals_to_replace = {2: 'functional', 1: 'functional needs repair',
                           0: 'non functional'}

        submit.status_group = submit.status_group.replace(vals_to_replace)

        submit.to_csv('pump_predictions.csv', index=False)


# Run model for submission.

model_for_submission(features, target, test)

# The model scored 0.8073. Which leaves me ranking 203/2147 (as of 04/10/2016) which is just
# inside the top ten percent.
# Below are scores from other models I ran using less variables.
# The modifications helped to improve the model.

# Score: 0.7809 without funder.
# Score: 0.7826 with funder.
# Score: 0.7859 with funder and installer.
# Score: 0.7875 with funder, installer and scheme management.
# Score: 0.7923 with funder, installer, scheme management and extractor type.
# Score: 0.7949 with funder, installer, scheme management, extractor type and basin
# Score: 0.7970 with funder, installer, scheme management, extractor type, basin and a
#              unmodified version of water quality.
