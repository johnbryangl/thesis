# import required libraries

# for data preparation
import pandas as pd

# for building the required models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# scoring functions
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# for saving the models
import pickle




# SECTION 1: DATA PREPARATION
print(f'[START]\tSECTION 1: DATA PREPARATION')

# load dataset for training the model
lms_train_test_data = pd.read_csv('dbs.csv', sep=';')

# identify features/columns from the dataset that will
# be used in training to help predict y
features = ['access', 'tests', 'exam', 'project', 'assignments']

# separate the dataset's features
x = lms_train_test_data[features]

# separate the dataset's target, y - which tells if a student will drop out
y = lms_train_test_data.graduate

# split the dataset into a training set and validation set
# assign 80% and 20%, respectively

# random state is set to any integer to ensure reproducibility
# of results across different runs
# TODO NORMALIZE!!!
# TODO FEATURE NAMES!!!  UserWarning: X does not have valid feature names, but RandomForestClassifier was
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)

print('Dataset has been split for training and testing.')
print(f'[END]\tSECTION 1: DATA PREPARATION\n\n')





# SECTION 2.1: TRAINING THE RANDOM FOREST MODEL
print(f'[START]\tSECTION 2.1: TRAINING THE RANDOM FOREST MODEL')

# create a model instance of the RandomForestClassifier class
rf_model = RandomForestClassifier(random_state=1)

# pass the training data as arguments for the model instance's fit method
# and start the training process
rf_model.fit(train_x, train_y)

pickle.dump(rf_model, open('rf_model.sav', 'wb'))

print('Random forest model has finished training.')
print(f'[END]\tSECTION 2.1: TRAINING THE RANDOM FOREST MODEL\n\n')





# SECTION 2.2: TESTING THE RANDOM FOREST MODEL
print(f'[START]\tSECTION 2.2: TESTING THE RANDOM FOREST MODEL')

# pass the testing data as arguments for the model's predict method
# and start the testing process
rf_model_test_predictions = rf_model.predict(test_x)

# calculate the testing accuracy score using the previously imported
# scoring function by passing the true values and predicted values as arguments
rf_model_test_accuracy = accuracy_score(test_y, rf_model_test_predictions)

print('Random forest model has finished testing.')
print(f'[END]\tSECTION 2.2: TESTING THE RANDOM FOREST MODEL\n\n')





# SECTION 3.1: TRAINING THE LOGISTIC REGRESSION MODEL
print(f'[START]\tSECTION 3.1: TRAINING THE LOGISTIC REGRESSION MODEL')

# create a model instance of the LogisticRegression class
lr_model = LogisticRegression(random_state=2)

# pass the training data as arguments for the model instance's fit method
# and start the training process
lr_model.fit(train_x, train_y)

pickle.dump(lr_model, open('lr_model.sav', 'wb'))

print('Logistic regression model has finished training.')
print(f'[END]\tSECTION 3.1: TRAINING THE LOGISTIC REGRESSION MODEL\n\n')





# SECTION 3.2: TESTING THE LOGISTIC REGRESSION MODEL
print(f'[START]\tSECTION 3.2: TESTING THE LOGISTIC REGRESSION MODEL')

# pass the testing data as arguments for the model's predict method
# and start the testing process
lr_model_test_predictions = lr_model.predict(test_x)

# calculate the testing accuracy score using the previously imported
# scoring function by passing the true values and predicted values as arguments
lr_model_test_accuracy = accuracy_score(test_y, lr_model_test_predictions)

print('Logistic regression model has finished testing.')
print(f'[END]\tSECTION 3.2: TESTING THE LOGISTIC REGRESSION MODEL\n\n')





# SECTION 4: TRAINING AND TESTING THE NEURAL NETWORK MODEL
print(f'[START]\tSECTION 4: TRAINING AND TESTING THE NEURAL NETWORK MODEL')

# create a model instance of the Sequential class

# build the layers of the neural network

# we ensure that the final layer only has 1 unit with a sigmoid
# activation function since this is a binary classification task
nn_model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=[len(features)]),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

# set the model instance's compilation arguments for binary classification
# and call its compile method

# here, we use adam as the optimizer which is a general-purpose optimizer for
# most problems
nn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

# pass the training and testing data and configure the number of epochs for training
# and start the training process
nn_model_history = nn_model.fit(
    train_x, train_y,
    validation_data=(test_x, test_y),
    epochs=100,
)

nn_model.save('nn_model.h5')

print('Neural network model has finished training.')
print(f'[END]\tSECTION 4: TRAINING AND TESTING THE NEURAL NETWORK MODEL\n\n')




# SECTION 5: COMPARING AND EVALUATING ALL MODELS' PERFORMANCE
print(f"[START]\tSECTION 5: COMPARING AND EVALUATING ALL MODELS' PERFORMANCE")

# prepare the data for evaluation
lms_eval_data = pd.read_csv('dbs_2020.csv', sep=';')
eval_x = lms_eval_data[features]
eval_y = lms_eval_data.graduate

rf_model_eval_predictions = rf_model.predict(eval_x)
lr_model_eval_predictions = lr_model.predict(eval_x)

rf_model_eval_scores = {
    'accuracy': accuracy_score(eval_y, rf_model_eval_predictions),
    'precision': precision_score(eval_y, rf_model_eval_predictions),
    'recall': recall_score(eval_y, rf_model_eval_predictions),
    'f1': f1_score(eval_y, rf_model_eval_predictions)
}

lr_model_eval_scores = {
    'accuracy': accuracy_score(eval_y, lr_model_eval_predictions),
    'precision': precision_score(eval_y, lr_model_eval_predictions),
    'recall': recall_score(eval_y, lr_model_eval_predictions),
    'f1': f1_score(eval_y, lr_model_eval_predictions)
}



print('Scores for random forest model')
for (k, v) in rf_model_eval_scores.items():
    print(f'{k}: {round(v, 2)* 100}%')

print('\nScores for logistic regression model')
for (k, v) in lr_model_eval_scores.items():
    print(f'{k}: {round(v, 2)* 100}%')


print(f"[END]\tSECTION 5: COMPARING AND EVALUATING ALL MODELS' PERFORMANCE")
