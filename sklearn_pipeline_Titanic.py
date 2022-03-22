''' ML pipeline for Titanic dataset '''
import pandas as pd                                                                                  
from sklearn.model_selection import train_test_split   
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV 
import joblib

 
'''Data'''
# Read data
training = pd.read_csv('input/train.csv',index_col=0)
# Select input and output
X = training.iloc[:,1:]
y = (training.iloc[:,0])
# Split data into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y,test_size=1/3,random_state=0)
 
'''Preprocessing'''
# Customed transformers
from utils import transformer_pipelines
num_trfm,cat_trfm_ticket,cat_trfm_cabin,cat_trfm_encode=transformer_pipelines()
   
# Categorical and numerical features
num_features=X.describe().columns.tolist()
cat_features_simplePreproc=['Sex','Embarked']
cat_features_complexPreproc=['Cabin','Ticket']
 
# Preprocessing step using customed transformers
preprocess = ColumnTransformer(
    transformers=[
        ('num', num_trfm, num_features),
        ('cat_ticket',cat_trfm_ticket, ['Ticket']),
        ('cat_cabin',cat_trfm_cabin, ['Cabin']),
        ('cat_encode',cat_trfm_encode,cat_features_simplePreproc)
    ],remainder ='drop',
    sparse_threshold=0)

 
'''Modelling'''
from utils import params, models

models_summary = {}
validation_scores = {}

for model_initials,[model_name,model_function] in models.items():
    # Tune pipeline
    pipe = Pipeline([('preprocess', preprocess),
    (model_initials, model_function)])
    # Randomized search over hyperparameters for best fit
    classifier_grid = RandomizedSearchCV(pipe, params[model_initials], cv=5,n_iter=5).fit(X_train, y_train)
    print('\x1b[43m'+ model_name + '\x1b[0m')
    print('Training set score: ' + '{:1.3f}'.format(classifier_grid.score(X_train, y_train)))
    print('Validation set score: ' + '{:1.3f}'.format(classifier_grid.score(X_val, y_val)))
    validation_scores[model_initials]=classifier_grid.score(X_val, y_val)
    # Hyperparameters tunning summary
    models_summary[model_initials] = pd.DataFrame.from_dict(classifier_grid.cv_results_, orient='columns')
    # Store best hyperparameters for each model
    joblib.dump(classifier_grid.best_estimator_, f'{model_initials}_best.pkl')


'''Prediction using best model'''
# Get best model
best=max(validation_scores, key=validation_scores.get)
# Upload best model
best_model=joblib.load(f'{best}_best.pkl')
# Upload data
X_test = pd.read_csv('input/test.csv',index_col=0)
# Predict classes
y_test= best_model.predict(X_test)
results_rf=pd.DataFrame({'Survived':y_test},index=X_test.index)
results_rf.to_csv
 
