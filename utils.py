''' Utilities to model Titanic dataset with a pipeline'''
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier  
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.utils.fixes import loguniform
 
 
#customed transformers
class TransformerTicket(BaseEstimator, TransformerMixin):
    '''classifying numeric and non-numeric tickets'''
    #estimator method
    def fit(self, X, y = None):
        return self
    #transfprmation
    def transform(self, X, y = None):
        X['Ticket']=X['Ticket'].apply(lambda x:x.isnumeric()).map({True: 'yes', False: 'no'}) 
        return X
 
class TransformerCabin(BaseEstimator, TransformerMixin):
    '''selecting first letter from Cabin'''
    #estimator method
    def fit(self, X, y = None):
        return self
    #transformation
    def transform(self, X, y = None):
        X['Cabin']=X['Cabin'].str.replace(r'(?P<letter>^[\w])([\w\W]*)',
        lambda m: m.group('letter'),regex=True)
        return X
 
# categorical and numerical transformers pipelines
def transformer_pipelines():
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
         ('scaler', StandardScaler())])
    cat_transformer_ticket=Pipeline(
        [('ticket', TransformerTicket()),
         ('imputer', SimpleImputer(strategy='constant',fill_value='other')),
         ('encoder', OneHotEncoder(handle_unknown='ignore'))])    
    cat_transformer_cabin=Pipeline(
        [('cabin', TransformerCabin()),
         ('imputer', SimpleImputer(strategy='constant',fill_value='other')),
         ('encoder', OneHotEncoder(handle_unknown='ignore'))])    
    cat_transformer_encode = Pipeline([
        ('imputer', SimpleImputer(strategy='constant',fill_value='other')),
         ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    return num_transformer,cat_transformer_ticket,cat_transformer_cabin,cat_transformer_encode

# Classification models
models={'rf': ['Random Forest',RandomForestClassifier()],
        'SVC':['Support Vector Classification',SVC()],
        'lr': ['Logistic Regression',LogisticRegression()]}
    
# Hyperparameters grid
params={}
params['rf'] = {
    'rf__n_estimators': [100,500,1000],
    'rf__bootstrap': [True,False],
    'rf__max_depth': [3,5,10,20,50,75,100,None],
    'rf__max_features': ['auto','sqrt'],
    'rf__min_samples_leaf': [1,2,4,10],
    'rf__min_samples_split': [2,5,10]
    }
params['SVC'] = [
    {'SVC__kernel': ['rbf'],
     'SVC__gamma': ['scale', 'auto'],
     'SVC__C': loguniform(1e0, 1e3)},
    {'SVC__kernel': ['linear'],
     'SVC__C': loguniform(1e0, 1e3)},
    {'SVC__kernel': ['poly'],
     'SVC__degree' : [2,3,4,5],
     'SVC__gamma' : ['scale', 'auto'],
     'SVC__C': loguniform(1e0, 1e3)}
              ]
params['lr'] = {
    'lr__penalty' : ['l1', 'l2'],
    'lr__C': loguniform(1e-1, 1e3),
    'lr__fit_intercept': [True,False],
    'lr__solver' : ['liblinear'],
    'lr__tol': [1e-5]
    }
    