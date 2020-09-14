import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    data = pd.read_csv(df_path)
    return data



def divide_train_test(df, target):
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(
    df.drop(target, axis=1),  # predictors
    df[target],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility
    
    return X_train, X_test, y_train, y_test



def extract_cabin_letter(df, var):
    # captures the first letter
    df[var] = df[var].str[0] 
    
    return df


def add_missing_indicator(df, vars):
    # function adds a binary missing value indicator
    for var in vars:
        df[var+'_NA'] = np.where(df[var].isnull(),1,0)

    return df

def impute_num(df, vars):
    for var in vars:
        df[var] = df[var].fillna(df[var].median())
        
    return df
    
def impute_na(df, vars, value='Missing'):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    for var in vars:
        df[var] = df[var].fillna(value)

    return df


def remove_rare_labels(df, vars, rare_perc=0.05):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    def find_frequent_labels(df, var, rare_perc):
        # function finds the labels that are shared by more than
        # a certain % of the passengers in the dataset

        df = df.copy()

        tmp = df.groupby(var)[var].count() / len(df)

        return tmp[tmp > rare_perc].index

    for var in vars:
        # find the frequent categories
        frequent_ls = find_frequent_labels(df, var, rare_perc)
        df[var] = np.where(df[var].isin(frequent_ls), df[var], 'Rare')
        
    return df


def encode_categorical(df, vars):
    # adds ohe variables and removes original categorical variable

    for var in vars:
        # to create the binary variables, we use get_dummies from pandas
        df = pd.concat([df, pd.get_dummies(df[var], prefix=var, drop_first=True)], axis=1)
        
    df.drop(labels=vars, axis=1, inplace=True)

    features = list(df.columns)

    return df,  features


def check_dummy_variables(df, dummy_list):
    
    missing = [var for var in dummy_list if var not in df.columns]
    
    
    for var in missing:
        df[var] = 0
    
    df.drop('title_Miss', axis=1, inplace=True)
    
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    return df
    

def train_scaler(df, output_path):
    # train and save scaler
    scaler = StandardScaler()
    scaler.fit(df)
    
    joblib.dump(scaler, output_path)


def scale_features(df, output_path):
    # load scaler and transform data
    scaler = joblib.load(output_path)
    
    df = scaler.transform(df)
    
    return df


def train_model(df, target, output_path):
    # train and save model
    model = LogisticRegression(C=0.0005, random_state=0)
    
    model.fit(df, target)
    
    joblib.dump(model, output_path)

def align_columns(df, columns):
    df = df.loc[:,columns]
    return df

def predict(df, output_path):
    # load model and get predictions
    model = joblib.load(output_path)

    class_ = model.predict(df)
    pred = model.predict_proba(df)[:,1]
    
    return class_, pred
    
    
