# ====   PATHS ===================

PATH_TO_DATASET = "titanic.csv"
OUTPUT_SCALER_PATH = 'scaler.pkl'
OUTPUT_MODEL_PATH = 'logistic_regression.pkl'


# ======= PARAMETERS ===============

# imputation parameters
#IMPUTATION_DICT = 


# encoding parameters
FREQUENT_LABELS = ['female','male','C','Missing','C','Q','S','Miss','Mr','Mrs']


DUMMY_VARIABLES = ['sex_male','cabin_Missing','cabin_Rare','embarked_Q','embarked_Rare','embarked_S','title_Mr','title_Mrs','title_Rare']


# ======= FEATURE GROUPS =============

TARGET = 'survived'

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_TO_IMPUTE = ['age', 'fare']
