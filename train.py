import preprocessing_functions as pf
import config

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
df = pf.load_data(config.PATH_TO_DATASET)

# divide data set
X_train, X_test, y_train, y_test = pf.divide_train_test(df, config.TARGET)

# get first letter from cabin variable
X_train = pf.extract_cabin_letter(X_train, 'cabin')

# impute categorical variables
X_train = pf.add_missing_indicator(X_train, config.CATEGORICAL_VARS)
X_train = pf.impute_na(X_train, config.CATEGORICAL_VARS)

# impute numerical variable
X_train = pf.add_missing_indicator(X_train, config.NUMERICAL_TO_IMPUTE)
X_train = pf.impute_num(X_train, config.NUMERICAL_TO_IMPUTE)

# Group rare labels
X_train = pf.remove_rare_labels(X_train, config.CATEGORICAL_VARS)

# encode categorical variables
X_train, X_train_features = pf.encode_categorical(X_train, config.CATEGORICAL_VARS)

# check dummy variables
X_check

# train scaler and save
pf.train_scaler(X_train, config.OUTPUT_SCALER_PATH)

# scale train set
X_train = pf.scale_features(X_train, config.OUTPUT_SCALER_PATH)

# train model and save
pf.train_model(X_train, y_train, config.OUTPUT_MODEL_PATH)

print('Finished training')