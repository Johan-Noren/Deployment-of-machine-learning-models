import preprocessing_functions as pf
import config

# =========== scoring pipeline =========

def predict(data):

    data = pf.load_data(config.PATH_TO_DATASET)

    X_train, X_test, y_train, y_test = pf.divide_train_test(data, config.TARGET)
    data = X_test.copy()

        # impute categorical variables
    data = pf.add_missing_indicator(data, config.CATEGORICAL_VARS)

    # extract first letter from cabin
    data = pf.extract_cabin_letter(data, 'cabin')

    # impute NA categorical
    data = pf.impute_na(data, config.CATEGORICAL_VARS)

    # impute NA numerical
    data = pf.add_missing_indicator(data, config.NUMERICAL_TO_IMPUTE)
    data = pf.impute_num(data, config.NUMERICAL_TO_IMPUTE)

    # Group rare labels
    data = pf.remove_rare_labels(data, config.CATEGORICAL_VARS)

    # encode variables
    data, data_features = pf.encode_categorical(data, config.CATEGORICAL_VARS)

    print(data.head(1))
    
    # check all dummies were added
    data = pf.check_dummy_variables(data, config.DUMMY_VARIABLES)
    
    # scale variables
    data = pf.scale_features(data, config.OUTPUT_SCALER_PATH)

    # make predictions
    class_, pred = pf.predict(data, config.OUTPUT_MODEL_PATH)

    
    return class_

# ======================================
    
# small test that scripts are working ok
    
if __name__ == '__main__':
        
    from sklearn.metrics import accuracy_score    
    import warnings
    warnings.simplefilter(action='ignore')

    # Load data
    data = pf.load_data(config.PATH_TO_DATASET)

    X_train, X_test, y_train, y_test = pf.divide_train_test(data, config.TARGET)

    pred = predict(X_test)

    # evaluate
    # if your code reprodues the notebook, your output should be:
    # test accuracy: 0.6832
    print('test accuracy: {}'.format(accuracy_score(y_test, pred)))
    print()
        
