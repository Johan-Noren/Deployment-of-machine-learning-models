import pandas as pd

import joblib
import config


def make_prediction(input_data):
    
    # load pipeline and make predictions
    # rturn predictions

    pipeline_titanic = joblib.load(config.PIPELINE_NAME)
    
    results = pipeline_titanic.predict(input_data)
    
    return results
   
if __name__ == '__main__':
    
    # test pipeline
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = pd.read_csv(config.TRAINING_DATA_FILE)

    training_data, testing_data, training_target, testing_target = train_test_split(
        data.drop(config.TARGET, axis=1),
        data[config.TARGET],
        test_size=0.2,
        random_state=0)  # we are setting the seed here
    
    pred = make_prediction(testing_data)
    
    # determine the accuracy
    print('test accuracy: {}'.format(accuracy_score(testing_target, pred)))
    print()

# results: test accuracy: 0.7175572519083969

# test accuracy: 0.6832


