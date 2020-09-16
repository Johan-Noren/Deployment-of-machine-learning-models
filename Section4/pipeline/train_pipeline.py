import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from pipeline import titanic_pipe
import config



def run_training():
    """Train the model."""

    data = pd.read_csv(config.TRAINING_DATA_FILE)

    training_data, testing_data, training_target, testing_target = train_test_split(
        data.drop(config.TARGET, axis=1),
        data[config.TARGET],
        test_size=0.2,
        random_state=0)  # we are setting the seed here
    
    titanic_pipe.fit(training_data, training_target)
    joblib.dump(titanic_pipe, config.PIPELINE_NAME)
    
    

if __name__ == '__main__':
    run_training()

    

    
    
    train_data
    train_target

    test_data
    test_target
    