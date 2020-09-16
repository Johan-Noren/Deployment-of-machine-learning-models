from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
import config


titanic_pipe = Pipeline(
    [
        ('Extract_First_Letter', pp.ExtractFirstLetter(variables=['cabin'])),
        ('Numerical_Imputer', pp.NumericalImputer(variables=config.NUMERICAL_VARS)),
        ('Categorical_Imputer', pp.CategoricalImputer(variables=config.CATEGORICAL_VARS)),
        ('Rare_Label_Categorical_Encoder', pp.RareLabelCategoricalEncoder(variables=config.CATEGORICAL_VARS)),
        ('Categorical_Encoder', pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
        ('scaler', StandardScaler()),
        ('Linear_model', LogisticRegression(C=0.005, random_state=0))
    ],
    verbose=True
)
    
    
    