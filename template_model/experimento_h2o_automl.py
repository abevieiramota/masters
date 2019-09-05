# -*- coding: utf-8 -*-
import h2o
from h2o.automl import H2OAutoML
import numpy as np

h2o.init()

# Import a sample binary outcome train/test set into H2O
a = np.load('../data/templates/discourse_plan_data_2.npy')
train = h2o.H2OFrame(a)

# Identify predictors and response
x = list(range(a.shape[1] - 1))
y = a.shape[1] - 1

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
