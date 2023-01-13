# %% flamlとりあえず動かしてみた。
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from flaml import AutoML

data_wine = sklearn.datasets.load_wine()
df_x = pd.DataFrame(data_wine.data, columns=data_wine.feature_names)
df_y = pd.Series(data_wine.target)

# dfのまま突っ込む
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)


automl = AutoML()
settings = {
    "time_budget": 30,  # total running time in seconds
    "metric": "accuracy",  # metric
    "task": "classification",  # task type
    # "estimator_list": ['rf','lgbm', 'xgboost'],  # list of ML learners
    "log_file_name": "automl.log",  # log file name
    "log_training_metric": True,  # whether to log training metric
    "seed": 1,  # random seed
}
automl.fit(X_train=X_train, y_train=y_train, **settings)

# %% 
print("best ML leaner:", automl.best_estimator)
print("best hyperparmeter config:", automl.best_config)
print("best accuracy on validation data: {0:.4g}".format(1 - automl.best_loss))
print("Training duration of best run: {0:.4g} s".format(automl.best_config_train_time))
# %%
from flaml.data import get_output_from_log
import matplotlib.pyplot as plt
import numpy as np

(
    time_history,
    best_valid_loss_history,
    valid_loss_history,
    config_history,
    metric_history,
) = get_output_from_log(filename=settings["log_file_name"], time_budget=240)

plt.title("Learning Curve")
plt.xlabel("Wall Clock Time (s)")
plt.ylabel("Validation Accuracy")
plt.scatter(time_history, 1 - np.array(valid_loss_history))
plt.step(time_history, 1 - np.array(best_valid_loss_history), where="post")
plt.show()
# %%　特徴量重要度も出せるらしい。
plt.barh(
    automl.model.estimator.feature_name_, automl.model.estimator.feature_importances_
)
# %%
