# %%
import lightgbm as lgb
import logging

from logs.logger import log_evaluation

import matplotlib.pyplot as plt


def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, lgbm_params):
    # データセットの作成
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    logging.debug(lgbm_params)

    # ロガーの作成
    logger = logging.getLogger('main')
    callbacks = [log_evaluation(logger, period=30)]

    model = lgb.train(
        params=lgbm_params,
        train_set=lgb_train,
        valid_sets=lgb_eval,
        num_boost_round=8000,
        early_stopping_rounds=10,
        callbacks=callbacks
    )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    # lgb.plot_importance(model, importance_type="gain", max_num_features=20,figsize=(12,6))
    plt.show()
    return y_pred, model
