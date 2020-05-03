import pandas as pd
import numpy as np
import re as re

from base import Feature, get_arguments, generate_features

Feature.dir = 'features'


# """sample usage
# """
# class Pclass(Feature):
#     def create_features(self):
#         self.train['Pclass'] = train['Pclass']
#         self.test['Pclass'] = test['Pclass']


class Year(Feature):
    def create_features(self):
        self.train["year"] = pd.to_datetime(train["publishedAt"]).dt.year
        self.test["year"] = pd.to_datetime(test["publishedAt"]).dt.year


class Month(Feature):
    def create_features(self):
        self.train["month"] = pd.to_datetime(train["publishedAt"]).dt.month
        self.test["month"] = pd.to_datetime(test["publishedAt"]).dt.month


class Day(Feature):
    def create_features(self):
        self.train["day"] = pd.to_datetime(train["publishedAt"]).dt.day
        self.test["day"] = pd.to_datetime(test["publishedAt"]).dt.day


class Hour(Feature):
    def create_features(self):
        self.train["hour"] = pd.to_datetime(train["publishedAt"]).dt.hour
        self.test["hour"] = pd.to_datetime(test["publishedAt"]).dt.hour


class Minute(Feature):
    def create_features(self):
        self.train["minute"] = pd.to_datetime(train["publishedAt"]).dt.minute
        self.test["minute"] = pd.to_datetime(test["publishedAt"]).dt.minute


def return_collection_dt(df):
    df['collection_date'] = df["collection_date"]
    return pd.to_datetime(df['collection_date'], format="%y.%d.%m")


class C_year(Feature):
    def create_features(self):
        self.train["c_year"] = return_collection_dt(train).dt.year
        self.test["c_year"] = return_collection_dt(test).dt.year


class C_month(Feature):
    def create_features(self):
        self.train["c_month"] = return_collection_dt(train).dt.month
        self.test["c_month"] = return_collection_dt(test).dt.month


class C_month(Feature):
    def create_features(self):
        self.train["c_month"] = return_collection_dt(train).dt.month
        self.test["c_month"] = return_collection_dt(test).dt.month


class C_day(Feature):
    def create_features(self):
        self.train["c_day"] = return_collection_dt(train).dt.day
        self.test["c_day"] = return_collection_dt(test).dt.day


class Length_tags(Feature):
    def create_features(self):
        self.train["length_tags"] = train['tags'].astype(
            str).apply(lambda x: len(x.split("|")))
        self.test["lenght_tags"] = test['tags'].astype(
            str).apply(lambda x: len(x.split("|")))


class Category_id(Feature):
    def create_features(self):
        self.train["categoryId"] = train['categoryId']
        self.test["categoryId"] = test['categoryId']


class Likes(Feature):
    def create_features(self):
        self.train["likes"] = train['likes']
        self.test["likes"] = test['likes']


class Dislikes(Feature):
    def create_features(self):
        self.train["dislikes"] = train['dislikes']
        self.test["dislikes"] = test['dislikes']


class Comment_count(Feature):
    def create_features(self):
        self.train["comment_count"] = train['comment_count']
        self.test["comment_count"] = test['comment_count']


class Comments_disabled(Feature):
    def create_features(self):
        self.train["comments_disabled"] = train['comments_disabled']
        self.test["comments_disabled"] = test['comments_disabled']


class Ratings_disabled(Feature):
    def create_features(self):
        self.train["ratings_disabled"] = train['ratings_disabled']
        self.test["ratings_disabled"] = test['ratings_disabled']


class Channel_id_enc(Feature):
    def create_features(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        cat_cols = 'channelId'
        df_all = pd.concat([train[cat_cols], test[cat_cols]])
        le.fit(df_all)
        self.train['channelId_enc'] = le.transform(train[cat_cols])
        self.test['channelId_enc'] = le.transform(test[cat_cols])
        # self.train["channel_id_enc"] = le.fit_transform(train[cat_cols])
        # self.test["channel_id_enc"] = le.fit_transform(test[cat_cols])


if __name__ == '__main__':
    args = get_arguments()

    # train = pd.read_feather('./data/input/train.feather')
    # test = pd.read_feather('./data/input/test.feather')

    train = pd.read_feather('./data/input/train.feather')
    test = pd.read_feather('./data/input/test.feather')

    generate_features(globals(), args.force)
