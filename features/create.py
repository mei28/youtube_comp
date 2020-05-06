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
        self.train["likes2"] = train['likes'] ** 2
        self.test['likes2'] = test['likes'] ** 2
        self.train['loglikes'] = np.log(train['likes']+1)
        self.test['loglikes'] = np.log(test['likes']+1)


class Dislikes(Feature):
    def create_features(self):
        self.train["dislikes"] = train['dislikes']
        self.test["dislikes"] = test['dislikes']
        self.train["dislikes2"] = train['dislikes'] ** 2
        self.test['dislikes2'] = test['dislikes'] ** 2
        self.train['logdislikes'] = np.log(train['dislikes']+1)
        self.test['logdislikes'] = np.log(test['dislikes']+1)


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


class Dislikes_rate(Feature):
    def create_features(self):
        self.train['dislike_rate'] = train['dislikes'] / \
            (train['likes'] + train["dislikes"])
        self.test['dislike_rate'] = test['dislikes'] / \
            (test['likes']+test['dislikes'])


class Likes_rate(Feature):
    def create_features(self):
        self.train["like_rate"] = train['likes'] / \
            (train['likes'] + train['dislikes'])
        self.test["like_rate"] = test['likes']/(test["dislikes"]+test["likes"])


class Likes_dislikes_rate(Feature):
    def create_features(self):
        self.train['likes_dislike_ratio'] = train['likes'] / \
            (train['dislikes'] + 1)
        self.test['likes_dislike_ratio'] = test['likes'] / (test['dislikes']+1)


class Channel_title_enc(Feature):
    def create_features(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        col = 'channelTitle'
        df_all = pd.concat([train[col], test[col]])
        le.fit(df_all)
        self.train[col+'_enc'] = le.transform(train[col])
        self.test[col+'_enc'] = le.transform(test[col])


class Comment_likes_dislikes_ratio(Feature):
    def create_features(self):
        self.train['comments_like_ratio'] = train['comment_count'] / \
            (train['likes'] + 1)
        self.test['comments_like_ratio'] = test['comment_count'] / \
            (test['likes'] + 1)
        self.train['comments_dislike_ratio'] = train['comment_count'] / \
            (train['dislikes'] + 1)
        self.test['comments_dislike_ratio'] = test['comment_count'] / \
            (test['dislikes'] + 1)


class Likes_comments_disable(Feature):
    def create_features(self):
        self.train['likes_com'] = train['likes'] * train["comments_disabled"]
        self.test['likes_com'] = test['likes'] * test["comments_disabled"]
        self.train['dislikes_com'] = train['dislikes'] * \
            train["comments_disabled"]
        self.test['dislikes_com'] = test['dislikes'] * \
            test["comments_disabled"]
        self.train['comments_likes'] = train['comment_count'] * \
            train['ratings_disabled']
        self.test['comments_likes'] = test['comment_count'] * \
            test['ratings_disabled']


class Delta_time(Feature):
    def create_features(self):
        train["collection_date"] = pd.to_datetime(
            "20" + train["collection_date"], format="%Y.%d.%m", utc=True)
        test["collection_date"] = pd.to_datetime(
            "20" + test["collection_date"], format="%Y.%d.%m", utc=True)
        train["publishedAt"] = pd.to_datetime(train['publishedAt'], utc=True)
        test["publishedAt"] = pd.to_datetime(test['publishedAt'], utc=True)

        self.train["delta"] = (train["collection_date"] - train["publishedAt"]
                               ).apply(lambda x: x.days)
        self.test["delta"] = (test["collection_date"] - test["publishedAt"]
                              ).apply(lambda x: x.days)
        self.train['log_delta'] = np.log(self.train['delta'])
        self.test['log_delta'] = np.log(self.test['delta'])


class Description(Feature):
    def create_features(self):
        train['description'].fillna(" ", inplace=True)
        test['description'].fillna(" ", inplace=True)
        self.train['has_http'] = train['description'].apply(
            lambda x: x.lower().count('http'))
        self.test['has_http'] = test['description'].apply(
            lambda x: x.lower().count('http'))
        self.train['len_description'] = train['description'].apply(
            lambda x: len(x))
        self.test['len_description'] = test['description'].apply(
            lambda x: len(x))


class Music(Feature):
    def create_features(self):
        train['tags'].fillna(" ", inplace=True)
        test['tags'].fillna(" ", inplace=True)
        self.train['music_title'] = train['title'].apply(
            lambda x: 'music' in x.lower())
        self.test['music_title'] = test['title'].apply(
            lambda x: 'music' in x.lower())
        self.train['music_tabs'] = train['tags'].apply(
            lambda x: 'music' in x.lower())
        self.test['music_tabs'] = test['tags'].apply(
            lambda x: 'music' in x.lower())


class Official(Feature):
    def create_features(self):
        self.train['official_title'] = train['title'].apply(
            lambda x: 'fficial' in x.lower())
        self.test['official_title'] = test['title'].apply(
            lambda x: 'fficial' in x.lower())
        self.train['official_ja'] = train['title'].apply(
            lambda x: '公式' in x.lower())
        self.test['official_ja'] = test['title'].apply(
            lambda x: '公式' in x.lower())


class CM(Feature):
    def create_features(self):
        train['tags'].fillna(" ", inplace=True)
        test['tags'].fillna(" ", inplace=True)
        train['description'].fillna(" ", inplace=True)
        test['description'].fillna(" ", inplace=True)

        self.train['cm_title'] = train['title'].apply(
            lambda x: 'cm' in x.lower())
        self.test['cm_title'] = test['title'].apply(
            lambda x: 'cm' in x.lower())
        self.train['cm_tags'] = train['tags'].apply(
            lambda x: 'cm' in x.lower())
        self.test['cm_tags'] = test['tags'].apply(
            lambda x: 'cm' in x.lower())
        self.train['cm_description'] = train['description'].apply(
            lambda x: 'cm' in x.lower())
        self.test['cm_description'] = test['description'].apply(
            lambda x: 'cm' in x.lower())


if __name__ == '__main__':
    args = get_arguments()

    # train = pd.read_feather('./data/input/train.feather')
    # test = pd.read_feather('./data/input/test.feather')

    train = pd.read_feather('./data/input/train.feather')
    test = pd.read_feather('./data/input/test.feather')

    generate_features(globals(), args.force)
