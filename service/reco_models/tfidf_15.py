import dill
import numpy as np
import pandas as pd
import scipy as sp
from rectools import Columns


class TFIDF15Model:
    def __init__(self):
        self.model = dill.load('checkpoints/TFIDF_15.dill')
        self.interactions = pd.read_csv('checkpoints/interactions.csv')
        # результат из нашего ноутбука homework 3
        self.rec_popular = [10440, 15297, 9728, 13865, 4151, 3734, 2657, 4880,
                            142, 6809]
        self.N = 10

        # Renaming columns, converting timestamp
        self.interactions.rename(columns={'last_watch_dt': Columns.Datetime,
                                          'total_dur': Columns.Weight},
                                 inplace=True)

        self.interactions['datetime'] = pd.to_datetime(
            self.interactions['datetime'])

        self.users_inv_mapping = dict(
            enumerate(self.interactions['user_id'].unique()))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

        self.items_inv_mapping = dict(
            enumerate(self.interactions['item_id'].unique()))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}
        self.mapper = self.generate_implicit_recs_mapper(
            self.model,
            N=self.N,
            users_mapping=self.users_mapping,
            users_inv_mapping=self.users_inv_mapping
        )

    def __call__(self, user_id):
        return self.rec_popular

    def predict(self, user_id):
        if any(self.interactions['user_id'] == user_id):

            recs = pd.DataFrame({
                'user_id': [user_id]
            })
            recs['similar_user_id'], recs['similarity'] = zip(
                *recs['user_id'].map(self.mapper))

            # Exploding lists to get vertical representation
            recs = recs.set_index('user_id').apply(
                pd.Series.explode).reset_index()
            # Deleting recommendations of itself
            recs = recs[~(recs['user_id'] == recs['similar_user_id'])]

            # Joining watched items
            watched = self.interactions.groupby('user_id').agg(
                {'item_id': list})
            recs = recs.merge(watched, left_on=['similar_user_id'],
                              right_on=['user_id'], how='left')
            recs = recs.explode('item_id')

            # Dropping duplicates pairs user_id-item_id and keeping rows with the highest similiarity
            recs = recs.sort_values(['user_id', 'similarity'], ascending=False)
            recs = recs.drop_duplicates(['user_id', 'item_id'], keep='first')

            # Making rank
            recs['rank'] = recs.groupby('user_id').cumcount() + 1

            # Decreasing rank to 10
            recs = recs[recs['rank'] <= 10]

            # Dealing witn less than N recommendations
            s = recs['user_id'].value_counts()
            not_enough_dict = \
                recs[recs['user_id'].isin(s[s < self.N].index)].groupby(
                    'user_id')[
                    'item_id'].agg(lambda row: list(row)).to_dict()

            d = {"user_id": [], "item_id": [], "rank": []}
            # for user_id, max_rank in not_enough_dict.items():
            for user_id, item_id_list in not_enough_dict.items():
                max_rank = len(item_id_list)
                # drop popular items, that is already in recommendation list
                corrected_popular_list = [item for item in self.rec_popular
                                          if item not in item_id_list]
                for i in range(1, self.N - max_rank + 1):
                    d['user_id'].append(user_id)
                    d['item_id'].append(corrected_popular_list[i])
                    d['rank'].append(max_rank + i)

            filled_with_popular = pd.DataFrame.from_dict(d)
            final_result = pd.concat([recs, filled_with_popular])

            return final_result.item_id.to_list()
        else:
            # recommend for cold user
            return self.rec_popular

    def get_coo_matrix(self, df,
                       user_col='user_id',
                       item_col='item_id',
                       weight_col=None,
                       users_mapping=None,
                       items_mapping=None):
        if weight_col:
            weights = df[weight_col].astype(np.float32)
        else:
            weights = np.ones(len(df), dtype=np.float32)

        interaction_matrix = sp.sparse.coo_matrix((
            weights,
            (
                df[user_col].map(users_mapping.get),
                df[item_col].map(items_mapping.get)
            )
        ))
        return interaction_matrix

    def generate_implicit_recs_mapper(self, model, N, users_mapping,
                                      users_inv_mapping):
        def _recs_mapper(user):
            user_id = users_mapping[user]
            recs = model.similar_items(user_id, N=N)
            return [users_inv_mapping[user] for user, _ in recs], [sim for
                                                                   _, sim in
                                                                   recs]

        return _recs_mapper
