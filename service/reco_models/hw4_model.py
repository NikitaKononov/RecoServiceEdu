from joblib import dump, load
import numpy as np
import hnswlib


class FmModel:
    def __init__(self, model_path, popular_model_path, dataset_path):
        self.model = load(model_path)
        self.popular_model = load(popular_model_path)
        self.dataset = load(dataset_path)
        self.item_id_map = self.dataset.item_id_map
        self.user_id_map = self.dataset.user_id_map
        self.popular_items = self.dataset.item_id_map.convert_to_external(self.popular_model
                                                                     .popularity_list[0][:10])

        user_embeddings, item_embeddings = self.model.get_vectors(self.dataset)
        max_norm, augmented_item_embeddings = self.augment_inner_product(item_embeddings)
        extra_zero = np.zeros((user_embeddings.shape[0], 1))
        augmented_user_embeddings = np.append(user_embeddings, extra_zero, axis=1)

        M = 48
        efC = 100
        efS = 100

        max_elements, dim = augmented_item_embeddings.shape
        hnsw = hnswlib.Index("ip", dim)
        hnsw.init_index(max_elements, M, efC)
        hnsw.add_items(augmented_item_embeddings)
        hnsw.set_ef(efS)

        self.augmented_item_embeddings = augmented_item_embeddings
        self.augmented_user_embeddings = augmented_user_embeddings
        self.hnsw = hnsw

    def __call__(self, user_id):
        rec_items = self.get_top_10(user_id)
        return rec_items

    def get_top_10(self, user_id):
        try:
            recos = self.approx_candidates(user_id,top_k = 10)
            return recos

        except KeyError:
            return self.popular_items

    def approx_candidates(self, user_ids, top_k = 10):
        internal_user_ids = self.user_id_map.convert_to_internal([user_ids])
        query = self.augmented_user_embeddings[internal_user_ids]
        item_labels, _ = self.hnsw.knn_query(query, k=top_k)
        real_labels = np.array(list(map(self.item_id_map.convert_to_external,item_labels)))
        if real_labels.shape[0] == 1:
            real_labels = real_labels[0]
        return real_labels

    def augment_inner_product(self, factors):
        normed_factors = np.linalg.norm(factors, axis=1)
        max_norm = normed_factors.max()

        extra_dim = np.sqrt(max_norm ** 2 - normed_factors ** 2).reshape(-1, 1)
        augmented_factors = np.append(factors, extra_dim, axis=1)
        return max_norm, augmented_factors
