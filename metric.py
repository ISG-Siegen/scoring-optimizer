import numpy as np


class Metric:

    def __init__(self):
        pass

    def score(self, truth, recommendations, topn_score, index_mask) -> float:
        raise NotImplementedError("This method should be implemented by a subclass.")


class NDCG(Metric):

    def __init__(self):
        super().__init__()
        self.metric_name = "NDCG"

    def score(self, truth, recommendations, topn_score, index_mask) -> float:
        # calculate and store the ndcg for each user
        ndcg_per_user = []
        # pre-compute the dg for each ranked element
        discounted_gain_per_k = np.array([1 / np.log2(i + 1) for i in range(1, topn_score + 1)])
        # pre-compute the idcg
        idcg = discounted_gain_per_k.sum()
        # loop through recommendation lists of each user
        for user, predictions in recommendations.items():
            # if there are no or too few recommendations for this user, skip
            if predictions.shape[0] < len(index_mask):
                ndcg_per_user.append(0)
                continue
            # get sampling indices
            sample_indices = np.argwhere(index_mask).flatten()
            # look only at the sampled recommendations
            top_k_predictions = predictions.values[:, 0][sample_indices]
            # filter interactions for current user from test set
            positive_test_interactions = truth["item"][truth["user"] == user].values
            # check how many of the top-k recommendations appear in the test set
            hits = np.in1d(top_k_predictions, positive_test_interactions)
            # calculate the dcg for this user
            user_dcg = discounted_gain_per_k[hits].sum()
            # calculate the ndcg for this user
            user_ndcg = user_dcg / idcg
            # append current ndcg
            ndcg_per_user.append(user_ndcg)
        return sum(ndcg_per_user) / len(ndcg_per_user)


class Precision(Metric):

    def __init__(self):
        super().__init__()
        self.metric_name = "Precision"

    def score(self, truth, recommendations, topn_score, index_mask) -> float:
        # calculate and store the precision for each user
        precision_per_user = []
        # loop through recommendation lists of each user
        for user, predictions in recommendations.items():
            # if there are no or too few recommendations for this user, skip
            if predictions.shape[0] < len(index_mask):
                precision_per_user.append(0)
                continue
            # get sampling indices
            sample_indices = np.argwhere(np.array(index_mask) == 1).flatten()
            # look only at the sampled recommendations
            top_k_predictions = predictions.values[:, 0][sample_indices]
            # filter interactions for current user from test set
            positive_test_interactions = truth["item"][truth["user"] == user].values
            # check how many of the top-k recommendations appear in the test set
            hits = np.in1d(top_k_predictions, positive_test_interactions).sum()
            # calculate the precision for this user
            user_precision = hits / topn_score
            # append current precision
            precision_per_user.append(user_precision)
        # the final result is the average precision over each user
        return sum(precision_per_user) / len(precision_per_user)
