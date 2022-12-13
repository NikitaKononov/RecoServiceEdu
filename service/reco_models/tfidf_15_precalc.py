import json


class TFIDF15_precalc_Model:
    def __init__(self):
        # результат из нашего ноутбука homework 3
        self.rec_popular = [10440, 15297, 9728, 13865, 4151, 3734, 2657, 4880,
                            142, 6809]
        with open("checkpoints/tfidf.json", "r") as read_file:
            self.precalc_recs = json.load(read_file)

    def __call__(self, user_id):
        if user_id in self.precalc_recs.keys():
            return self.precalc_recs[user_id]
        return self.rec_popular
