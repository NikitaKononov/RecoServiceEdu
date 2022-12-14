class PopularModel:
    def __init__(self):
        # результат из нашего ноутбука homework 3
        self.rec_popular = [10440, 15297, 9728, 13865, 4151, 3734, 2657, 4880,
                            142, 6809]

    def __call__(self, user_id):
        return self.rec_popular
