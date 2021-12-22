import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        add_bedrooms_per_room=True,
        add_rooms_per_household=True,
        add_population_per_household=True,
    ):  # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.add_rooms_per_household = add_rooms_per_household
        self.add_population_per_household = add_population_per_household

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        features_to_return = [X]

        if self.add_rooms_per_household:
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            features_to_return.append(
                rooms_per_household.reshape((len(rooms_per_household), -1))
            )

        if self.add_population_per_household:
            population_per_household = X[:, population_ix] / X[:, households_ix]
            features_to_return.append(
                population_per_household.reshape((len(population_per_household), -1))
            )

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            features_to_return.append(
                bedrooms_per_room.reshape((len(bedrooms_per_room), -1))
            )

        return np.hstack(features_to_return)
