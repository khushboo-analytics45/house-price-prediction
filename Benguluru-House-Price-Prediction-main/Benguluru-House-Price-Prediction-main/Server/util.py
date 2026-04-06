import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None


def get_estimated_price(loaction, sqft, bath, bhk):
    load_saved_artifacts()
    try:
        locationname = loaction.lower()
        loc_index = __data_columns.index(locationname)
    except:
        loc_index = -1

    X = np.zeros(len(__data_columns))
    X[0] = sqft
    X[1] = bath
    X[2] = bhk
    if loc_index >= 0:
        X[loc_index] = 1
    return round(__model.predict([X])[0], 2)


def get_locationnames():
    load_saved_artifacts()
    return __locations


def load_saved_artifacts():
    print("Loading saved artifacts...")
    global __locations
    global __data_columns
    global __model

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open("./artifacts/bengluru_price_model.pickle", 'rb') as f:
        __model = pickle.load(f)

    print("Loading of artifacts is done")


if __name__ == '__main__':
    print(get_locationnames())


