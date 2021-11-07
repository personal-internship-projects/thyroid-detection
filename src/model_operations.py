import pickle

import sklearn
from src.logger.auto_logger import autolog


def saveModel(path, model):
    """
    This function saves a model to a file.

    Args:
        path ([string]): [The path to save the model to]
        model ([pickle]): [The model to save]
    """
    try:
        with open(path, 'wb') as file:
            pickle.dump(model, file)
        autolog(f"Model saved to {path}.")

    except Exception as e:
        autolog("An exception occured while saving model.", 3)


def loadModel(path):
    """
    This function loads a model from a file.

    Args:
        path ([string]): [Loads model from this path]

    Returns:
        [model]: [Variable of saved model]
    """
    try:
        with open(path, 'rb') as md:
            model = pickle.load(md)

        autolog(f"Model loaded successfully.")
        return model

    except Exception as e:
        autolog(f"An exception occured while loading model. {e}", 3)
