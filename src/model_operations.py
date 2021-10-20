import pickle
from src.logger.auto_logger import autolog

def saveModel(path, model):
    try:
        with open(path, 'wb') as file:
            pickle.dump(model, file)
        autolog(f"Model saved to {path}.")
    
    except Exception as e:
        autolog("An exception occured while saving model.", 3)


def loadModel(path):
    try:
        with open(path, 'rb'):
            model = pickle.load(path)
        
        autolog(f"Model loaded successfully.")
        return model

    except Exception as e:
        autolog("An exception occured while loading model.", 3)


