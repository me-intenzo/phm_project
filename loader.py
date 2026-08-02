from src.preprocessing.loader import CMAPSSLoader

def load_dataset(subset="FD001"):
    return CMAPSSLoader().load_dataset(subset)
