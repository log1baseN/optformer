from uci_datasets import Dataset

# run python -m pip install git+https://github.com/treforevans/uci_datasets.git

def load_data():
    """
    Load the dataset.
    """
    # Load the dataset
    # The dataset is downloaded and saved in the data directory
    # The data directory is created if it does not exist
    # The dataset is downloaded from the UCI Machine Learning Repository
    # The dataset is saved in the data directory
    # The dataset is loaded from the data directory
    # The dataset is split into training and testing sets
    # The training and testing sets are returned
    data = Dataset("challenger")
    x_train, y_train, x_test, y_test = data.get_split(split=0)
    return x_train, y_train, x_test, y_test