import pickle
import pandas as pd

def load_resampled_data(pickle_dir: str):
    """
    Loads the resampled data for further processing in the notebook.

    Parameters:
    pickle_dir (str): Directory where the pickled data is stored.

    Returns:
    X_train_resampled (pd.DataFrame): Resampled training features.
    y_train_resampled (pd.Series): Resampled training labels.
    """
    with open(f"{pickle_dir}/train_data_resampled.pkl", 'rb') as f_data:
        X_train_resampled = pickle.load(f_data)

    with open(f"{pickle_dir}/train_data_labels_resampled.pkl", 'rb') as f_labels:
        y_train_resampled = pickle.load(f_labels)

    # Convert to pandas DataFrame for easier manipulation in Jupyter
    X_train_resampled = pd.DataFrame(X_train_resampled)
    y_train_resampled = pd.Series(y_train_resampled)

    return X_train_resampled, y_train_resampled

if __name__ == "__main__":
    X_train_resampled, y_train_resampled = load_resampled_data("pickle")
    print(X_train_resampled.head())
    print(y_train_resampled.value_counts())

