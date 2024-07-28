from sklearn.model_selection import train_test_split

def create_train_test_split(X, y, test_size, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    - X: Features.
    - y: Labels.
    - test_size: Size of the test set (can be an int or a float representing the proportion).
    - random_state: Seed for reproducibility. Default is 42.

    Returns:
    - X_train: Training features.
    - X_test: Testing features.
    - y_train: Training labels.
    - y_test: Testing labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def split_data_by_test_sizes(X_train, y_train, test_sizes, random_state=42):
    """
    Splits the training data into new training sets and pools based on provided test sizes.

    Args:
        X_train (array-like): Features of the original training set.
        y_train (array-like): Labels of the original training set.
        test_sizes (list): List of test sizes for splitting the data.
        random_state (int): Random state for reproducibility.

    Returns:
        data_dict (dict): A dictionary containing the splits for each test size.
    """
    data_dict = {}
    for test_size in test_sizes:
        X_pool, X_train_new, Y_pool, Y_train_new = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
        data_dict[test_size] = {'X_train_new': X_train_new, 'Y_train_new': Y_train_new, 'X_pool': X_pool, 'Y_pool': Y_pool}
    return data_dict
