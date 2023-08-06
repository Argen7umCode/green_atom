from sklearn.model_selection import train_test_split

def split_data(data, labels, test_size=0.2, random_state=None):
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )
    return train_data, test_data, train_labels, test_labels
