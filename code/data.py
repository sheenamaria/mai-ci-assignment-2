import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class Data:
    def __init__(self, file_path, train_split=80, val_split=10, test_split=10):
        # Validate split percentages
        if train_split + val_split + test_split != 100:
            raise ValueError("Train, validation, and test split percentages must sum to 100")
        
        self.splits = {'train': train_split,
                       'val': val_split,
                       'test': test_split}

        # Load .mat file
        mat_data = sio.loadmat(file_path)
        
        # Extract features and labels
        X = mat_data['X']  # Features: 8671 x 784 matrix
        y = mat_data['Y'].ravel()  # Labels: 8671 vector
        
        # One-hot encode labels
        encoder = OneHotEncoder()
        y_encoded = encoder.fit_transform(y.reshape(-1, 1)).toarray()
        
        # Perform stratified splits
        if test_split > 0:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y_encoded, test_size=test_split/100, random_state=42
            )
        else:
            X_train_val = X
            y_train_val = y_encoded
            X_test = None
            y_test = None
        
        # Store splits as DataFrames with numeric index
        self.train_val_data = pd.DataFrame(X_train_val)
        self.train_val_labels = pd.DataFrame(y_train_val)
        
        self.test_data = pd.DataFrame(X_test)
        self.test_labels = pd.DataFrame(y_test)
        
        # Metadata
        self.num_classes = y_encoded.shape[1]
        self.input_shape = X.shape[1]
