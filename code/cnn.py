import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import sigmoid, relu, softmax, tanh

class CNN:
    def __init__(self, output_layer_activation='softmax', filter_sizes=None, dense_layer_size = 128, hidden_layer_activation='relu'):
        # Validate input parameters
        self._validate_activation_function(output_layer_activation, 'output')
        self._validate_activation_function(hidden_layer_activation, 'hidden')
        
        self.output_layer_activation = output_layer_activation
        self.filter_sizes = filter_sizes or [32]
        self.dense_layer_size = dense_layer_size
        self.hidden_layer_activation = hidden_layer_activation
        
        # Mapping of activation function names to Keras activation functions
        self._activation_map = {
            'sigmoid': sigmoid,
            'relu': relu,
            'softmax': softmax,
            'tanh': tanh
        }
        
        # The model will be defined during fitting
        self.model = None
        
    def _validate_activation_function(self, func_name, layer_type):
        if func_name not in ['sigmoid', 'relu', 'tanh', 'softmax']:
            raise ValueError(f"Unsupported {layer_type} layer activation function: {func_name}")
    
    def _build_model(self, input_shape, num_classes):

        model = Sequential()
        
        # Convolutional blocks
        for i, filter_size in enumerate(self.filter_sizes):
            if i == 0:
                # First layer needs input_shape
                model.add(Input(input_shape))

                model.add(Conv2D(
                    filters=filter_size, 
                    kernel_size=(3, 3), 
                    activation=self._activation_map[self.hidden_layer_activation]
                ))
            else:
                # Subsequent layers
                model.add(Conv2D(
                    filters=filter_size, 
                    kernel_size=(3, 3),
                    activation=self._activation_map[self.hidden_layer_activation]
                ))
            
            # Max pooling layer
            model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Flatten layer
        model.add(Flatten())
        
        # Fully connected layer
        model.add(Dense(
            self.dense_layer_size, 
            activation=self.hidden_layer_activation
        ))
        
        # Output layer
        model.add(Dense(
            num_classes, 
            activation=self._activation_map[self.output_layer_activation]
        ))
        
        return model
    
    def fit(self, dataset, cost_function='categorical_crossentropy', max_epochs=50, learning_rate=0.01, k_folds=5):
        # Reshape data for CNN (add channel dimension)
        X_train_val = dataset.train_val_data.values.reshape(
            -1, 28, 28, 1
        ).astype('float32')
        y_train_val = dataset.train_val_labels.values
        
        # Prepare k-fold cross-validation
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # Track models and performance
        best_val_accuracy = 0
        best_model = None
        best_history = None
        
        # Perform k-fold cross-validation
        for fold, (train_index, val_index) in enumerate(kfold.split(X_train_val), 1):
            # Split data for this fold
            X_train_fold = X_train_val[train_index]
            y_train_fold = y_train_val[train_index]
            X_val_fold = X_train_val[val_index]
            y_val_fold = y_train_val[val_index]
            
            # Build model (create a fresh model for each fold)
            model = self._build_model(
                input_shape=(28, 28, 1), 
                num_classes=dataset.num_classes
            )
            
            # Compile model
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss=cost_function,
                metrics=['accuracy']
            )
            
            # Fit model for this fold
            history = model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=max_epochs,
                batch_size=32,
                verbose=0
            )
            
            # Check if this fold's validation accuracy is the best
            final_val_accuracy = history.history['val_accuracy'][-1]
            if final_val_accuracy > best_val_accuracy:
                best_val_accuracy = final_val_accuracy
                best_model = model
                best_history = history
            
            # Optional: Print fold performance (can be removed if not needed)
            # print(f"Fold {fold} - Best Validation Accuracy: {final_val_accuracy:.4f}")
        
        self.model = best_model
        # Return the history of the best-performing model
        return best_history
    
    def predict(self, X):
        # Reshape and normalize input data
        X_reshaped = X.values.reshape(-1, 28, 28, 1).astype('float32')
        
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X_reshaped)
    
    def evaluate(self, dataset):
        # Reshape and normalize test data
        X_test = dataset.test_data.values.reshape(
            -1, 28, 28, 1
        ).astype('float32')
        y_test = dataset.test_labels.values
        
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        return self.model.evaluate(X_test, y_test)