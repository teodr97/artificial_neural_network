from NeuralNetwork import ANN, Layer, Perceptron
import numpy as np
import matplotlib.pyplot as plt


class Optimization:
    def render_accuracy_10fold_cross(self, features, targets, hidden_layer_sizes):
        accuracies = self.perform_10_fold_crossvalidation(features, targets, 100, 0.1, hidden_layer_sizes)
        plt.plot(hidden_layer_sizes, accuracies, 'bo')
        plt.title(f'10-fold cross-validation')
        plt.xlabel('Hidden layer size')
        plt.ylabel('Accuracy')
        plt.show()

    " Take a template architecture for the ANN and plot its performance on the"
    " training set and the validation set during training, across epochs."
    def layout_template_one_hidden_layer_ann(self, X, y, n_epochs, learning_rate):
        "Split the data into training and validation sets"
        # # Convert X and y to numpy arrays
        # X = np.asarray(X)
        # y = np.asarray(y)
        # print("x: " + str(X.shape) + " y: " + str(y.shape))
        # print("x: " + str(X[0]) + " y: " + str(y[0]))
        # Compute the number of samples and the number of folds
        n_samples = len(X)
        n_folds = 10

        # Compute the size of each fold
        fold_size = int(n_samples / n_folds)

        input_layer = Layer(10, 16, "sigmoid")
        hidden_layer_in_ann = Layer(16, 16, "sigmoid")
        output_layer = Layer(16, 7, "softmax")
        ann = ANN([input_layer, hidden_layer_in_ann, output_layer])

        X_val, X_train = X[0:fold_size], X[fold_size:]
        y_val, y_train = y[0:fold_size], y[fold_size:]

        # print("y_train: " + str(y_train[1]))
        y_val = self.one_hot_encoder(y_val, 7)
        y_train = self.one_hot_encoder(y_train, 7)

        errors_train, errors_val = ann.train_with_validation(X_train, y_train, X_val, y_val, n_epochs, learning_rate)
        plt.plot(errors_train)
        plt.title(f'Error rate for training set')
        plt.xlabel('Epoch')
        plt.ylabel('Error rate')
        plt.show()

        # errors = ann.train_with_validation(X_train, y_train, X_val, y_val, n_epochs, learning_rate)
        plt.plot(errors_val)
        plt.title(f'Error rate for validation set')
        plt.xlabel('Epoch')
        plt.ylabel('Error rate')
        plt.show()

    def one_hot_encoder(self, y, size):
        y = y - 1
        y_int = np.array(y, dtype=int)
        y = np.eye(size)[y_int]
        return y

    def one_hot_decoder(self, y):
        y = np.argmax(y, axis=1)
        y = y + 1
        return y

    def get_accuracy(self, ann, x_test, y_test):
        test_prediction = self.one_hot_decoder(ann.predict(x_test))
        c = 0
        for i in range(len(test_prediction)):
            if(test_prediction[i] == y_test[i]):
                c+=1
        return c/len(test_prediction)

    def perform_10_fold_cross_validation(self, X, y, n_epochs, learning_rate, hidden_layer_sizes):
        """
        Perform 10-fold cross-validation on the given data.

        Parameters:
        -----------
        X : numpy.ndarray or list
            An array of feature sets for each object.
        y : numpy.ndarray or list
            An array of target labels for each object.
        n_epochs : int
            The number of epochs to train the network for.
        learning_rate : float
            The learning rate to use for training the network.
        hidden_layer_sizes : list
            A list of integers specifying the number of nodes in each hidden layer.
        activation_function : str
            The activation function to use for the hidden layers.

        Returns:
        --------
        A tuple (accuracies, mean_accuracy, std_accuracy) containing the accuracies for each fold, the mean accuracy, and
        the standard deviation of the accuracies.
        """
        # Convert X and y to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # Compute the number of samples and the number of folds
        n_samples = len(X)
        n_folds = 10

        # Compute the size of each fold
        fold_size = int(n_samples / n_folds)

        # Initialize the accuracies
        accuracies = []


        "hidden_layer_size = [7,15,30] e.g."
        for i in range(len(hidden_layer_sizes)):
            input_layer = Layer(10, hidden_layer_sizes[i], "sigmoid")
            hidden_layer_in_ann = Layer(hidden_layer_sizes[i], hidden_layer_sizes[i], "sigmoid")
            output_layer = Layer(hidden_layer_sizes[i], 7, "softmax")
            ann = ANN([input_layer, hidden_layer_in_ann, output_layer])

            #sum of the accuracies for each fold
            sum_accuracies = 0
            # Perform 10-fold cross-validation
            for i in range(n_folds):
                # Split the data into training and testing sets
                X_val, X_train = X[i * fold_size:(i + 1) * fold_size], np.concatenate((X[:i * fold_size], X[(i + 1) * fold_size:]))
                y_val, y_train = y[i * fold_size:(i + 1) * fold_size], np.concatenate((y[:i * fold_size], y[(i + 1) * fold_size:]))

                # Convert the target labels to one-hot encoding
                y_train = self.one_hot_encoder(y_train, 7)

                # Train the network
                ann.train(X_train, y_train, n_epochs, learning_rate)

                # Compute the accuracy
                sum_accuracies = self.get_accuracy(ann, X_val, y_val) + sum_accuracies

            accuracies.append(sum_accuracies/n_folds)

        return accuracies


class Evaluation:
    def execute_ann_and_render_confusion_matrix(self, y_hat, y):
        y_hat = y_hat.reshape(-1)
        y = y.astype(int)
        confusion_matrix = [[0] * 7 for _ in range(7)]
        for i in range(len(y_hat)):
            confusion_matrix[y[i]-1][y_hat[i]-1] += 1
        for i in range(7):
            for j in range(7):
                confusion_matrix[i][j] /= np.count_nonzero(y == i+1)
        plt.imshow(confusion_matrix, cmap='Blues')
        plt.colorbar()
        tick_marks = np.arange(7)
        plt.xticks(tick_marks, [1, 2, 3, 4, 5, 6, 7])
        plt.yticks(tick_marks, [1, 2, 3, 4, 5, 6, 7])

        for i in range(7):
            for j in range(7):
                plt.text(j, i, format(confusion_matrix[i][j], '.3f'), ha="center", va="center", color = "white" if confusion_matrix[i][j] > 0.5 else "black")

        plt.title(f'Confusion matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def one_hot_encoder(self, y, size):
        y = y - 1
        y_int = np.array(y, dtype=int)
        y = np.eye(size)[y_int]
        return y

    def one_hot_decoder(self, y):
        y = np.argmax(y, axis=1)
        y = y + 1
        return y

    def train_val_test_split(self, X, y, test_size=0.2, val_size=0.2, train_size=None, random_state=None):
        """
        Split the given arrays of feature sets and target labels into training, validation, and testing sets.

        Parameters:
        -----------
        X : numpy.ndarray or list
            An array of feature sets for each object.
        y : numpy.ndarray or list
            An array of target labels for each object.
        test_size : float, optional (default=0.2)
            The fraction of the data to use for testing.
        val_size : float, optional (default=0.2)
            The fraction of the data to use for validation.
        train_size : float, optional (default=None)
            The fraction of the data to use for training. If not provided, the remaining fraction after test and validation
            data is used for training.
        random_state : int, optional (default=None)
            The seed used by the random number generator to shuffle the data before splitting. If None, no shuffling is done.

        Returns:
        --------
        A tuple (X_train, X_val, X_test, y_train, y_val, y_test) containing the feature sets and target labels for each set.
        """
        # Convert X and y to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # Compute the number of samples for each set
        n_samples = len(X)
        n_test = int(test_size * n_samples)
        n_val = int(val_size * n_samples)
        n_train = n_samples - n_test - n_val

        # Shuffle the data (if a random seed is provided)
        if random_state is not None:
            np.random.seed(random_state)
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]

        # Split the data into sets
        X_test, X_remaining, y_test, y_remaining = X[:n_test], X[n_test:], y[:n_test], y[n_test:]
        X_val, X_train, y_val, y_train = X_remaining[:n_val], X_remaining[n_val:], y_remaining[:n_val], y_remaining[n_val:]

        # Compute the size of the training set (if not provided)
        if train_size is None:
            train_size = 1.0 - test_size - val_size

        # Split the remaining data into the training set
        n_train = int(train_size * n_samples)
        X_train, y_train = X_train[:n_train], y_train[:n_train]

        # Return the sets of feature sets and target labels
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_accuracy(self, ann, x_test, y_test):
        test_prediction = self.one_hot_decoder(ann.predict(x_test))
        c = 0
        for i in range(len(test_prediction)):
            if(test_prediction[i] == y_test[i]):
                c+=1
        return c/len(test_prediction)

