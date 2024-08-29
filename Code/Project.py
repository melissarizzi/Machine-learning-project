import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations_with_replacement

class Preprocessing:
    def __init__(self, path,test_size=0.2, val_size=0.2):
        self.df = pd.read_csv(path)
        self.test_size = test_size
        self.val_size = val_size
        self.outliers = None

    def check_missing_value(self):
        self.n_missing = self.df.isnull().sum()
        self.df = self.df.dropna()

    def find_outliers(self):
        outlier_counts = pd.Series(0, index=self.df.index)
        for col in self.df.columns:
            if self.df[col].dtype in ['float64', 'int64']:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_counts += col_outliers.astype(int)
        self.outliers = self.df[outlier_counts >= 2]

    def remove_outliers(self):
        if self.outliers is None:
            self.find_outliers()
        self.df = self.df.drop(self.outliers.index)

    def create_boxplot(self):
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(self.df.columns[:-1], 1):
            plt.subplot(2, 5, i)
            sns.boxplot(y=self.df[col])
            plt.title(col)
        plt.tight_layout()
        plt.savefig('boxplot.png')

    def check_correlation(self,correlation_threshold=0.8):
        x_df = self.df.iloc[:, :-1]
        sns.pairplot(x_df)
        plt.suptitle('Scatterplot', y=1.02)
        plt.savefig('Scatterplot.png')

        self.correlation_matrix = x_df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation matrix')
        plt.savefig('Heatmap.png')

        high_corr_pairs = [(col1, col2) for col1 in self.correlation_matrix.columns
                           for col2 in self.correlation_matrix.columns
                           if
                           col1 != col2 and abs(self.correlation_matrix.loc[col1, col2]) > correlation_threshold]
        self.vars_to_remove = set()
        for col1, col2 in high_corr_pairs:
            if col1 not in self.vars_to_remove:
                self.vars_to_remove.add(col2)
        self.df = self.df.drop(columns=list(self.vars_to_remove))

    def describe_variables(self):
        self.n_label = self.df['y'].value_counts()
        self.var_description = self.df.iloc[:, :-1].describe()
        self.var_description = pd.DataFrame(self.var_description)
        latex_table = self.var_description.to_latex(index=True)
        with open('variables.tex', 'w') as f:
            f.write(latex_table)

    def standardize_df(self):
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]
        means = X.mean()
        stds = X.std()
        X_standardized = (X - means) / stds
        X_standardized_df = pd.DataFrame(X_standardized, columns=X.columns)
        self.df_stand = pd.concat([X_standardized_df, y], axis=1)

    def divide_df(self):
        X = self.df_stand.iloc[:, :-1].values
        y = self.df_stand.iloc[:, -1].values
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        split_index = int(num_samples * (1 - self.test_size))
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
        self.X_train = X[train_indices]
        self.y_train = y[train_indices]
        self.X_test = X[test_indices]
        self.y_test = y[test_indices]
        self.df_train = pd.DataFrame(self.X_train, columns=self.df_stand.columns[:-1])
        self.df_train['y'] = self.y_train
        val_size = int(len(self.df_train) * self.val_size)
        indices = np.random.permutation(self.df_train.index)
        val_indices = indices[:val_size]
        add_indices = indices[val_size:]
        self.df_val = self.df_train.loc[val_indices]
        self.df_add = self.df_train.loc[add_indices]
        self.X_add = self.df_add.iloc[:, :-1].values
        self.y_add = self.df_add.iloc[:, -1].values
        self.X_val = self.df_val.iloc[:, :-1].values
        self.y_val = self.df_val.iloc[:, -1].values

class Perceptron:
    def __init__(self, epochs=None):
        self.epochs = epochs
        self.best_max_epochs = None
        self.best_accuracy = 0
        self.w = None

    def perceptron_train(self, X, y, max_epochs):
        m, n = X.shape
        self.w = np.zeros(n)
        epoch = 0
        while epoch < max_epochs:
            updates = False
            for i in range(m):
                if y[i] * np.dot(self.w, X[i]) <= 0:
                    self.w += y[i] * X[i]
                    updates = True
            if not updates:
                break
            epoch += 1
        print(f"Training terminates after {epoch} epochs.")
        return self.w

    def perceptron_predict(self, X):
        return np.sign(np.dot(X,self.w))

    def cross_validate(self, X, y, k=5):
        n = len(y)
        fold_size = n // k
        for epoch in self.epochs:
            print(f"Testing epochs: {epoch}")
            accuracies = []
            for i in range(k):
                start, end = i * fold_size, (i + 1) * fold_size
                X_val, y_val = X[start:end], y[start:end]
                X_add = np.concatenate((X[:start], X[end:]), axis=0)
                y_add = np.concatenate((y[:start], y[end:]), axis=0)
                self.w = self.perceptron_train(X_add, y_add, epoch)
                y_pred = self.perceptron_predict(X_val)
                accuracy = np.mean(y_val == y_pred)
                print(f"accuracy: {accuracy}")
                accuracies.append(accuracy)
            mean_accuracy = np.mean(accuracies)
            print(f"Mean accuracy for epochs {epoch}: {mean_accuracy:.4f}")
            if mean_accuracy > self.best_accuracy:
                self.best_accuracy = mean_accuracy
                self.best_max_epochs = epoch
        print(f"Best max epochs = {self.best_max_epochs} with accuracy = {self.best_accuracy:.4f}")
        return self.best_max_epochs

    def compute_test_accuracy(self, X_train, y_train, X_test, y_test):
        self.w = self.perceptron_train(X_train, y_train, self.best_max_epochs)
        y_pred = self.perceptron_predict(X_test)
        self.accuracy = np.mean(y_test != y_pred)
        print(f"Misclassification rate: {self.accuracy:.4f}")

class Pegasos:
    def __init__(self, T_values, lambd_values, eta_functions, k=5):
        self.T_values = T_values
        self.lambd_values = lambd_values
        self.eta_functions = eta_functions
        self.k = k
        self.best_score = 0
        self.best_eta = None
        self.best_lambd = None
        self.best_T = None

    def fit_pegasos(self,T, lambd, X, y, eta):
        m, d = X.shape
        self.w = np.zeros(d)
        for t in range(1, T + 1):
            i = np.random.randint(m)
            xi, yi = X[i], y[i]
            if yi * np.dot(self.w, xi) < 1:
                grad = lambd * self.w - yi * xi
            else:
                grad = lambd * self.w
            self.w -= eta(t) * grad
        return self.w

    def predict(self, X):
        return np.sign(X.dot(self.w))

    def cross_validate(self, X, y):
        n = len(y)
        fold_size = n // self.k
        for T in self.T_values:
            for lambd in self.lambd_values:
                for eta in self.eta_functions:
                    scores = []
                    for i in range(self.k):
                        start, end = i * fold_size, (i + 1) * fold_size
                        X_val, y_val = X[start:end], y[start:end]
                        X_add = np.concatenate((X[:start], X[end:]), axis=0)
                        y_add = np.concatenate((y[:start], y[end:]), axis=0)
                        self.w = self.fit_pegasos(T, lambd, X_add, y_add, eta)
                        y_pred = self.predict(X_val)
                        score = np.mean(y_val == y_pred)
                        scores.append(score)
                    mean_score = np.mean(scores)
                    print(f"T={T}, lambda={lambd}, eta={eta.__name__}: mean accuracy={mean_score:.4f}")
                    if mean_score > self.best_score:
                        self.best_score = mean_score
                        self.best_eta = eta
                        self.best_lambd = lambd
                        self.best_T = T
        print(
            f"Best T = {self.best_T}, Best lambda = {self.best_lambd}, Best eta = {self.best_eta.__name__}, Best score = {self.best_score:.4f}")

    def compute_test_accuracy(self, X_train, y_train, X_test, y_test):
        self.w = self.fit_pegasos(self.best_T, self.best_lambd, X_train, y_train, self.best_eta)
        y_pred = self.predict(X_test)
        self.accuracy = np.mean(y_test != y_pred)
        print(f"Misclassification rate: {self.accuracy:.4f}")

class Logistic:
    def __init__(self, T_values, lambd_values, eta_functions,k=5):
        self.T_values = T_values
        self.lambd_values = lambd_values
        self.eta_functions = eta_functions
        self.k = k
        self.best_score = 0
        self.best_eta = None
        self.best_lambd = None
        self.best_T = None

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def fit_logistic(self,X, y, lambd, T, eta):
        m, d = X.shape
        self.w = np.zeros(d)
        for t in range(1,T+1):
            idx = np.random.randint(m)
            x_t = X[idx]
            y_t = y[idx]
            self.sigma = self.sigmoid(np.dot(x_t, self.w))
            y_bin = (y_t + 1) / 2
            gradient = (self.sigma - y_bin) * x_t
            gradient += lambd * self.w
            self.w -= eta(t) * gradient
        return self.w

    def predict_logistic(self,X):
        predictions = self.sigmoid(np.dot(X, self.w))
        return np.where(predictions >= 0.5, 1, -1)

    def cross_validate(self, X, y):
        n = len(y)
        fold_size = n // self.k
        for T in self.T_values:
            for lambd in self.lambd_values:
                for eta in self.eta_functions:
                    scores = []
                    for i in range(self.k):
                        start, end = i * fold_size, (i + 1) * fold_size
                        X_val, y_val = X[start:end], y[start:end]
                        X_add = np.concatenate((X[:start], X[end:]), axis=0)
                        y_add = np.concatenate((y[:start], y[end:]), axis=0)
                        self.w = self.fit_logistic(X_add, y_add, lambd, T, eta)
                        predictions = self.predict_logistic(X_val)
                        score = np.mean(predictions == y_val)
                        scores.append(score)
                    mean_score = np.mean(scores)
                    print(f"T={T}, lambda={lambd}, eta={eta.__name__}: mean accuracy={mean_score:.4f}")
                    if mean_score > self.best_score:
                        self.best_score = mean_score
                        self.best_eta = eta
                        self.best_lambd = lambd
                        self.best_T = T
        print(
            f"Best T = {self.best_T}, Best lambda = {self.best_lambd}, Best eta = {self.best_eta.__name__}, Best score = {self.best_score:.4f}")
        return self.best_T, self.best_lambd, self.best_eta

    def compute_test_accuracy(self, X_train, y_train, X_test, y_test):
        self.w = self.fit_logistic(X_train, y_train, self.best_lambd, self.best_T,self.best_eta)
        y_pred = self.predict_logistic(X_test)
        self.accuracy = np.mean(y_test != y_pred)
        print(f"Misclassification rate: {self.accuracy:.4f}")
        return self.accuracy

class Feature_expansion:
    def __init__(self):
        pass

    def polynomial_feature_expansion(self,X, degree=2):
        m, d = X.shape
        if degree != 2:
            raise NotImplementedError("This function currently only supports degree 2.")
        self.poly_features = [X]
        for i in range(d):
            self.poly_features.append(X[:, i:i + 1] ** 2)
        for i in range(d):
            for j in range(i + 1, d):
                self.poly_features.append(X[:, i:i + 1] * X[:, j:j + 1])
        self.poly_features =  np.hstack(self.poly_features)
        return self.poly_features

class Linear_weight:
    def __init__(self):
        pass

    def create_table(self,w_perceptron, w_pegasos, w_logistic,df):
        self.features = df.columns.tolist()[:-1]
        self.data = {
            'Feature': self.features,
            'Perceptron': w_perceptron,
            'Pegasos': w_pegasos,
            'Logistic': w_logistic
        }
        self.df_weights = pd.DataFrame(self.data)
        self.df_weights.set_index('Feature', inplace=True)
        latex_table = self.df_weights.to_latex(index=True)
        with open('weights_original.tex', 'w') as f:
            f.write(latex_table)

    def create_plot(self,w_perceptron, w_pegasos, w_logistic):
        features_to_plot = self.features
        weights_to_plot = {
            'perceptron': w_perceptron,
            'pegasos': w_pegasos,
            'logistic': w_logistic
        }
        plt.figure(figsize=(12, 8))
        plt.plot(features_to_plot, weights_to_plot['perceptron'], label='Perceptron', color='b', marker='o')
        plt.plot(features_to_plot, weights_to_plot['pegasos'], label='Pegasos', color='r', marker='s')
        plt.plot(features_to_plot, weights_to_plot['logistic'], label='Logistic', color='g', marker='^')
        plt.xlabel('Features')
        plt.ylabel('Weights')
        plt.title('Feature Weights Comparison Across Models')
        plt.legend()
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('weights.png')

    def create_table_poly(self,w_poly_perceptron,w_poly_pegasos, w_poly_logistic):
        variables = [f"x{i + 1}" for i in range(len(self.features))]
        names = []
        for i in range(len(variables)):
            names.append(f"{variables[i]}^2")
        for i, j in combinations_with_replacement(range(len(variables)), 2):
            if i != j:
                names.append(f"{variables[i]}*{variables[j]}")
        df = pd.DataFrame({
            "Variables": [f"{variables[i]}" for i in range(len(variables))] + names,
            "w_Percep.": w_poly_perceptron,
            "w_Pegasos": w_poly_pegasos,
            "w_logistic": w_poly_logistic
        })
        blocchi = [df[i:i + 11].reset_index(drop=True) for i in range(0, len(df), 11)]
        self.df_poly_w = pd.concat(blocchi, axis=1)
        latex_table = self.df_poly_w.to_latex(index=True)
        with open('weights_poly.tex', 'w') as f:
            f.write(latex_table)

class Kernelized_perceptron:
    def __init__(self):
        self.S = []
        self.y_S = []

    def gaussian_kernel(self,x1, x2, sigma):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))

    def polynomial_kernel(self,x1, x2, degree=3, c=1):
        return (np.dot(x1, x2) + c) ** degree

    def train_kernel_perceptron(self,X, y, kernel_function, epochs=1):
        converged = False
        for epoch in range(epochs):
            error_count = 0
            for t in range(len(X)):
                xt = X[t]
                yt = y[t]
                kernel_sum = sum(self.y_S[i] * kernel_function(self.S[i], xt) for i in range(len(self.S)))
                y_pred = np.sign(kernel_sum)
                if y_pred != yt:
                    self.S.append(xt)
                    self.y_S.append(yt)
                    error_count += 1
            if error_count == 0:
                print(f"Converged after {epoch + 1} epochs.")
                converged = True
                break
        if not converged:
            print("Did not converge within the maximum number of epochs.")
        return self.S, self.y_S

    def predict_kernel_perceptron(self,X, kernel_function):
        self.y_pred = []
        for x in X:
            kernel_sum = sum(self.y_S[i] * kernel_function(self.S[i], x) for i in range(len(self.S)))
            self.y_pred.append(np.sign(kernel_sum))
        return np.array(self.y_pred)

    def tune_hyperparameters(self, X_add, y_add, X_val, y_val, kernel_type, params, epochs_list):
        self.best_score = -np.inf
        self.best_params = {}
        if kernel_type == 'gaussian':
            for sigma in params['sigmas']:
                for epochs in epochs_list:
                    print(f"Testing sigma={sigma} and epochs={epochs}")
                    self.S, self.y_S = self.train_kernel_perceptron(X_add, y_add,
                                                                 lambda x1, x2: self.gaussian_kernel(x1, x2, sigma),
                                                                 epochs)
                    y_pred = self.predict_kernel_perceptron(X_val,
                                                            lambda x1, x2: self.gaussian_kernel(x1, x2, sigma))
                    accuracy = np.mean(y_pred == y_val)
                    if accuracy > self.best_score:
                        self.best_score = accuracy
                        self.best_params = {'sigma': sigma, 'epochs': epochs}
            self.best_kernel = 'gaussian'
        elif kernel_type == 'polynomial':
            for degree in params['degrees']:
                for c in params['cs']:
                    for epochs in epochs_list:
                        print(f"Testing degree={degree}, c={c}, and epochs={epochs}")
                        self.S, self.y_S = self.train_kernel_perceptron(X_add, y_add,
                                                                     lambda x1, x2: self.polynomial_kernel(x1, x2,
                                                                                                           degree, c),
                                                                     epochs)
                        y_pred = self.predict_kernel_perceptron(X_val,
                                                                lambda x1, x2: self.polynomial_kernel(x1, x2, degree,
                                                                                                      c))
                        accuracy = np.mean(y_pred == y_val)
                        print(f'Accuracy: {accuracy}')
                        if accuracy > self.best_score:
                            self.best_score = accuracy
                            self.best_params = {'degree': degree, 'c': c, 'epochs': epochs}
            self.best_kernel = 'polynomial'

        print(f"Best kernel: {self.best_kernel}")
        print(f"Best parameters: {self.best_params}")
        print(f"Best score: {self.best_score}")

    def compute_accuracy(self, X_train, y_train, X_test, y_test, kernel_type):
        if kernel_type == 'gaussian':
            if 'sigma' not in self.best_params or 'epochs' not in self.best_params:
                raise ValueError("Gaussian kernel parameters not set. Run tune_hyperparameters first.")
            kernel_function = lambda x1, x2: self.gaussian_kernel(x1, x2, self.best_params['sigma'])
            self.S, self.y_S = self.train_kernel_perceptron(X_train, y_train, kernel_function, self.best_params['epochs'])
        elif kernel_type == 'polynomial':
            if 'degree' not in self.best_params or 'c' not in self.best_params or 'epochs' not in self.best_params:
                raise ValueError("Polynomial kernel parameters not set. Run tune_hyperparameters first.")
            kernel_function = lambda x1, x2: self.polynomial_kernel(x1, x2, self.best_params['degree'],
                                                                    self.best_params['c'])
            self.S, self.y_S = self.train_kernel_perceptron(X_train, y_train, kernel_function, self.best_params['epochs'])
        else:
            raise ValueError("Invalid kernel type. Choose 'gaussian' or 'polynomial'.")
        y_pred = self.predict_kernel_perceptron(X_test, kernel_function)
        correct_predictions = np.sum(y_pred != y_test)
        self.accuracy = correct_predictions / len(y_test)
        return self.accuracy

class Kernelized_pegasos:
    def __init__(self):
        self.alpha = None

    def gaussian_kernel(self,x1, x2, sigma):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))

    def polynomial_kernel(self,x1, x2, degree=3, c=1):
        return (np.dot(x1, x2) + c) ** degree

    def train_pegasos(self,S, lambda_, T, kernel_function, x, y):
        self.alpha = np.zeros(len(S), dtype=float)
        for t in range(1,T+1):
            it = np.random.randint(0, len(S))
            sum_term = 0
            for j in range(len(S)):
                sum_term += self.alpha[j] * y[j] * kernel_function(x[it], x[j])
            if y[it] * sum_term < lambda_:
                self.alpha[it] += 1
        return self.alpha

    def predict_kernel_pegasos(self, x_new, x, y, kernel_function):
        sum_term = 0
        for j in range(len(x)):
            sum_term += self.alpha[j] * y[j] * kernel_function(x_new, x[j])
        self.prediction = np.sign(sum_term)
        return self.prediction

    def tune_hyperparameters(self, X_add, y_add, X_val, y_val, kernel_type, params, T_values, lambda_values):
        self.best_score = -np.inf
        self.best_params = {}
        if kernel_type == 'gaussian':
            for sigma in params['sigmas']:
                for lambda_ in lambda_values:
                    for T in T_values:
                        print(f"Testing sigma={sigma}, lambda={lambda_}, T={T}")
                        K = lambda x1, x2: self.gaussian_kernel(x1, x2, sigma)
                        S = X_add
                        alpha = self.train_pegasos(S, lambda_, T, K, X_add, y_add)
                        y_pred = np.array([self.predict_kernel_pegasos(x, X_add, y_add, K) for x in X_val])
                        accuracy = np.mean(y_pred == y_val)
                        if accuracy > self.best_score:
                            self.best_score = accuracy
                            self.best_params = {'sigma': sigma, 'lambda': lambda_, 'T': T}
            self.best_kernel = 'gaussian'
        elif kernel_type == 'polynomial':
            for degree in params['degrees']:
                for c in params['cs']:
                    for lambda_ in lambda_values:
                        for T in T_values:
                            print(f"Testing degree={degree}, c={c}, lambda={lambda_}, T={T}")
                            K = lambda x1, x2: self.polynomial_kernel(x1, x2, degree, c)
                            S = X_add
                            alpha = self.train_pegasos(S, lambda_, T, K, X_add, y_add)
                            y_pred = np.array([self.predict_kernel_pegasos(x, X_add, y_add, K) for x in X_val])
                            accuracy = np.mean(y_pred == y_val)
                            print(f'Accuracy: {accuracy}')
                            if accuracy > self.best_score:
                                self.best_score = accuracy
                                self.best_params = {'degree': degree, 'c': c, 'lambda': lambda_, 'T': T}
            self.best_kernel = 'polynomial'
        print(f"Best kernel: {self.best_kernel}")
        print(f"Best parameters: {self.best_params}")
        print(f"Best score: {self.best_score}")

    def compute_accuracy(self, X_train, y_train, X_test, y_test, kernel_type):
        if kernel_type == 'gaussian':
            if 'sigma' not in self.best_params or 'lambda' not in self.best_params or 'T' not in self.best_params:
                raise ValueError("Gaussian kernel parameters not set. Run tune_hyperparameters first.")
            kernel_function = lambda x1, x2: self.gaussian_kernel(x1, x2, self.best_params['sigma'])
            S = X_train
            self.alpha = self.train_pegasos(S, self.best_params['lambda'], self.best_params['T'], kernel_function, X_train,
                                       y_train)
        elif kernel_type == 'polynomial':
            if 'degree' not in self.best_params or 'c' not in self.best_params or 'lambda' not in self.best_params or 'T' not in self.best_params:
                raise ValueError("Polynomial kernel parameters not set. Run tune_hyperparameters first.")
            kernel_function = lambda x1, x2: self.polynomial_kernel(x1, x2, self.best_params['degree'],
                                                                    self.best_params['c'])
            S = X_train
            self.alpha = self.train_pegasos(S, self.best_params['lambda'], self.best_params['T'], kernel_function, X_train,
                                       y_train)
        else:
            raise ValueError("Invalid kernel type. Choose 'gaussian' or 'polynomial'.")

        y_pred = np.array([self.predict_kernel_pegasos(x, X_train, y_train, kernel_function) for x in X_test])
        self.accuracy = np.mean(y_pred != y_test)
        return self.accuracy
