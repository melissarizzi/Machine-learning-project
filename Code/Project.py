import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
