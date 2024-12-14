from typing import List, Tuple

import numpy as np
import pandas as pd

import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report


from matplotlib import pyplot as plt


class LogisticRegression:
    """
    A class to fit and predict a logistic regression model.
    
    The data frame that is provided can have many features, but the target feature must be binary.
    
    Attributes:
        data (pd.DataFrame): The data frame containing the features and target.
        features (List[str]): A list of the features to use in the model.
        target_feature (str): The name of the binary target feature.
        scaler (StandardScaler): A StandardScaler object to normalize the data.
        has_constant (bool): A flag to indicate whether a constant has been added to the features.
        threshold (float): The threshold used to convert probabilities to binary predictions.
        model (sm.Logit): The fitted logistic regression model.
    """
    def __init__(
            self, 
            data: pd.DataFrame, 
            features: List[str],
            target_feature: str = 'finished',
            has_constant_term: bool = False
    ):
        self.data = data.copy()
        self.features = features
        self.target_feature = target_feature
        
        self.scaler: StandardScaler = StandardScaler()
        self.has_constant: bool = has_constant_term
        self.threshold: float = 0.5
        self.model: sm.Logit = None
        
        
    def balance_data(self):
        # Separate the majority and minority classes
        majority_class = self.data.copy()[self.data[self.target_feature] == False]
        minority_class = self.data.copy()[self.data[self.target_feature] == True]
        
        if len(minority_class) > len(majority_class):
            majority_class, minority_class = minority_class, majority_class

        # Randomly sample from the majority class to match the minority class size
        majority_sampled = majority_class.sample(n=len(minority_class), random_state=42)

        # Save the unused portion of the majority class
        self.unused = majority_class.drop(majority_sampled.index)
        
        # Combine the resampled majority class with the original minority class
        balanced_games = pd.concat([majority_sampled, minority_class])

        # Shuffle the dataset (optional but recommended)
        balanced_games = balanced_games.sample(frac=1, random_state=42).reset_index(drop=True)

        # Check the new class distribution
        print(f"Class distribution: {balanced_games[self.target_feature].value_counts(normalize=True)}")
        print(f"Total number of samples: {len(balanced_games)}")

        self.data = balanced_games
        
    def determine_best_thresh(
            self, 
            x_train: pd.DataFrame,
            y_train: pd.Series,
            plot: bool = False, 
            threshold_range: Tuple[float, float] = (0.4, 0.6), 
            n_samples: int = 201
    ):
        thresholds = np.linspace(*threshold_range, n_samples)
        
        accuracies = []
        precisions = []
        f1_scores = []
        
        y_pred = self.model.predict(x_train)
        
        for threshold in thresholds:
            y_predicted = y_pred > threshold
            
            f1_scores.append(f1_score(y_train, y_predicted, average='weighted'))
            precisions.append(precision_score(y_train, y_predicted, average='weighted'))
            accuracies.append(accuracy_score(y_train, y_predicted))
            
        # Determine the threshold that maximizes the F1 score
        max_f1_score = max(f1_scores)
        max_f1_score_index = f1_scores.index(max_f1_score)
        
        print("Training Set Metrics:")
        print(f"Threshold:   {thresholds[max_f1_score_index]:.4f}")
        print(f"F1 Score:    {max_f1_score:.4f}")
        print(f"Precision:   {precisions[max_f1_score_index]:.4f}")
        print(f"Accuracy:    {accuracies[max_f1_score_index]:.4f}")
            
        if plot:
            plt.plot(thresholds, f1_scores)
            plt.plot(thresholds, accuracies)
            plt.plot(thresholds, precisions)
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.legend(["F1 Score", "Accuracy", "Precision"])
    
        return thresholds[max_f1_score_index]

    def fit(
            self, 
            add_constant: bool = False, 
            balance_data: bool = True,
            train_size: float = 0.8,
            random_state: int = 42,
            report: bool = True
    ):
        if balance_data:
            self.balance_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.data[self.features], 
            self.data[self.target_feature], 
            train_size=train_size,
            random_state=random_state,
            stratify=self.data[self.target_feature]
        )
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        if add_constant | self.has_constant:
            self.has_constant = True
            X_train = sm.add_constant(X_train)
            X_test = sm.add_constant(X_test)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
            
        self.model = sm.Logit(y_train, X_train).fit()
        self.threshold = self.determine_best_thresh(X_train, y_train)
        
        # Predict and report on the test set
        if report:
            y_pred = self.model.predict(X_test) > self.threshold
            print(classification_report(y_test, y_pred, digits=4))
        
    def predict(self, X: pd.DataFrame, threshold: float = None):
        self._check_input(X)
        
        if threshold is None:
            threshold = self.threshold
            
        scaled_X = self.scaler.transform(X)
        if self.has_constant:
            scaled_X = sm.add_constant(scaled_X)
        
        return self.model.predict(scaled_X) > threshold

    def return_train_test(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def predict_proba(self, X: pd.DataFrame):
        self._check_input(X)
        
        scaled_X = self.scaler.transform(X)
        if self.has_constant:
            scaled_X = sm.add_constant(scaled_X)
        
        return self.model.predict(scaled_X)
        
    def summary(self):
        return self.model.summary()
    
    def classification_report(self, X: pd.DataFrame, y: pd.Series, threshold: float = None):
        self._check_input(X)
        
        if threshold is None:
            threshold = self.threshold
        
        scaled_X = self.scaler.transform(X)
        if self.has_constant:
            scaled_X = sm.add_constant(scaled_X)
            
        return classification_report(y, self.model.predict(scaled_X) > threshold, digits=4)
    
    def _check_input(self, X: pd.DataFrame):
        # Make sure the model has been fitted
        if self.model is None:
            raise ValueError("The model has not been fitted yet.")
        
        # Make sure the input data has the same number of features as the training data
        if X.shape[1] != len(self.features):
            raise ValueError("The input data has the wrong number of features.")
        