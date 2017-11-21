# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


def main():
    train_df = pd.read_csv('data_files/train.csv')
    test_df = pd.read_csv('data_files/test.csv')
    combined = [train_df, test_df]
    # print('{}'.format(train_df.columns.values))
    # print('{}'.format(test_df.columns.values))
    # print('{}'.format(train_df.head()))
    # print('{}'.format(train_df.tail()))
    # print('*' * 40)
    # train_df.info()
    # print('*'*40)
    # test_df.info()
    # print('*' * 40)
    # print('{}'.format(train_df.describe(percentiles=[.61, .62])))
    print('{}'.format(train_df.describe(include=['O'])))
    print('{}'.format(
        train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
            .sort_values(by='Survived', ascending=False)
    ))
    print('{}'.format(
        train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
            .sort_values(by='Survived', ascending=False)

    ))


if __name__ == '__main__':
    pd.set_option('display.width', 300)
    main()
