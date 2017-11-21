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
    train_df.info()
    # print('*'*40)
    # test_df.info()
    # print('*' * 40)
    # print('{}'.format(train_df.describe(percentiles=[.61, .62])))
    print('{}'.format(train_df.describe(include=['O'])))
    # print('{}'.format(
    #     train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
    #         .sort_values(by='Survived', ascending=False)
    # ))
    # print('{}'.format(
    #     train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
    #         .sort_values(by='Survived', ascending=False)
    # ))
    # print('{}'.format(
    #     train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()
    #         .sort_values(by='Survived', ascending=False)
    # ))

    # print('{}'.format(
    #     train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean()
    #         .sort_values(by='Survived', ascending=False)
    # ))
    train_df_age = train_df[["Age", "Survived"]]
    train_df_age['Age'] = train_df_age['Age'].apply(np.round)
    print('{}'.format(
        train_df_age[["Age", "Survived"]].groupby(['Age'], as_index=False).mean()
            .sort_values(by='Age', ascending=True)
    ))
    # g = sns.FacetGrid(train_df, col='Survived')
    # g.map(plt.hist, 'Age', bins=40)
    #
    # grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
    # # grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
    # grid.map(plt.hist, 'Age', alpha=.8, bins=20)
    # grid.add_legend()
    #
    # # grid = sns.FacetGrid(train_df, col='Embarked')
    # grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
    # grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    # grid.add_legend()

    grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
    # grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
    grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
    grid.add_legend()

    plt.show()


if __name__ == '__main__':
    pd.set_option('display.width', 300)
    main()
