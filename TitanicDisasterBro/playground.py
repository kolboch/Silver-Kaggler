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
    combine = [train_df, test_df]
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

    # grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
    # # grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
    # grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
    # grid.add_legend()

    # plt.show()

    #     lets do come cleanup of data
    print('Data before cleanup: {} {} {} {}'.format(train_df.shape, test_df.shape, combine[0].shape, combine[1].shape))
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]
    print(
        'Data after cleanup: train shape: {} test shape: {} combine shapes:{} {}'.format(train_df.shape, test_df.shape,
                                                                                         combine[0].shape,
                                                                                         combine[1].shape))

    # extracting titles from names and replacement
    for data_set in combine:
        data_set['Title'] = data_set.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    print('{}'.format(pd.crosstab(train_df['Title'], train_df['Sex'])))

    for data_set in combine:
        data_set['Title'] = data_set['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        data_set['Title'] = data_set['Title'].replace(['Mlle', 'Ms'], 'Miss')
        data_set['Title'] = data_set['Title'].replace('Mme', 'Mrs')

    print('{}'.format(pd.crosstab(train_df['Title'], train_df['Sex'])))

    print('{}'.format(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()))
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for data_set in combine:
        data_set['Title'] = data_set['Title'].map(title_mapping)
        data_set['Title'] = data_set['Title'].fillna(0)

    print('{}'.format(combine[0].head()))

    train_df.drop(train_df[['Name', 'PassengerId']], axis=1, inplace=True)
    test_df.drop(test_df[['Name']], axis=1, inplace=True)

    # print('{}'.format(combine[0].head()))
    # print('{}'.format(combine[1].head()))

    #     further changing features to numerical, ex sex: male -> 0, female -> 1
    sex_mapping = {"male": 0, "female": 1}
    for data_set in combine:
        data_set['Sex'] = data_set['Sex'].map(sex_mapping).astype(int)

    # print('{}'.format(combine[0].head()))

    #     we will guess NaN values of age through median,
    #     but for given record from correlation between gender and Pclass of all passengers
    # grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
    # grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    # grid.add_legend()
    # plt.show()

    guess_ages = np.zeros((2, 3))  # for every combination of sex and Pclass
    for data_set in combine:
        for i in [0, 1]:  # gender
            for j in [1, 2, 3]:  # Pclass
                guess_df = data_set[(data_set['Sex'] == i) & (data_set['Pclass'] == j)]['Age'].dropna()
                # alternative for median
                # age_mean = guess_df.mean()
                # age_std = guess_df.std()
                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)
                age_guess = guess_df.median()
                guess_ages[i, j - 1] = int(age_guess / 0.5 + 0.5) * 0.5
        # now assigning computed age guesses
        for i in [0, 1]:  # gender
            for j in [1, 2, 3]:  # Pclass
                data_set.loc[(data_set.Age.isnull()) & (data_set.Sex == i) & (data_set.Pclass == j), 'Age'] = \
                    guess_ages[
                        i, j - 1]
        data_set['Age'] = data_set['Age'].astype(int)

    # print('{}'.format(train_df.head()))
    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
    # print('{}'.format(train_df.head()))
    # print('{}'.format(
    #     train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand')
    # )
    # )

    #     replacing age values based on bands
    for data_set in combine:
        data_set.loc[data_set['Age'] <= 16, 'Age'] = 0
        data_set.loc[(data_set['Age'] > 16) & (data_set['Age'] <= 32), 'Age'] = 1
        data_set.loc[(data_set['Age'] > 32) & (data_set['Age'] <= 48), 'Age'] = 2
        data_set.loc[(data_set['Age'] > 48) & (data_set['Age'] <= 64), 'Age'] = 3
        data_set.loc[data_set['Age'] > 64, 'Age'] = 4

    train_df.drop(['AgeBand'], 1, inplace=True)
    combine = [train_df, test_df]

    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset[
            'Parch'] + 1  # creating new feature family size, by combining parent-child, sibling-spouse

    print('{}'.format(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).agg(
        ['mean', 'count']).reset_index().sort_values([('Survived', 'mean')], ascending=False)))

    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    print('{}'.format(train_df.loc[train_df['IsAlone'] == 1, ['IsAlone']].count()))

    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

    combine = [train_df, test_df]

    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass

    print('{}'.format(train_df.head()))
    print('{}'.format(train_df[['Age*Class', 'Survived']].groupby(['Age*Class'], as_index=False).mean()))


if __name__ == '__main__':
    pd.set_option('display.width', 300)
    main()
