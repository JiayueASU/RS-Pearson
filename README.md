# Collaborative Filtering Recommendation System

<br />

## Introduction

A recommendation system makes prediction based on the historical behaviors of users. Specifically, it is to predict user preference for a set of items based on past experience. To build a recommendation system, the most two popular approaches are **Content-based** and **Collaborative Filtering**.

**Content-based** approach requires the features of items with a good amount, rather than using the interactions and feedbacks of users. For example, it can be movie attributes such as release date, director, actor, rating, etc. **Collaborative Filtering**, on the other hand, doesn’t need anything else except the historical preference of users on a set of items. Since it is based on historical data, *the core assumption here is that the users who have agreed in the past tend to also agree in the future.* In terms of user preference, it usually expressed by two categories. **Explicit Rating**, is a rate given by a user to an item on a sliding scale, like 5 stars for Titanic. This is the most direct feedback from users to show how much they like an item. **Implicit Rating**, suggests users preference indirectly, such as page views, clicks, purchase records, whether or not listen to a music track, and so on. 

In this repository, we shall focus on the collaborative filtering recommendation system with explicit ratings. Unlike content-based filtering, collaborative filtering uses *similarities between users and items simultaneously* to provide recommendations. This allows collaborative filtering models can recommend an item to a user based on the interests of similar users. Furthermore, the embeddings can be learned automatically, without relying on hand-engineering of features. 

This repository contains a recommendation system for a hospital website which recommends textual content of articles (*items*) to patients and doctors. 

<br />

## System Model

The training data consists of a feedback matrix in which:

* Each row represents a user
* Each column represents an item (an article)

The feedback about articles falls into an explicit rating, in which users specify how much they liked a particular article by providing a numerical rating (5: Very interest, 4: Interest, 3: Normal, 2: not very interest, 1: no interest).

When a user visits the homepage, the system should recommend articles based on both:

* Similarity to articles the user has liked in the past

* Articles that similar users liked

<br />

## Implementation

### Step 1: Choose an approriate data structure for your training data

We use a nested dictionary to store all the information about users, articles, and ratings. A nested dictionary is a dictionary inside a dictionary. It's a collection of dictionaries into one single dictionary.

In [1]: 

```python
# Since we need a nested dictionary to store data, collections is imported.
import pandas as pd
from math import sqrt
import collections
```

In [2]: 

```Python
# An example nested dictionary
dataset={'A': {'article 1': 5,
               'article 2': 3,
               'article 3': 3,
               'article 4': 3,
               'article 5': 2,
               'article 6': 3},
         'B': {'article 1': 2,
               'article 3': 5,
               'article 4': 3,
               'article 6': 4}}
```

Here, `dataset` is a nested dictionary with the dictionary user `'A'` and `'B'`. Each dictionary has its own key (Article #) and value (Rating). 

Dataframe is also a good option to store your data:

In [3]:

```python
dataset_df = pd.DataFrame(dataset)
dataset_df.fillna("Null",inplace = True)
print(dataset_df)
```

Out [1]:

| Item/User | 'A'  | 'B'  |
| :-------: | :--: | :--: |
| article 1 |  5   |  2   |
| article 2 |  3   | Null |
| article 3 |  3   |  5   |
| article 4 |  3   |  3   |
| article 5 |  2   | Null |
| article 6 |  3   |  4   |

In [4]:
```python
def readingFile(filename):
    # If you need to load your data from a .csv file
    f = open(filename, "r", encoding = 'UTF-8-sig')
    temp = []
    for row in f:
        r = row.split(',')
        e = [r[0], r[1], int(r[2])]
        temp.append(e)
    dataset = collections.defaultdict(dict)
    for data in temp:
        dataset[data[0]][data[1]] = data[2]
    return dataset
print(readingFile("dataset.csv")
```

Out [2]:
defaultdict(<class 'dict'>, {'A': {'1': 5, '2': 3, '5': 2, '4': 3, '6': 3, '3': 3}, 'B': {'1': 5, '5': 3, '2': 3, '4': 5, '3': 5, '6': 3}, 'D': {'3': 5, '4': 4, '6': 4}, 'F': {'6': 3, '4': 5, '5': 3, '3': 4, '1': 3}, 'E': {'1': 4, '3': 4, '2': 4, '6': 3, '5': 2}, 'C': {'6': 4, '3': 5, '1': 2, '4': 3}, 'G': {'4': 4, '3': 4, '5': 1}})

### Step 2: Generate a list to store all unique items

In [5]:

```Python
def unique_items():
    unique_items_list = []
    for user in dataset.keys():
        for items in dataset[user]:
            unique_items_list.append(items)
    s = set(unique_items_list)
    # s = {'article 1', 'article 2', 'article 3', 'article 4', 'article 5', 'article 6'}
    unique_items_list = list(s)
    return unique_items_list

print(unique_items())
```

Out [3]:
['article 1', 'article 2', 'article 3', 'article 4', 'article 5', 'article 6']

### Step 3: User-based Collaborative Filtering Algorithm (Pearson Similarity)

The standard method of Collaborative Filtering (CF) is known as *Nearest Neighborhood algorithm*. There are *user-based CF* and *item-based CF*. In this repository, we focus on the user-based CF algorithm with *Pearson Similarity*.

Suppose we have an *M* × *N* matrix of ratings, with *M* users and *N* article. Now we want to predict the rating *r(i, j)* if target user *i = 1, ..., M* did not rate the article *j = 1, ..., N*. Basically, the idea is to find the most similar users to your target user (nearest neighbors) and weight their ratings of an item as the prediction of the rating of this item for target user. The process is to calculate the similarities between target user *i* and all other users, select the top *X* similar users, and take the weighted average of ratings from these *X* users with similarities as weights. Pearson correlation is used to describe the similarity between the users, which is calulated with the bottom equation:

<img src="https://github.com/JiayueASU/RS_Pearson/blob/main/pearson_sim.png?raw=true" width=50% height=50%>

In [6]:
```python
def user_corelation(user1,user2):
    both_rated = {}
    for item in dataset[user1]:
        if item in dataset[user2]:
            both_rated[item] = 1

    number_of_ratings = len(both_rated)
    if number_of_ratings == 0:
        return 0
    
    # Calculate E(user1) and E(user2)
    user1_preferences_sum = sum([dataset[user1][item] for item in both_rated])
    user2_preferences_sum = sum([dataset[user2][item] for item in both_rated])

    # Sum up the squares of preferences of each user, calculate E(user1^2) and E(user2^2)
    user1_square_preferences_sum = sum([pow(dataset[user1][item], 2) for item in both_rated])
    user2_square_preferences_sum = sum([pow(dataset[user2][item], 2) for item in both_rated])

    # Sum up the product value of both preferences for each item, calculate E(user1 * user2)
    product_sum_of_both_users = sum([dataset[user1][item] * dataset[user2][item] for item in both_rated])

    # Calculate the pearson correlation, 
    # which is equal to the ratio between (E(user1 * user2) - E(user1) * E(user2)) and
    # the square root value of (E(user1^2) - E(user1)^2) * (E(user2^2) - E(user2)^2)
    numerator_value = product_sum_of_both_users - (
    user1_preferences_sum * user2_preferences_sum / number_of_ratings)
    denominator_value = sqrt((user1_square_preferences_sum - pow(user1_preferences_sum, 2) / number_of_ratings) * (
    user2_square_preferences_sum - pow(user2_preferences_sum, 2) / number_of_ratings))
    if denominator_value == 0:
        return 0
    else:
        r = numerator_value / denominator_value
        return r
```

In [7]:
```python
def most_similar_users(target_user,no_of_users):    
    # Using list comprehension for finding pearson similarity between users
    scores = [(user_corelation(target_user,other_user),other_user) for other_user in dataset if other_user !=target_user]
    
    # Sort the scores in descending order
    scores.sort(reverse=True)
    
    # Return the scores between the target user & other users
    return scores[0:no_of_users]
  
def target_article_to_users(target_user):
    target_user_article_lst = []
    unique_list = unique_items()
    for articles in dataset[target_user]:
        target_user_article_lst.append(articles)
    s = set(unique_list)
    recommended_articles = list(s.difference(target_user_article_lst))
    a = len(recommended_articles)
    if a == 0:
        return 0
    return recommended_articles,target_user_article_lst
  
def recommendation_phase(user):
    # Gets recommendations for a user by using a weighted average of every other user's rankings
    totals = {}  #empty dictionary
    simSums = {} # empty dictionary
    for other in dataset:
        # don't compare me to myself
        if other == user:
            continue
        sim = user_corelation(user, other)

        # ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in dataset[other]:
            # only score article   s i haven't seen yet
            if item not in dataset[user]:
                # Similrity * score
                totals.setdefault(item, 0)
                totals[item] += dataset[other][item] * sim
                # sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim
                # Create the normalized list

    rankings = [(total / simSums[item], item) for item, total in totals.items()]
    rankings.sort(reverse=True)
#   return rankings
    # returns the recommended items
    recommendataions_list = [(recommend_item,score) for score, recommend_item in rankings]
    return recommendataions_list
```


