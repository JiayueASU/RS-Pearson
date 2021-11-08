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
# Import libraries and functions you need
import pandas as pd
from math import sqrt
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
dataset_df=pd.DataFrame(dataset)
dataset_df.fillna("Null",inplace=True)
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

### Step 2: Generate a list to store all unique items

In [4]:

```Python
def unique_items():
    unique_items_list = []
    for user in dataset.keys():
        for items in dataset[user]:
            unique_items_list.append(items)
    s=set(unique_items_list)
    # s = {'article 1', 'article 2', 'article 3', 'article 4', 'article 5', 'article 6'}
    unique_items_list=list(s)
    return unique_items_list

print(unique_items())
```

Out [2]:

['article 1', 'article 2', 'article 3', 'article 4', 'article 5', 'article 6']

### Step 3: User-based Collaborative Filtering Algorithm (Pearson Similarity)

The standard method of Collaborative Filtering (CF) is known as *Nearest Neighborhood algorithm*. There are *user-based CF* and *item-based CF*. In this repository, we focus on the user-based CF algorithm with *Pearson Similarity*.

Suppose we have an *M* × *N* matrix of ratings, with *M* users and *N* article. Now we want to predict the rating *r(i, j)* if target user *i = 1, ..., M* did not rate the article *j = 1, ..., N*. The process is to calculate the similarities between target user *i* and all other users, select the top *X* similar users, and take the weighted average of ratings from these *X* users with similarities as weights.

![alt text](https://github.com/JiayueASU/RS-Pearson/blob/main/pearson_sim.png?raw=true)

<img src="https://github.com/JiayueASU/RS-Pearson/blob/main/pearson_sim.png?raw=true" width="100" height="100">

