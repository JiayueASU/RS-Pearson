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
dataset={'A': {'Article1': 5,
               'Article2': 3,
               'Article3': 3,
               'Article4': 3,
               'Article5': 2,
               'Article6': 3},
         'B': {'Article1': 2,
               'Article3': 5,
               'Article4': 3,
               'Article6': 4}}
```

Here, @dataset is a nested dictionary with the dictionary user @'A' and @'B'. Each dictionary has its own key (Article #) and value (Rating).

