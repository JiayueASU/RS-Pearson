import pandas as pd
from math import sqrt

dataset={'A': {'article 1': 5,
               'article 2': 3,
               'article 3': 3,
               'article 4': 3,
               'article 5': 2,
               'article 6': 3},

         'B': {'article 1': 5,
               'article 2': 3,
               'article 3': 5,
               'article 4': 5,
               'article 5': 3,
               'article 6': 3},
               
         'C': {'article 1': 2,
               'article 3': 5,
               'article 4': 3,
               'article 6': 4},

         'D': {'article 3': 5,
               'article 4': 4,
               'article 6': 4},

         'E': {'article 1': 4,
               'article 2': 4,
               'article 3': 4,
               'article 5': 2,
               'article 6': 3},

         'F': {'article 1': 3,
               'article 3': 4,
               'article 4': 5,
               'article 5': 3,
               'article 6': 3},

         'G': {'article 3': 4,
               'article 4': 4,
               'article 5': 1}}

dataset_df = pd.DataFrame(dataset)
dataset_df.fillna("Null",inplace = True)

# print(dataset_df)

# Output:
# Users\Items   A      B  ...       F             G
# article   1      5      5  ...      3.0           Null
# article   2      3      3  ...      Null          Null
# article   3      3      5  ...      4.0           4.0
# article   4      3      5  ...      5.0           4.0
# article   5      2      3  ...      3.0           1.0
# article   6      3      3  ...      3.0           Null

def unique_items():
    unique_items_list = []
    for user in dataset.keys():
        for items in dataset[user]:
            unique_items_list.append(items)
    s = set(unique_items_list)
    # s = {'article   4', 'article   6', 'article   2', 'article   1', 'article   5', 'article   3'}
    unique_items_list = list(s)
    return unique_items_list

# print(unique_items())
# Output:
# ['article   2', 'article   4', 'article   3', 'article   1', 'article   5', 'article   6']

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

    # Calculate the pearson score, 
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

# print(user_corelation("A", "B"))
# Output: 
# 0.5570860145311555
    
def most_similar_users(target_user,no_of_users):    
    # Using list comprehension for finding pearson similarity between users
    scores = [(user_corelation(target_user,other_user),other_user) for other_user in dataset if other_user !=target_user]
    
    # Sort the scores in descending order
    scores.sort(reverse=True)
    
    # Return the scores between the target user & other users
    return scores[0:no_of_users]

# print(most_similar_users("A", 3))
# [(0.9999999999999991, 'G'), (0.6634034720037777, 'E'), (0.5570860145311555, 'B')]

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

# unseen_article   s,seen_article   s=target_article   _to_users('D')
# dct = {"Unseen article   s":unseen_article   s,"Seen article   s":seen_article   s}
# pd.DataFrame(dct)
# print(dct)

# Output:
# {'Unseen article   s': ['article   2', 'article   1', 'article   5'], 
#    'Seen article   s': ['article   3', 'article   4', 'article   6']}
    
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

# print(recommendation_phase('D'))
# Output:
# [(3.666666666666667, 'article   2'), (3.479274057836309, 'article   1'), (2.333333333333333, 'article   5')]

print("Enter the target user")
tp = input().title()
if tp in dataset.keys():
    a = recommendation_phase(tp)
    if a != -1 and len(dataset[tp]) < len(unique_items()):
        print("Recommendation Using User based Collaborative Filtering:  ")
        for webseries,weights in a:
            print(webseries,'---->',weights)
    else:
        print("No more recommendation needed!")
else:
    print("user not found in the dataset..please try again")
    
# Enter the target user:
# D
# Recommendation Using User based Collaborative Filtering:  
# article 2 ----> 3.666666666666667
# article 1 ----> 3.479274057836309
# article 5 ----> 2.333333333333333
