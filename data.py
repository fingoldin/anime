import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

users = pd.read_csv("clean_data/users.csv") # each row is a user, each column is an anime (each cell contains the users score of that anime)
users_array = users.to_numpy()


all_anime = pd.read_csv("data/AnimeList.csv")

anime_id_map_reverse = np.loadtxt("clean_data/anime_id_map_reverse.csv").astype(np.int)
anime_id_map = {}
for idx in range(len(anime_id_map_reverse)):
  anime_id_map[anime_id_map_reverse[idx]] = idx

animes = []
test_users = [ [
  { "anime_id": 16498, "score": 1.0 }, # Attack on Titan
  { "anime_id": 30276, "score": 1.0 } # One Punch Man
],
  [{ "anime_id": 16498, "score": 10.0 }], # Attack on Titan
  [{ "anime_id": 30276, "score": 7.0 }] # One Punch Man

]

for anime_id in range(len(users.columns)):
  animes.append(users[users[str(anime_id)] != 0.0])

def kernel(user1, user2):
  sim = 0.0
  dif = np.zeros(user1.shape)
  for i in range(len(user1)):
    if (user1[i] == 0 or user2[i] == 0):
        dif[i] = 0.0
    else:
        dif[i] = user1[i] - user2[i]
        sim += 1.0
  norm = math.sqrt(np.sum(dif * dif))/sim if sim != 0 else 1000
  return math.exp(-norm)

def convert_test(test_user):
  test_vector = np.zeros(len(animes))
  for anime in test_user:
    test_vector[anime_id_map[anime["anime_id"]]] = anime["score"]

  return test_vector

def predict(test_user):
  best_score = float("-inf")
  best_anime = -1

  anime = np.zeros(len(test_user))

  for user in users_array:
      s = kernel(test_user, user)
      anime = anime + s * user

  for i in range(len(anime)):
      if (anime[i] > best_score and test_user[i] == 0.0):
          best_anime = i
          best_score = anime[i]

  return best_anime

def convert_predict(anime_id):
  return all_anime[all_anime["anime_id"] == anime_id_map_reverse[anime_id]].iloc[0]["title"]

# for test_user in test_users:
#  print("Predicting for [", ",".join([ all_anime[all_anime["anime_id"] == anime["anime_id"]].iloc[0]["title"] for anime in test_user ]), "]:  ", convert_predict(predict(convert_test(test_user))))
for test_user in test_users:
    converted = convert_test(test_user)
    print(convert_predict(predict(converted)))
