import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

users = pd.read_csv("clean_data/users.csv")

all_anime = pd.read_csv("data/AnimeList.csv")

anime_id_map_reverse = np.loadtxt("clean_data/anime_id_map_reverse.csv").astype(np.int)
anime_id_map = {}
for idx in range(len(anime_id_map_reverse)):
  anime_id_map[anime_id_map_reverse[idx]] = idx

animes = []
test_users = [ [
  { "anime_id": anime_id_map_reverse[3], "score": 8.0 },
] ]

for anime_id in range(len(users.columns)):
  animes.append(users[users[str(anime_id)] != 0.0])

def kernel(anime_users, test_user):
  val = anime_users.dot(test_user).mean()
  return val

def convert_test(test_user):
  test_vector = np.zeros(len(animes))
  for anime in test_user:
    test_vector[anime_id_map[anime["anime_id"]]] = anime["score"] - 5

  return test_vector

def predict(test_user):
  best_score = float("-inf")
  best_anime = -1
  for anime_id in range(len(animes)):
    score = kernel(animes[anime_id], test_user)
    print(anime_id, ": ", score, " ", animes[anime_id][str(anime_id)])
    if score > best_score and test_user[anime_id] == 0.0:
      best_anime = anime_id
      best_score = score

  return best_anime

def convert_predict(anime_id):
  return all_anime[all_anime["anime_id"] == anime_id_map_reverse[anime_id]].iloc[0]["title"]

for test_user in test_users:
  print("Predicting for [", ",".join([ all_anime[all_anime["anime_id"] == anime["anime_id"]].iloc[0]["title"] for anime in test_user ]), "]:  ", convert_predict(predict(convert_test(test_user))))
