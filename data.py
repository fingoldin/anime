import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

users1 = pd.read_csv("clean_data/users1.csv") # each row is a user, each column is an anime (each cell contains the users score of that anime)
users2 = pd.read_csv("clean_data/users2.csv")
users3 = pd.read_csv("clean_data/users3.csv")
users4 = pd.read_csv("clean_data/users4.csv")
users = pd.concat([users1, users2, users3, users4],ignore_index=True)
users_array = users.to_numpy()

total_watchers = np.ceil(np.sum(users_array, axis=0) / 10.0)
total_watchers = np.sqrt(total_watchers)
total_watchers = np.maximum(np.ones(len(total_watchers)), total_watchers)

print('Finishing Reading Users')

d = {}
d['One Punch Man'] = 30276
d['Death Note'] = 1535
d['Shingeki no Kyojin'] = 16498
d['Angel Beats!'] = 6547
d['Tengen Toppa Gurren Lagann'] = 2001
d['Kimi no Na wa'] = 32281
d['Haikyuu!!'] = 20583
d['Haikyuu!! Second Season'] = 28891
d['Noragami'] = 20507
d['Made in Abyss'] = 34599
d['Planetus'] = 329
d['Hunter x Hunter (2011)'] = 11061
d['Boku no Hero Academia'] = 31964
d['Fairy Tail'] = 6702

all_anime = pd.read_csv("data/AnimeList.csv")

anime_id_map_reverse = np.loadtxt("clean_data/anime_id_map_reverse.csv").astype(np.int)
anime_id_map = {}
for idx in range(len(anime_id_map_reverse)):
  anime_id_map[anime_id_map_reverse[idx]] = idx

test_users = [ [
  { "anime_id": d['Shingeki no Kyojin'], "score": 8.0 },
  { "anime_id": d['One Punch Man'], "score": 6.0 },
  { "anime_id": 21, "score": 8.0 }
  ],
  [{ "anime_id": d['Angel Beats!'], "score": 10.0 }],
  [{ "anime_id": d['Kimi no Na wa'], "score": 10.0 }],
  [
  { "anime_id": d['Angel Beats!'], "score": 10.0 },
  { "anime_id": d['Kimi no Na wa'], "score": 10.0 }
  ],
  [
  { "anime_id": d['Shingeki no Kyojin'], "score": 9.0 },
  { "anime_id": d['Death Note'], "score": 10.0 },
  { "anime_id": d['Tengen Toppa Gurren Lagann'], "score": 10.0 },
  { "anime_id": d['One Punch Man'], "score": 10.0 }
  ],
  [
  { "anime_id": d['Haikyuu!!'], "score": 10.0 },
  { "anime_id": d['Haikyuu!! Second Season'], "score": 10.0 }
  ],
  [{ "anime_id": d['Planetus'], "score": 10.0 }],
  [
  { "anime_id": d['Noragami'], "score": 9.0 },
  { "anime_id": d['Made in Abyss'], "score": 10.0 },
  { "anime_id": d['Hunter x Hunter (2011)'], "score": 10.0 },
  { "anime_id": d['Boku no Hero Academia'], "score": 10.0 },
  { "anime_id": d['Fairy Tail'], "score": 8.0 }
  ]
]

# print('Beginning Scan')
#
# animes = []
# for anime_id in range(len(users.columns)):
#   animes.append(users[users[str(anime_id)] != 0.0])

print('Beginning Calculations')

def kernel(user1, user2):
  product = np.ceil(user1/10.0) * np.ceil(user2/10.0)
  dif = 2 * (user1 - user2) * product
  n = np.sum(product)
  return math.exp(-math.sqrt(np.sum(dif * dif))/n) if n != 0 else 0

def convert_test(test_user):
  test_vector = np.zeros(len(users.columns))
  for anime in test_user:
    test_vector[anime_id_map[anime["anime_id"]]] = anime["score"]

  return test_vector

def predict(test_user):
  anime = np.zeros(len(test_user))
  for user in users_array:
      s = kernel(test_user, user)
      anime = anime + s * user
  anime = anime / total_watchers

  anime = (1 - np.ceil(test_user / 10.0)) * anime
  best_anime_idx = np.argpartition(anime, -5)[-5:]
  best_anime = anime[best_anime_idx]
  best_anime = np.stack((best_anime.T, best_anime_idx.T)).T
  best_anime_idx = best_anime[best_anime[:,0].argsort()[::-1]][:,1].T
  return np.array(best_anime_idx,dtype=np.int)


def convert_predict(anime_id):
  return all_anime[all_anime["anime_id"] == anime_id_map_reverse[anime_id]].iloc[0]["title"]

# for test_user in test_users:
#  print("Predicting for [", ",".join([ all_anime[all_anime["anime_id"] == anime["anime_id"]].iloc[0]["title"] for anime in test_user ]), "]:  ", convert_predict(predict(convert_test(test_user))))
for test_user in test_users:
    pred = predict(convert_test(test_user))
    string = ''
    for i in range(len(pred)):
        string += str(i+1) + '. ' + convert_predict(pred[i]) + "   "
    print(string + '\n')
