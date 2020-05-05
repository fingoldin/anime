import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

rel_data = pd.read_csv("data/UserAnimeList.csv", nrows=100000)

rel_data = rel_data[rel_data["my_score"] != 0.0]

unique_users = rel_data["username"].unique()
unique_anime = rel_data["anime_id"].unique()
all_user_data = []

anime_id_map = {}
for idx in range(len(unique_anime)):
  anime_id_map[unique_anime[idx]] = idx

for user in unique_users:
  user_anime = rel_data[rel_data["username"] == user][["anime_id", "my_score"]]
  user_data = pd.Series(np.zeros(len(unique_anime)))
  # user_anime["my_score"][user_anime["my_score"] == 0.0] = 5
  user_data[user_anime["anime_id"].apply(lambda x: anime_id_map[x])] = user_anime["my_score"]

  all_user_data.append(user_data)

all_user_data = pd.DataFrame(all_user_data)
print(all_user_data)

all_user_data.to_csv("clean_data/users.csv", index=False)
np.savetxt("clean_data/anime_id_map_reverse.csv", unique_anime.astype(np.int), delimiter=",")
