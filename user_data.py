import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

rel_data = pd.read_csv("data/UserAnimeList.csv", nrows=100000)

rel_data = rel_data[rel_data["my_score"] != 0.0]

unique_users = rel_data["username"].unique()
unique_anime = rel_data["anime_id"].unique()

anime_id_map = {}
for idx in range(len(unique_anime)):
  anime_id_map[unique_anime[idx]] = idx

rel_data["amime_id"] = rel_data["anime_id"].apply(lambda x: anime_id_map[x])

def user_func(df):
  out = pd.DataFrame([np.zeros(len(unique_anime))], columns=np.arange(0, len(unique_anime)))
  out[df["anime_id"]] = df["my_score"]
  print(out)
  return out

all_user_data = rel_data.groupby(["username"]).apply(user_func)

all_user_data.to_csv("clean_data/users.csv", index=False)
np.savetxt("clean_data/anime_id_map_reverse.csv", unique_anime.astype(np.int), delimiter=",")
