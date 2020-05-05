import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

rel_data = pd.read_csv("data/UserAnimeList.csv", usecols=["username", "anime_id", "my_score"])

print("Finished Initial Read")

print("Read", len(rel_data.index), "rows")

rel_data = rel_data[rel_data["my_score"] != 0.0]

unique_users = rel_data["username"].unique()
unique_anime = rel_data["anime_id"].unique()
all_user_data = []

print("Unique Users:",len(unique_users))
print("Unique Anime:",len(unique_anime))


anime_id_map = {}
for idx in range(len(unique_anime)):
  anime_id_map[unique_anime[idx]] = idx

print("Finished Anime Mapping", "Creating User Vectors")

count = 0;
for user in unique_users:
  start = count
  while (count < len(rel_data.index) and rel_data["username"].iloc[count] == user):
    count += 1
    if (count % 5000000 == 0):
      print("Read", count, "rows")
  user_anime = rel_data[start:count][["anime_id", "my_score"]]
  user_data = pd.Series(np.zeros(len(unique_anime)))
  # user_anime["my_score"][user_anime["my_score"] == 0.0] = 5
  user_data[user_anime["anime_id"].apply(lambda x: anime_id_map[x])] = user_anime["my_score"]
  all_user_data.append(user_data)


print("Printing and Saving CSV")

# all_user_data = pd.DataFrame(all_user_data)
# print(all_user_data)

step = 6000000
pd.DataFrame(all_user_data[:step]).to_csv("clean_data/users1.csv", index=False)
pd.DataFrame(all_user_data[step:2 * step]).to_csv("clean_data/users2.csv", index=False)
pd.DataFrame(all_user_data[2 * step:3 * step]).to_csv("clean_data/users3.csv", index=False)
pd.DataFrame(all_user_data[3 * step:]).to_csv("clean_data/users4.csv", index=False)

np.savetxt("clean_data/anime_id_map_reverse.csv", unique_anime.astype(np.int), delimiter=",")
