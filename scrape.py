from requests import get
import json
from bs4 import BeautifulSoup

def get_user_anime(username):
  res = get("https://myanimelist.net/animelist/" + username)
  html_soup = BeautifulSoup(res.text, "html.parser")
  table = html_soup.find("table")
  data = json.loads(table["data-items"])

  return [ { "anime_id": row["anime_id"], "score": row["score"], "title": row["anime_title"] } for row in data ]

