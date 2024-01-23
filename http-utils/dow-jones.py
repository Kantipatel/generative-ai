import requests as req
import pandas as pd


def load_data():
    dow_url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
    sp_response = req.get(dow_url)   

    df = pd.read_html(sp_response.content)
    dt_companies = df[1]
    return dt_companies


dow_table = load_data()
print(dow_table)
