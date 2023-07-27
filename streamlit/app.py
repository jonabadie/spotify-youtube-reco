import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def preprocessing(df):
    columns = [
        'Danceability',
        'Energy',
        'Speechiness',
        'Acousticness',
        'Instrumentalness',
        'Valence',
        'Tempo'
    ]
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns)
    return df

def get_data():
    df = pd.read_csv('data/data_clean.csv')
    return df

def nearest_songs(idx, df):
    df = preprocessing(df)
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
    nbrs = nbrs.fit(df)
    _, indices = nbrs.kneighbors(df)
    res = np.delete(indices[idx], np.where(indices[idx] == idx))
    return res

st.title('Song Recommender')
st.write('This is a simple song recommender built with Streamlit and Scikit-learn.')

df = get_data()
song = st.selectbox('Select a song', options=[f"{row['Artist']} - {row['Track']}" for _, row in df.iterrows()])
if song:
    idx = df[df['Artist'] + ' - ' + df['Track'] == song].index[0]
    indices = nearest_songs(idx, df)
    cols = st.columns(2)
    cols[0].video(df.iloc[indices[0]]['Url_youtube'])
    cols[1].video(df.iloc[indices[1]]['Url_youtube'])
    cols = st.columns(3)
    cols[0].video(df.iloc[indices[2]]['Url_youtube'])
    cols[1].video(df.iloc[indices[3]]['Url_youtube'])
    cols[2].video(df.iloc[indices[4]]['Url_youtube'])
    st.write(df.iloc[indices])