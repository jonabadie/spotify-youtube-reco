import streamlit as st
from streamlit_tags import st_tags
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
        # 'Instrumentalness',
        # 'Valence',
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

st.set_page_config(page_title='Song Recommender', layout='wide')
st.title('Song Recommender')
st.write('This is a simple song recommender built with Streamlit and Scikit-learn.')

df = get_data()
cols = st.columns(2)

artist = cols[0].selectbox('Select an artist', options=df['Artist'].unique())
artist_songs = df[df['Artist'] == artist]

song = cols[1].selectbox('Select a song', options=[f"{row['Track']}" for _, row in artist_songs.iterrows()])
if song:
    idx = df[df['Artist'] + df['Track'] == artist + song].index[0]
    indices = nearest_songs(idx, df)
    st.title('**Recommended songs**')
    cols = st.columns(2)
    cols[0].video(df.iloc[indices[0]]['Url_youtube'])
    cols[1].video(df.iloc[indices[1]]['Url_youtube'])
    cols = st.columns(3)
    cols[0].video(df.iloc[indices[2]]['Url_youtube'])
    cols[1].video(df.iloc[indices[3]]['Url_youtube'])
    cols[2].video(df.iloc[indices[4]]['Url_youtube'])
    # st.write(df.iloc[indices])
    st.title('Chosen song')
    st.video(df.iloc[idx]['Url_youtube'])