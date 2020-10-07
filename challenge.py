#------------------------------------------------------------
# DataEngineerChallenge
# https://github.com/Pay-Baymax/DataEngineerChallenge
#------------------------------------------------------------

import pandas as pd
import numpy as np
from functools import reduce
from sklearn.ensemble import RandomForestRegressor

def group_by_window(group, duration):
    # Group by temporal window
    start = group.iloc[0]["Timestamp"]
    delta = np.timedelta64(duration, 's')
    return (group["Timestamp"] > start + delta).cumsum()

def get_session_duration(df_):
    # Get the duration of a session.
    df_ = df_.sort_values(by=['Timestamp'])
    start = df_['Timestamp'].values[0]
    end = df_['Timestamp'].values[-1]
    duration = pd.Timedelta(end-start)
    return duration

def main():

    #############################################################
    # Part 1
    #############################################################

    #------------------------------------------------------------
    # Load and process the data
    #------------------------------------------------------------

    filename = 'data/2015_07_22_mktplace_shop_web_log_sample.log'

    # For test purposes: head -100000 2015_07_22_mktplace_shop_web_log_sample.log > sample.log
    # filename = 'data/sample.log'

    # Load the log file into a dataframe
    columns = ['Timestamp', 'Marketplace', 'Body'] # Initial columns
    weblog_df = pd.read_fwf(filename, names=columns)

    # Convert timestamp to datetime
    weblog_df["Timestamp"] = pd.to_datetime(weblog_df["Timestamp"])

    # Get the IPs from 'Body'
    weblog_df['IP'] = weblog_df.Body.str.split(' ').str[0]

    # Get URLs using a regexp
    url_pattern = '(https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}[-a-zA-Z0-9()@:%_+.~#?&/=]*)'
    weblog_df['URL'] = weblog_df.Body.str.extract(url_pattern, expand=True)

    # Sort by 'Timestamp'
    weblog_df = weblog_df.sort_values(by = ['IP', 'Timestamp'])

    # Store in a working dataframe
    df = weblog_df

    #------------------------------------------------------------
    # (1) Sessionize by IP
    # Sessions are defiend based on requests made within a
    # 'window_length' seconds window coming from the same IP address
    #------------------------------------------------------------
    window_length = 30 # seconds
    df["Session"] = df.groupby("IP").apply(group_by_window, window_length).reset_index(level=0, drop=True)

    #------------------------------------------------------------
    # (2) Average session duration (+ median, max, and min)
    #------------------------------------------------------------
    sessions_durations = df.groupby('Session').apply(get_session_duration)
    print (sessions_durations.describe())

    #------------------------------------------------------------
    # (3) Unique URL visits per session.
    # Count a hit to a unique URL only once per session
    #------------------------------------------------------------

    unique_urls_per_sessions = df.groupby(['Session'])['URL'].agg(['nunique'])
    print (unique_urls_per_sessions)

    #------------------------------------------------------------
    # (4) The 'top_users' top users/IPs with the longest session times
    #------------------------------------------------------------

    top_users = 10
    users_session_durations = df.groupby('IP').apply(get_session_duration)\
                                .sort_values(ascending=False)\
                                .head(top_users)
    print (users_session_durations)

    #############################################################
    # Part 2
    #############################################################

    #------------------------------------------------------------
    # (1) Predict the expected load in the next minute
    # with a regression on 3 statistical features: mean, std, max
    #------------------------------------------------------------

    df = df[['Timestamp', 'IP']]
    df = df.sort_values(by=['Timestamp'])

    # Group by 'prediction_window' and count the number of hits per user/IP
    prediction_window = '60s'

    # Hit per minute
    f0 = df.set_index('Timestamp')\
                    .groupby([pd.Grouper(freq=prediction_window), 'IP'])\
                    .agg({'IP': 'count'})\
                    .rename(columns={'IP': 'HPM'})\
                    .reset_index() \
                    .rename(columns={'Timestamp': 'Timewindow'})

    # Extract the three features from f0
    f1 = f0.groupby("Timewindow").agg({'HPM' : 'std'}).rename(columns={'HPM': 'stdHPM'})
    f2 = f0.groupby("Timewindow").agg({'HPM' : 'mean'}).rename(columns={'HPM':'meanHPM'})
    f3 = f0.groupby("Timewindow").agg({'HPM' : 'max'}).rename(columns={'HPM': 'maxHPM'})

    # Join f0_3 into fs based on 'Timewindow'
    fs = reduce(lambda l, r: pd.merge(l, r, on='Timewindow'), [f0, f1, f2, f3])

    # Rename, flatten, and clean
    fs = fs.rename(columns={'stdHPM':'stdHPS',
                            'meanHPM':'meanHPS',
                            'maxHPM':'maxHPS',
                            'HPM':'currHPS'})\
            .reset_index()\
            .dropna()

    # Convert from HPM to HPS
    fs[['stdHPS', 'meanHPS', 'maxHPS', 'currHPS']] /= int(prediction_window[:-1])

    # Adding new label for training. Shift from next HPS
    fs['nextHPS'] = fs['currHPS'].shift(-1)
    data = fs[['stdHPS', 'meanHPS', 'maxHPS', 'currHPS', 'nextHPS']].to_numpy()

    # Split data into featues and labels. Remove last entry (-1)
    #   feature: stdHPS, meanHPS, maxHPS, currHPS
    #   label: nextHPS
    X, y = data[:-1,:4], data[:-1,4]

    # Fit into the regressor
    regressor = RandomForestRegressor(max_depth=3, random_state=0)
    regressor.fit(X, y)
    print(regressor.predict(X[[-1]]))

    #------------------------------------------------------------
    # (2) Predict the session length for a given IP
    #------------------------------------------------------------

    session_length_per_IP = df.groupby('IP').apply(get_session_duration)\
                                .sort_values(ascending=False) \
                                .reset_index()\
                                .rename(columns={0: 'Duration'})
    # Remove inactive users on average
    session_length_per_IP = session_length_per_IP[session_length_per_IP != '00:00:00' ].dropna()
    print (session_length_per_IP)

    #------------------------------------------------------------
    # (3) Predict the number of unique URL visits by a given IP
    #------------------------------------------------------------

    # Average unique url visits of for each user/IP
    df = weblog_df[['IP', 'URL']]
    average_url_visit = df.groupby(['IP', 'URL'])['IP']\
                            .count()\
                            .groupby('IP')\
                            .mean()
    print (average_url_visit)


main()
#
