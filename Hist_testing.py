import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

backticks = 20 #Number of ticks we want to have when doign the thing calculation i forgot the name 67

database1 = pd.read_csv('five_arrays1.csv')
database2 = pd.read_csv('five_arrays2.csv')

def reverse_database(database):
    database = database.iloc[::-1].values
    return database

def remove_shit(database): #Shortens the databases for easier testing
    return database.iloc[:5].reset_index(drop=True)

def move_rows(df1, df2, data_tick_num_for_loops): #FOR TESTING PURPOSES BUT CAN BE USED FOR OUR CODE, moves rows from the second database to the first simulating individual ticks like in the main RTI Thingymagigy 

    for _ in range(data_tick_num_for_loops):
        if df2.empty:
            break  

   
        row = df2.iloc[0:1]

        # Add it to the TOP of df1
        df1 = pd.concat([row, df1], ignore_index=True)


        df2 = df2.iloc[1:].reset_index(drop=True)
    print(df1)
    sleep(1)
    return df1.reset_index(drop=True), df2.reset_index(drop=True)

database1 = reverse_database(database1)
database2 = reverse_database(database2)   
database1 = remove_shit(database1)
database2 = remove_shit(database2)