import  sqlite3
import pandas as pd

# connect to the db  
db_path = "merra2_daily.db"

conn = sqlite3.connect(db_path)

# table  we will  be working with 
table_Name = 'weather'

# Loading the first  2  years of data for training 
# pd  we  might have  to use 3 depending on the results of training 


# it  takes  abot 1  min to load the  data 




# Lets  visualize  one  lat and lot over  1 year with matpotlib


import matplotlib.pyplot as plt

target_lat = 25.0
target_lon = -80
year = 2022

df = pd.read_sql_query(f"SELECT date, T2MMAX, T2MMEAN, T2MMIN, HOURNORAIN FROM weather WHERE lat = 25.0 AND lon = -80.0 AND strftime('%Y', date) = '2025' ORDER BY date; ", conn)


# Preview first 10 rows
print(df.head(10))

# load all  columns 
print(df.columns)

# Number  of  rows we  have 
print("Total rows:", len(df))





