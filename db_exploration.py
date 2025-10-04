import sqlite3
import pandas as pd

# Connecting to  the db 
db_path = "merra2_daily.db"  
conn = sqlite3.connect(db_path)

# Table name of the table we want  to  work with 
table_name = "weather"

# Load  the table  partially (the  db has millions  of  rows)
df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 1000000", conn)

# Preview first 10 rows
print(df.head(1000))

# load all  columns 
print(df.columns)

# Number  of  rows we  have 
print("Total rows:", len(df))

conn.close()
