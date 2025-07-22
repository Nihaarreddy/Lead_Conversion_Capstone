# src/data_ingestion/db.py
# Import required libraries
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd

# Configuration for PostgreSQL connection
DB_USER = 'postgres'
DB_PASS = 'Nihaar6'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'postgres'

DB_URL = f'postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Global SQLAlchemy engine and sessionmaker
engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#Method to return engine
def get_engine():
    return engine
#method to return session
def get_session():
    return SessionLocal()

# Method to read data from PostgreSQL table and store it in a dataframa
def read_table(table_name, sql_query=None):
    """
    Read full table or custom query as a pandas DataFrame.
    """
    if sql_query:
        return pd.read_sql(sql_query, engine)
    else:
        return pd.read_sql_table(table_name, engine)

#method to write the data from dataframe to PostgreSQL table
def write_table(df, table_name, if_exists='append'):
    """
    Write or overwrite DataFrame to the db table.
    """
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
