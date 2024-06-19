import pandas as pd
from load import load_data
from transfer import transfer


if __name__=="__main__":
    df = load_data("data/data.json")
    df = transfer(df)
    print(df.columns)
    print(df.head())
    df.to_csv('transfer.csv')