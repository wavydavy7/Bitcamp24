import pandas as pd
import sys
import os

def main():

    df = pd.read_csv("../dataset/data.csv")
    print(df.columns)

if __name__ == '__main__':
    main()

