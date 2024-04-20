import mne.io
import pandas as pd
import sys
import os

def main():

    df = pd.read_csv("./dataset/test.csv")
    print(df)

if __name__ == '__main__':
    main()

