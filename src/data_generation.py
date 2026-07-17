import pandas as pd
import random
from datetime import datetime,timedelta

products_df = pd.read_csv("data/raw/products.csv")
print(products_df.head())
print(products_df.shape)