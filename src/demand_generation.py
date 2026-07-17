import pandas as pd
import random

products=pd.read_csv("data/raw/products.csv")
stores=pd.read_csv("data/raw/stores.csv")

demand=[]

for _,store in stores.iterrows():

    for _,product in products.iterrows():

        score=random.randint(30,100)

        if score>=80:
            level="High"
        elif score>=60:
            level="Medium"
        else:
            level="Low"

        demand.append({

            "Store_ID":store["Store_ID"],
            "Product_ID":product["Product_ID"],
            "Demand_Score":score,
            "Demand_Level":level

        })

demand_df=pd.DataFrame(demand)

demand_df.to_csv(
    "data/raw/demand.csv",
    index=False
)

print(demand_df.head())
print(demand_df.shape)