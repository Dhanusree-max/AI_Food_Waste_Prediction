import pandas as pd
import random

products = pd.read_csv("data/raw/products.csv")

promotions = []

for _, product in products.iterrows():

    for month in range(1,13):

        promotion = random.choice(["Yes","No"])

        if promotion=="Yes":
            discount=random.choice([5,10,15,20,25,30])
        else:
            discount=0

        promotions.append({
            "Product_ID":product["Product_ID"],
            "Month":month,
            "Promotion":promotion,
            "Discount_Percentage":discount
        })

promotion_df=pd.DataFrame(promotions)

promotion_df.to_csv(
    "data/raw/promotions.csv",
    index=False
)

print(promotion_df.head())
print(promotion_df.shape)