import pandas as pd
import random

products = pd.read_csv("data/raw/products.csv")
stores = pd.read_csv("data/raw/stores.csv")

inventory = []

for _, store in stores.iterrows():
    for _, product in products.iterrows():

        received = random.randint(100, 500)
        sold = random.randint(50, received)
        remaining = received - sold

        inventory.append({
            "Store_ID": store["Store_ID"],
            "Product_ID": product["Product_ID"],
            "Inventory_Received": received,
            "Inventory_Sold": sold,
            "Inventory_Remaining": remaining
        })

inventory_df = pd.DataFrame(inventory)

inventory_df.to_csv(
    "data/raw/inventory.csv",
    index=False
)

print(inventory_df.head())
print(inventory_df.shape)