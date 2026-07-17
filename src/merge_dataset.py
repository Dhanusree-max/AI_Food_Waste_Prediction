import pandas as pd

# Load datasets
products = pd.read_csv("data/raw/products.csv")
stores = pd.read_csv("data/raw/stores.csv")
inventory = pd.read_csv("data/raw/inventory.csv")
demand = pd.read_csv("data/raw/demand.csv")
promotions = pd.read_csv("data/raw/promotions.csv")
weather = pd.read_csv("data/raw/weather.csv")
calendar = pd.read_csv("data/raw/calendar.csv")

print("Datasets Loaded Successfully")
# Merge Inventory + Products
df = inventory.merge(
    products,
    on="Product_ID",
    how="left"
)

# Merge Stores
df = df.merge(
    stores,
    on="Store_ID",
    how="left"
)

# Merge Demand
df = df.merge(
    demand,
    on=["Store_ID", "Product_ID"],
    how="left"
)
# Create Month column

weather["Date"] = pd.to_datetime(weather["Date"], format="mixed", dayfirst=True)
weather["Month"] = weather["Date"].dt.month

calendar["Date"] = pd.to_datetime(calendar["Date"], format="mixed", dayfirst=True)
calendar["Month"] = calendar["Date"].dt.month

# Merge Promotions
df["Month"] = 1

df = df.merge(
    promotions,
    on=["Product_ID", "Month"],
    how="left"
)
# Merge Weather

weather = weather.sample(len(df), replace=True).reset_index(drop=True)

df = pd.concat(
    [df.reset_index(drop=True),
     weather[["Date","Temperature","Humidity","Rainfall","Weather"]]],
    axis=1
)
# Merge Calendar

calendar = calendar.sample(len(df), replace=True).reset_index(drop=True)

df["Weekend"] = calendar["Weekend"]
df["Holiday"] = calendar["Holiday"]
# Create Target Columns

df["Expiry_Days_Left"] = (
    df["Shelf_Life_Days"] *
    0.6
).astype(int)

df["Waste_Quantity"] = (
    (
        df["Inventory_Remaining"]
        * (100 - df["Demand_Score"])
        / 100
    )
    * 0.25
).round(2)

df["Waste_Cost"] = (
    df["Waste_Quantity"]
    * df["Unit_Price"]
).round(2)

df["Waste_Risk"] = pd.cut(
    df["Waste_Quantity"],
    bins=[-1,20,40,1000],
    labels=["Low","Medium","High"]
)

df["Inventory_Recommendation"] = df["Waste_Risk"].map({
    "High":"Reduce Order",
    "Medium":"Maintain Stock",
    "Low":"Increase Order"
})
# Save Final Dataset

df.to_csv(
    "data/processed/food_waste_master_dataset.csv",
    index=False
)

print(df.head())

print("\nShape :", df.shape)

print("\nDataset Generated Successfully!")