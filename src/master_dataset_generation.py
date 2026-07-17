import pandas as pd
import numpy as np
import random

# Load datasets
products = pd.read_csv("data/raw/products.csv")
stores = pd.read_csv("data/raw/stores.csv")
calendar = pd.read_csv("data/raw/calendar.csv")
weather = pd.read_csv("data/raw/weather.csv")

# Standardize date format
calendar["Date"] = pd.to_datetime(calendar["Date"],format="mixed",dayfirst=True)
weather["Date"] = pd.to_datetime(weather["Date"],format="mixed",dayfirst=True)

# Merge weather with calendar
weather = weather.merge(
    calendar[["Date", "Weekend", "Holiday"]],
    on="Date",
    how="left"
)

rows = []

num_records = 10000

for _ in range(num_records):

    product = products.sample(1).iloc[0]
    store = stores.sample(1).iloc[0]

    city_weather = weather[
        weather["City"] == store["City"]
    ].sample(1).iloc[0]

    inventory_received = random.randint(80, 500)

    demand_score = random.randint(30, 100)

    promotion = random.choice(["Yes", "No"])

    if promotion == "Yes":
        demand_score += random.randint(5, 15)

    inventory_sold = min(
        inventory_received,
        int(inventory_received * demand_score / 100)
    )

    inventory_remaining = inventory_received - inventory_sold

    shelf_life = product["Shelf_Life_Days"]

    expiry_days_left = random.randint(
        0,
        shelf_life
    )

    temperature = city_weather["Temperature"]

    if temperature > 35:
        spoilage = random.uniform(0.20, 0.35)
    elif temperature > 30:
        spoilage = random.uniform(0.10, 0.20)
    else:
        spoilage = random.uniform(0.03, 0.10)

    waste_quantity = (
        inventory_remaining
        * spoilage
        * ((100 - demand_score) / 100)
    )

    waste_quantity += random.uniform(0, 5)

    waste_quantity = round(waste_quantity, 2)

    waste_cost = round(
        waste_quantity * product["Unit_Price"],
        2
    )

    if waste_quantity >= 40:
        waste_risk = "High"
    elif waste_quantity >= 20:
        waste_risk = "Medium"
    else:
        waste_risk = "Low"

    if waste_risk == "High":
        recommendation = "Reduce Order"
    elif demand_score > 80 and inventory_remaining < 30:
        recommendation = "Increase Order"
    else:
        recommendation = "Maintain Stock"
    rows.append({
        "Date": city_weather["Date"].strftime("%Y-%m-%d"),

        "Store_ID": store["Store_ID"],
        "Store_Name": store["Store_Name"],
        "City": store["City"],
        "State": store["State"],
        "Region": store["Region"],
        "Store_Type": store["Store_Type"],
        "Store_Size": store["Store_Size"],

        "Product_ID": product["Product_ID"],
        "Product_Name": product["Product_Name"],
        "Category": product["Category"],
        "Brand": product["Brand"],

        "Shelf_Life_Days": shelf_life,
        "Expiry_Days_Left": expiry_days_left,

        "Unit_Price": product["Unit_Price"],

        "Inventory_Received": inventory_received,
        "Inventory_Sold": inventory_sold,
        "Inventory_Remaining": inventory_remaining,

        "Demand_Score": demand_score,
        "Promotion": promotion,

        "Temperature": city_weather["Temperature"],
        "Humidity": city_weather["Humidity"],
        "Rainfall": city_weather["Rainfall"],
        "Weather": city_weather["Weather"],

        "Weekend": city_weather["Weekend"],
        "Holiday": city_weather["Holiday"],

        "Waste_Quantity": waste_quantity,
        "Waste_Cost": waste_cost,
        "Waste_Risk": waste_risk,

        "Inventory_Recommendation": recommendation
    })

# -----------------------------
# Create Final Dataset
# -----------------------------
master_df = pd.DataFrame(rows)

master_df.to_csv(
    "data/processed/food_waste_master_dataset.csv",
    index=False
)

print("\n✅ Dataset Generated Successfully!")
print(master_df.head())
print("\nDataset Shape:", master_df.shape)       