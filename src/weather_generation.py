import pandas as pd
import random

# Read stores
stores = pd.read_csv("data/raw/stores.csv")

# Generate dates
dates = pd.date_range(start="2025-01-01", end="2025-12-31")

weather_conditions = ["Sunny", "Cloudy", "Rainy", "Stormy", "Foggy"]

weather_data = []

for _, store in stores.iterrows():
    city = store["City"]

    for date in dates:
        month = date.month

        # Seasonal temperatures
        if month in [3, 4, 5]:
            temp = random.randint(30, 42)
        elif month in [6, 7, 8, 9]:
            temp = random.randint(24, 35)
        elif month in [10, 11]:
            temp = random.randint(22, 32)
        else:
            temp = random.randint(15, 28)

        humidity = random.randint(40, 95)

        if humidity > 80:
            weather = "Rainy"
        elif humidity > 65:
            weather = "Cloudy"
        elif temp > 35:
            weather = "Sunny"
        else:
            weather = random.choice(weather_conditions)

        rainfall = 0

        if weather == "Rainy":
            rainfall = random.randint(10, 80)

        weather_data.append([
            date.date(),
            city,
            temp,
            humidity,
            rainfall,
            weather
        ])

weather_df = pd.DataFrame(
    weather_data,
    columns=[
        "Date",
        "City",
        "Temperature",
        "Humidity",
        "Rainfall",
        "Weather"
    ]
)

weather_df.to_csv("data/raw/weather.csv", index=False)

print(weather_df.head())
print(weather_df.shape)
print("weather.csv generated successfully!")