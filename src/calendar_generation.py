import pandas as pd
from datetime import datetime

# Generate all dates for 2025
dates = pd.date_range(start="2025-01-01", end="2025-12-31")

# Major Indian holidays (can be extended later)
holidays = [
    "2025-01-26",  # Republic Day
    "2025-08-15",  # Independence Day
    "2025-10-02",  # Gandhi Jayanti
    "2025-12-25"   # Christmas
]

calendar = pd.DataFrame({
    "Date": dates,
})

calendar["Day"] = calendar["Date"].dt.day_name()
calendar["Month"] = calendar["Date"].dt.month_name()
calendar["Month_Number"] = calendar["Date"].dt.month
calendar["Quarter"] = calendar["Date"].dt.quarter
calendar["Week_Number"] = calendar["Date"].dt.isocalendar().week
calendar["Year"] = calendar["Date"].dt.year

calendar["Weekend"] = calendar["Day"].isin(["Saturday", "Sunday"]).astype(int)

calendar["Holiday"] = calendar["Date"].astype(str).isin(holidays).astype(int)

calendar.to_csv("data/raw/calendar.csv", index=False)

print("calendar.csv created successfully!")
print(calendar.head())
print(calendar.shape)