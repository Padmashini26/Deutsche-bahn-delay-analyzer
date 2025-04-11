# train_delay_analysis.ipynb (Colab-compatible version)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Load dataset
df = pd.read_csv("train_delays.csv", parse_dates=["scheduled_time"])

# Extract useful time features
df["hour"] = df["scheduled_time"].dt.hour
df["weekday"] = df["scheduled_time"].dt.day_name()

# Overview
print("Dataset shape:", df.shape)
print("\nSample rows:")
print(df.head())

# Average delay by station
station_delay = df.groupby("station")["delay_minutes"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
station_delay.plot(kind="bar", color="teal")
plt.title("Average Delay by Station")
plt.ylabel("Average Delay (minutes)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Average delay by hour
hourly_delay = df.groupby("hour")["delay_minutes"].mean()
plt.figure(figsize=(10, 5))
hourly_delay.plot(kind="line", marker="o", color="orange")
plt.title("Average Delay by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Delay (minutes)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Heatmap: delay by weekday and hour
pivot = df.pivot_table(index="weekday", columns="hour", values="delay_minutes", aggfunc="mean")
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
pivot = pivot.reindex(weekday_order)

plt.figure(figsize=(12, 6))
sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.3, annot=True, fmt=".1f")
plt.title("Heatmap of Delays by Weekday and Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Weekday")
plt.tight_layout()
plt.show()
