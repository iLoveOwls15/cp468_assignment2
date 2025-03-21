import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("WeatherData_Q3.csv")

#filter data by rain
no_rain = df[df["rain"] == 0]
rain = df[df["rain"] == 1]
#seperate
plt.scatter(no_rain["temp"], no_rain["humid"], color="blue", marker ="s",label = "No rain (y=0)")
plt.scatter(rain["temp"],rain["humid"], color="red",marker="o",label="Rain (y=1)")

plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.title("Weather Data: Rain vs No Rain")

plt.legend()
plt.show()


