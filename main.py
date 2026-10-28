"""Explore sample weather data and train a simple temperature model."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FILE = PROJECT_FOLDER / "weather_data.csv"
OUTPUT_FOLDER = PROJECT_FOLDER / "output"


def main():
    # Load the CSV file into a Pandas table.
    weather = pd.read_csv(DATA_FILE, parse_dates=["date"])

    # These are the measurements the model will use to estimate temperature.
    features = ["humidity", "wind_speed"]
    target = "temperature"
    x = weather[features]
    y = weather[target]

    # Keep some rows aside so we can test the model on data it did not train on.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    # Linear regression learns a simple relationship between the inputs and temperature.
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Check how far the test predictions are from the recorded temperatures on average.
    predictions = model.predict(x_test)
    error = mean_absolute_error(y_test, predictions)

    print("Weather Prediction App")
    print("----------------------")
    print(f"Weather records loaded: {len(weather)}")
    print(f"Mean absolute error on test data: {error:.2f} °C")

    # Demonstrate a prediction for a made-up set of weather conditions.
    example_weather = pd.DataFrame(
        {"humidity": [60], "wind_speed": [6.0]}
    )
    predicted_temperature = model.predict(example_weather)[0]
    print(
        "Predicted temperature for 60% humidity and 6 km/h wind: "
        f"{predicted_temperature:.1f} °C"
    )

    # Create the folder for the graph image files if it does not exist yet.
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Graph 1: show how temperature changes across the sample dates.
    plt.figure(figsize=(9, 4))
    plt.plot(weather["date"], weather["temperature"], marker="o", color="tomato")
    plt.title("Sample Temperature Over Time")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "temperature_over_time.png")
    plt.close()

    # Graph 2: compare humidity and wind speed across the same dates.
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(weather["date"], weather["humidity"], color="royalblue")
    axes[0].set_title("Humidity Over Time")
    axes[0].set_ylabel("Humidity (%)")
    axes[1].plot(weather["date"], weather["wind_speed"], color="seagreen")
    axes[1].set_title("Wind Speed Over Time")
    axes[1].set_ylabel("Wind speed (km/h)")
    axes[1].set_xlabel("Date")
    axes[1].tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(OUTPUT_FOLDER / "humidity_and_wind.png")
    plt.close(figure)

    # Graph 3: see how the model's estimates compare with the actual test values.
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, color="purple")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--")
    plt.title("Actual and Predicted Temperatures")
    plt.xlabel("Actual temperature (°C)")
    plt.ylabel("Predicted temperature (°C)")
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "actual_vs_predicted.png")
    plt.close()

    print(f"Graphs saved in: {OUTPUT_FOLDER}")
    print("Note: this is a learning example, not a real weather forecast.")


if __name__ == "__main__":
    main()
