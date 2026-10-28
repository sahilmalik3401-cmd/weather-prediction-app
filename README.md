# Weather Prediction App

A beginner-friendly Python project that explores sample weather measurements,
draws graphs, and trains a small machine-learning model to estimate temperature.
It uses **Pandas**, **Matplotlib**, and **Scikit-learn**. It does not connect to
the internet or require a paid API.

> **Learning example only:** the included data is small, fictional sample data.
> A model trained on it is not suitable for real forecasts or safety decisions.

## What the project does

- Reads `weather_data.csv`, which contains dates, temperatures, humidity, and wind speed.
- Uses Pandas to load and work with the data.
- Draws graphs for temperature, humidity, and wind speed.
- Trains a linear regression model using humidity and wind speed to estimate temperature.
- Tests the model on a held-out portion of the sample data and reports its mean absolute error.
- Makes one example prediction and saves three graph images into the `output` folder.

## Project files

```text
weather-prediction-app/
├── main.py
├── requirements.txt
├── weather_data.csv
└── README.md
```

The `output` folder and its graphs are created the first time you run the program.

## Run it on Windows

1. Install Python 3 from [python.org](https://www.python.org/downloads/windows/).
   During installation, select **Add python.exe to PATH**.
2. Open **Command Prompt** or **PowerShell** and move into the project folder.
   For example:

   ```powershell
   cd "$HOME\weather-prediction-app"
   ```

   If you saved the folder elsewhere, use its actual path instead.
3. (Recommended) Create a virtual environment so the project packages stay together:

   ```powershell
   py -m venv .venv
   ```

4. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt instead and run:

   ```bat
   .venv\Scripts\activate.bat
   ```

5. Install the project packages:

   ```powershell
   py -m pip install -r requirements.txt
   ```

6. Start the app:

   ```powershell
   py main.py
   ```

The program prints the test error and example prediction. Its graph files are:

- `output\temperature_over_time.png`
- `output\humidity_and_wind.png`
- `output\actual_vs_predicted.png`

## Understanding the steps

1. **Load data:** `pd.read_csv` reads the CSV file into a Pandas DataFrame (a table).
   The `date` column is converted to dates so it can be plotted as time.
2. **Choose inputs and output:** humidity and wind speed are the model inputs
   (`features`). Temperature is the value the model tries to predict (`target`).
3. **Split the data:** `train_test_split` keeps 25% of the rows for testing.
   The model learns from the remaining 75%; `random_state=42` makes the split
   repeatable.
4. **Train the model:** `LinearRegression` learns a straight-line relationship
   between humidity, wind speed, and temperature.
5. **Evaluate:** the model predicts temperatures for the test rows.
   Mean absolute error is the average size of the prediction error in degrees
   Celsius; smaller is generally better.
6. **Try a prediction:** the example estimates temperature for 60% humidity and
   6 km/h wind. The number demonstrates how to use the model; it is not a forecast.
7. **Draw graphs:** Matplotlib creates image files showing the sample measurements
   and how predictions compare with actual test temperatures.

## Change the sample data

Open `weather_data.csv` in a text editor or spreadsheet program. Add rows using
the same column names and order:

```text
date,temperature,humidity,wind_speed
2026-02-01,21.5,65,7.0
```

Use temperature in degrees Celsius, humidity as a percentage, and wind speed in
kilometres per hour. This simple example expects numeric values and dates in
`YYYY-MM-DD` format.
