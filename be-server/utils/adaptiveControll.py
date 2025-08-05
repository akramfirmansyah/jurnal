from datetime import datetime
from dotenv import load_dotenv, set_key
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

import csv
import influxdb_client
import numpy as np
import os
import pandas as pd
import pickle
import skfuzzy as fuzz
import warnings
import xgboost as xgb

warnings.filterwarnings("ignore")


class AdaptiveControll:
    def __init__(self) -> None:
        self.logs_dir = "logs/"
        self.model_dir = "model/"
        self.plot_dir = "plots/"

    def FuzzyLogicNutrientPump(
        self, airTemperature: int | float, humidity: int | float
    ) -> int:
        # Check input value
        if airTemperature < 0:
            airTemperature = 0
        elif airTemperature > 45:
            airTemperature = 45

        if humidity < 0:
            humidity = 0
        elif humidity > 100:
            humidity = 100

        # Defining membership scope
        memberAirTemperature = np.arange(0, 45 + 1)
        memberHumidity = np.arange(0, 100 + 1)
        memberDelay = np.arange(5, 45 + 1)

        # Create Membership Air Temperature
        memberAirTemperature_cool = fuzz.trapmf(memberAirTemperature, [0, 0, 13, 18])
        memberAirTemperature_optimal = fuzz.trapmf(
            memberAirTemperature, [13, 18, 25, 30]
        )
        memberAirTemperature_hot = fuzz.trapmf(memberAirTemperature, [25, 30, 45, 45])

        # Create Membership Humidity
        memberHumidity_dry = fuzz.trapmf(memberHumidity, [0, 0, 55, 60])
        memberHumidity_optimal = fuzz.trapmf(memberHumidity, [55, 60, 80, 85])
        memberHumidity_moist = fuzz.trapmf(memberHumidity, [80, 85, 100, 100])

        # Create Membership Spraying Delay
        memberDelay_short = fuzz.trapmf(memberDelay, [5, 5, 25, 30])
        memberDelay_normal = fuzz.trimf(memberDelay, [25, 30, 35])
        memberDelay_long = fuzz.trapmf(memberDelay, [30, 35, 45, 45])

        # Calculating the degree of membership Air Temperature
        memberAirTemperature_cool_degree = fuzz.interp_membership(
            memberAirTemperature, memberAirTemperature_cool, airTemperature
        )
        memberAirTemperature_optimal_degree = fuzz.interp_membership(
            memberAirTemperature, memberAirTemperature_optimal, airTemperature
        )
        memberAirTemperature_hot_degree = fuzz.interp_membership(
            memberAirTemperature, memberAirTemperature_hot, airTemperature
        )

        # Calculating the degree of membership Humidity
        memberHumidity_dry_degree = fuzz.interp_membership(
            memberHumidity, memberHumidity_dry, humidity
        )
        memberHumidity_optimal_degree = fuzz.interp_membership(
            memberHumidity, memberHumidity_optimal, humidity
        )
        memberHumidity_moist_degree = fuzz.interp_membership(
            memberHumidity, memberHumidity_moist, humidity
        )

        # Create rules
        rule1 = np.fmin(memberAirTemperature_hot_degree, memberHumidity_dry_degree)
        rule2 = np.fmin(memberAirTemperature_hot_degree, memberHumidity_optimal_degree)
        rule3 = np.fmin(memberAirTemperature_hot_degree, memberHumidity_moist_degree)
        rule4 = np.fmin(memberAirTemperature_optimal_degree, memberHumidity_dry_degree)
        rule5 = np.fmin(
            memberAirTemperature_optimal_degree, memberHumidity_optimal_degree
        )
        rule6 = np.fmin(
            memberAirTemperature_optimal_degree, memberHumidity_moist_degree
        )
        rule7 = np.fmin(memberAirTemperature_cool_degree, memberHumidity_dry_degree)
        rule8 = np.fmin(memberAirTemperature_cool_degree, memberHumidity_optimal_degree)
        rule9 = np.fmin(memberAirTemperature_cool_degree, memberHumidity_moist_degree)

        # Mamdani Inference System
        delay_short = np.fmax(rule1, np.fmax(rule2, rule4))  # Rule 1, 2, 4
        delay_normal = np.fmax(rule3, np.fmax(rule5, rule7))  # Rule 3, 5, 7
        delay_long = np.fmax(rule6, np.fmax(rule8, rule9))  # Rule 6, 8, 9

        activation_short = np.fmin(delay_short, memberDelay_short)
        activation_normal = np.fmin(delay_normal, memberDelay_normal)
        activation_long = np.fmin(delay_long, memberDelay_long)

        # Aggregated
        aggregated = np.fmax(
            activation_short, np.fmax(activation_normal, activation_long)
        )

        # Calculate spraying delay (Defuzzification with Centroid Method)
        spraying_delay = fuzz.defuzz(memberDelay, aggregated, "centroid")

        # Return result as Decimal
        return int(spraying_delay)

    def get_data(
        self, measurement: str = "first", field: str = "airTemperature"
    ) -> pd.DataFrame:
        load_dotenv()

        token = os.getenv("TOKEN")
        url = os.getenv("URL")
        org = os.getenv("ORG")
        bucket = os.getenv("BUCKET")
        start_date = os.getenv("START_DATE")

        client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)

        query_api = client.query_api()

        query = f"""from(bucket: "{bucket}")
                |> range(start: {start_date})
                |> filter(fn: (r) => r["_measurement"] == "{measurement}")
                |> filter(fn: (r) => r["_field"] == "{field}")
                |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
                |> yield(name: "mean")"""

        tables = query_api.query(query=query, org=org)

        data = []
        for table in tables:
            for record in table.records:
                data.append(
                    {
                        "datetime": record.get_time(),
                        f"{field}": round(record.get_value(), 2),
                    }
                )

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
        df = df.set_index("datetime").asfreq("1min")

        return df

    def preprocessing(self) -> tuple:
        # Get data from InfluxDB
        df_airTemp = self.get_data(measurement="first", field="airTemperature")
        df_humidity = self.get_data(measurement="first", field="humidity")

        # Replace outlier with
        df_airTemp = self.replace_outlier(df_airTemp, "airTemperature")
        df_humidity = self.replace_outlier(df_humidity, "humidity")

        # Fill NaN with mean HH:MM
        df_airTemp = self.fill_na(df_airTemp, "airTemperature")
        df_humidity = self.fill_na(df_humidity, "humidity")

        # Create features
        df_airTemp = self.create_features(df_airTemp)
        df_humidity = self.create_features(df_humidity)

        return df_airTemp, df_humidity

    def replace_outlier(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        temp = data.copy()

        q1, q3 = temp[f"{column}"].quantile(0.25), temp[f"{column}"].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        temp.loc[temp[f"{column}"] > upper_limit, f"{column}"] = upper_limit
        temp.loc[temp[f"{column}"] < lower_limit, f"{column}"] = lower_limit

        return temp

    def fill_na(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        # Get mean HH:MM
        resampled_df = data.copy()
        resampled_df = resampled_df.resample("min").mean()
        resampled_df["datetime"] = resampled_df.index
        resampled_df["hour_minutes"] = resampled_df["datetime"].dt.strftime("%H:%M")
        resampled_df = resampled_df.groupby("hour_minutes").mean()
        resampled_df = resampled_df.drop(columns=["datetime"])
        resampled_df = resampled_df.round({f"{column}": 2})

        # Fill NaN with mean HH:MM
        df_fillna = data.copy()
        df_fillna["hour_minutes"] = df_fillna.index.strftime("%H:%M")
        df_fillna[f"{column}"] = df_fillna.apply(
            lambda row: (
                row[f"{column}"]
                if pd.notnull(row[f"{column}"])
                else resampled_df.loc[row["hour_minutes"], f"{column}"]
            ),
            axis=1,
        )
        df_fillna = df_fillna.drop(columns=["hour_minutes"])

        return df_fillna

    def create_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        data = dataframe.copy()
        data["year"] = data.index.year
        data["month"] = data.index.month
        data["day"] = data.index.day
        data["hour"] = data.index.hour
        data["minute"] = data.index.minute
        data["dayofyear"] = data.index.dayofyear
        data["dayofweek"] = data.index.dayofweek

        return data

    def create_model(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame
    ) -> xgb.XGBRegressor:
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=10,
            subsample=1,
            colsample_bytree=1,
            gamma=2.5,
            reg_alpha=0,
            reg_lambda=7,
        )

        model.fit(X_train, y_train)

        return model

    def save_model(self, model: xgb.XGBRegressor, model_name: str) -> None:
        filepath = Path(f"{self.model_dir}XGBoost_{model_name}.pkl")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(model, open(filepath, "wb"))

    def save_matrix(
        self, model, X_test: pd.DataFrame, y_test: pd.DataFrame, column_name: str
    ) -> None:
        test_time = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        model = model

        y_test["prediction"] = model.predict(X_test)

        mae = mean_absolute_error(y_test[f"{column_name}"], y_test["prediction"])
        mape = mean_absolute_percentage_error(
            y_test[f"{column_name}"], y_test["prediction"]
        )
        rmse = root_mean_squared_error(y_test[f"{column_name}"], y_test["prediction"])

        # Define the header and data
        header = ["datetime", "mae", "mape", "rmse"]
        data = [test_time, mae, mape, rmse]

        # Check if the file exists and has a header
        file_exists = os.path.isfile(f"{self.logs_dir}metrix_{column_name}.csv")
        header_exists = False

        if file_exists:
            with open(f"{self.logs_dir}metrix_{column_name}.csv", "r") as file:
                reader = csv.reader(file)
                # Check the first row to see if it's the header
                header_exists = any(row == header for row in reader)

        # Open the file in append mode
        with open(f"{self.logs_dir}metrix_{column_name}.csv", "a", newline="") as file:
            writer = csv.writer(file)
            # Write the header if it doesn't exist
            if not header_exists:
                writer.writerow(header)
            # Write the data
            writer.writerow(data)

    def training_model(self) -> None:
        # Retrieving preprocessed data
        df_temp, df_hum = self.preprocessing()

        # Split data to X and y
        X_temp, y_temp = (
            df_temp.drop(columns=["airTemperature"]),
            df_temp[["airTemperature"]],
        )
        X_hum, y_hum = (
            df_hum.drop(columns=["humidity"]),
            df_hum[["humidity"]],
        )

        # Split data to train and test
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
            X_temp, y_temp, test_size=0.3, shuffle=False
        )
        X_train_hum, X_test_hum, y_train_hum, y_test_hum = train_test_split(
            X_hum, y_hum, test_size=0.3, shuffle=False
        )

        model_temp = self.create_model(X_train_temp, y_train_temp)
        model_hum = self.create_model(X_train_hum, y_train_hum)

        self.save_model(model_temp, "airTemperature")
        self.save_model(model_hum, "humidity")

        self.save_matrix(model_temp, X_test_temp, y_test_temp, "airTemperature")
        self.save_matrix(model_hum, X_test_hum, y_test_hum, "humidity")

    def load_model(self, filepath: str) -> xgb.XGBRegressor:
        model = pickle.load(open(filepath, "rb"))

        return model

    def single_predict(self, model: xgb.XGBRegressor, column_name: str) -> pd.DataFrame:
        startdate = datetime.now().strftime("%Y-%m-%d, %H:%M:00")
        date_range = pd.date_range(start=startdate, periods=(60 * 24), freq="min")

        df_future = pd.DataFrame({"datetime": date_range})
        df_future = df_future.set_index("datetime")
        df_future = self.create_features(df_future)

        df_future[f"{column_name}"] = model.predict(df_future)

        return df_future[[f"{column_name}"]]

    def predict(self) -> pd.DataFrame:
        model_temperature = self.load_model(
            f"{self.model_dir}XGBoost_airTemperature.pkl"
        )
        model_humidity = self.load_model(f"{self.model_dir}XGBoost_humidity.pkl")

        df_temperature = self.single_predict(model_temperature, "airTemperature")
        df_humidity = self.single_predict(model_humidity, "humidity")

        df = pd.concat([df_temperature, df_humidity], axis=1)

        return df

    def get_last_active(self) -> int:
        load_dotenv(".env")
        predict_path = Path("./data/Prediction.csv")

        last_iter = 0

        if predict_path.exists():
            df_old = pd.read_csv("./data/Prediction.csv")
            last_index = df_old[df_old == 0].last_valid_index()
            last_index = len(df_old) - last_index

            last_iter = int(os.getenv("LAST_ITER"))

            last_iter -= last_index

        return np.absolute(last_iter)

    def compute(self):
        df = self.predict()
        index = self.get_last_active()
        num_active = 0
        result = []

        env_iter = Path(".env")

        value = np.nan

        for iter in range(len(df)):
            if iter == index and num_active < 5:
                value = 0

                index += 1
                num_active += 1
            elif iter == index and num_active >= 5:
                value = 1

                delay = self.FuzzyLogicNutrientPump(
                    df.iloc[index]["airTemperature"], df.iloc[index]["humidity"]
                )

                os.environ["DELAY"] = str(delay)

                set_key(
                    dotenv_path=env_iter,
                    key_to_set="LAST_ITER",
                    value_to_set=os.environ["DELAY"],
                )

                index += delay
                num_active = 0
            else:
                value = 1

            result.append(value)

        result = np.array(result)

        df["is_pump_not_active"] = result.astype(int)

        filepath = Path("data/Prediction.csv")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.exists():
            df.to_csv(filepath, mode="a", header=False)
        else:
            df.to_csv(filepath)
