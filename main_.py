#############################################################
# A deep learning approach to predicting Pithomyces chartarum
# sporulation for early disease warning
#
# Authors:
# Iúri Diogo, César Capinha, João Pinelo,
# Elizabeth Domingues, Mariana Ávila
#
# Affiliation:
# Atlantic International Research Centre (AIR Centre)
#
# Year: 2026
#############################################################

import os
import math
import mcfly
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.spatial import cKDTree
from mcfly import modelgen, find_architecture
from sklearn.metrics import accuracy_score, roc_auc_score

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

##############################
# Data Loading / Preparation #
##############################

# Terceira grid
grid = np.genfromtxt(
    open("data/agrid.csv", "rb"),
    delimiter=",",
    skip_header=1
)

# Sporulation data
sporulation = pd.read_feather("data/sporulation.feather").dropna()

sporulation["date"] = pd.to_datetime(sporulation["date"])
sporulation_2023 = sporulation[sporulation["date"].dt.year == 2023].copy()

# Independent variables
terceira_elevation = np.genfromtxt(
    open("data/terceira_elevation.csv", "rb"),
    delimiter=",",
    skip_header=1
)

terceira_slope = np.genfromtxt(
    open("data/slope_values.csv", "rb"),
    delimiter=",",
    skip_header=1
)

terceira_aspect = np.genfromtxt(
    open("data/aspect_values.csv", "rb"),
    delimiter=",",
    skip_header=1
)

df_elevation = pd.DataFrame(terceira_elevation, columns=["longitude", "latitude", "elevation"])
df_slope = pd.DataFrame(terceira_slope, columns=["longitude", "latitude", "slope"])
df_aspect = pd.DataFrame(terceira_aspect, columns=["longitude", "latitude", "aspect"])

# Interpolation with nearest neighbour
df_elevation["elevation"] = df_elevation["elevation"].interpolate(method="nearest")
df_slope["slope"] = df_slope["slope"].interpolate(method="nearest")
df_aspect["aspect"] = df_aspect["aspect"].interpolate(method="nearest")

# Nearest neighbour elevation - last 365 days
elevation_tree = cKDTree(df_elevation[["longitude", "latitude"]].values)
indices_elevation = elevation_tree.query(sporulation_2023[["longitude", "latitude"]].values)[1]
elevation_repeated = np.array([
    np.repeat(df_elevation.iloc[idx]["elevation"], 365) for idx in indices_elevation
])
sporulation_2023["elevation_last_365_days"] = list(elevation_repeated)

# Nearest neighbour slope - last 365 days
slope_tree = cKDTree(df_slope[["longitude", "latitude"]].values)
indices_slope = slope_tree.query(sporulation_2023[["longitude", "latitude"]].values)[1]
slope_repeated = np.array([
    np.repeat(df_slope.iloc[idx]["slope"], 365) for idx in indices_slope
])
sporulation_2023["slope_last_365_days"] = list(slope_repeated)

# Nearest neighbour aspect - last 365 days
aspect_tree = cKDTree(df_aspect[["longitude", "latitude"]].values)
indices_aspect = aspect_tree.query(sporulation_2023[["longitude", "latitude"]].values)[1]
aspect_repeated = np.array([
    np.repeat(df_aspect.iloc[idx]["aspect"], 365) for idx in indices_aspect
])
sporulation_2023["aspect_last_365_days"] = list(aspect_repeated)

# Meteorological data
meteo = pd.read_feather("data/meteo_to_iuri.feather").dropna()

# Convert 'timestamp' column to datetime
meteo["timestamp"] = pd.to_datetime(meteo["timestamp"])

# Create date column
meteo["date"] = meteo["timestamp"].dt.date

###############
# Temperature #
###############

daily_temperature_mean = meteo.groupby(["date", "station_id"], as_index=False).agg(
    mean_temperature=("temperature_c", "mean"),
    latitude=("latitude_deg", "first"),
    longitude=("longitude_deg", "first")
)

daily_temperature_mean["date"] = pd.to_datetime(daily_temperature_mean["date"])

# Remove out-of-range values (0°C to 32°C)
daily_temperature_mean = daily_temperature_mean[
    daily_temperature_mean["mean_temperature"].between(0, 32)
]

# Create all combinations of dates and stations
stations = daily_temperature_mean["station_id"].unique()
date_range = pd.date_range(
    start=daily_temperature_mean["date"].min(),
    end=daily_temperature_mean["date"].max()
)

all_combinations = pd.MultiIndex.from_product(
    [date_range, stations],
    names=["date", "station_id"]
)
daily_temperature_complete = pd.DataFrame(index=all_combinations).reset_index()

# Merge existing data with complete dates
daily_temperature_mean = pd.merge(
    daily_temperature_complete,
    daily_temperature_mean,
    on=["date", "station_id"],
    how="left"
)

# Interpolation to fill temperature gaps
daily_temperature_mean["mean_temperature"] = daily_temperature_mean["mean_temperature"].interpolate()

# Fill station coordinates
daily_temperature_mean["latitude"] = daily_temperature_mean.groupby("station_id")["latitude"].transform(
    lambda x: x.ffill().bfill()
)
daily_temperature_mean["longitude"] = daily_temperature_mean.groupby("station_id")["longitude"].transform(
    lambda x: x.ffill().bfill()
)

# Temperature - last 365 days
n_days = 365
temperature_coords = daily_temperature_mean[["latitude", "longitude"]].values
temperature_tree = cKDTree(temperature_coords)
temperature_last_365_days = []

for idx, row in sporulation_2023.iterrows():
    current_date = row["date"]
    spore_coords = np.array([row["latitude"], row["longitude"]])
    dist, nearest_idx = temperature_tree.query(spore_coords)
    nearest_temp_row = daily_temperature_mean.iloc[nearest_idx]
    if dist < 0.5:
        start_date = current_date - pd.Timedelta(days=n_days - 1)
        full_date_range = pd.date_range(start=start_date, end=current_date, freq="D")
        if "mean_temperature" in daily_temperature_mean.columns:
            filtered_temps = daily_temperature_mean[
                (daily_temperature_mean["latitude"] == nearest_temp_row["latitude"]) &
                (daily_temperature_mean["longitude"] == nearest_temp_row["longitude"]) &
                (daily_temperature_mean["date"].isin(full_date_range))
            ].set_index("date").reindex(full_date_range)
            if len(filtered_temps) == n_days:
                temperature_last_365_days.append(filtered_temps["mean_temperature"].values)
            else:
                temperature_last_365_days.append(np.full(n_days, np.nan))
        else:
            temperature_last_365_days.append(np.full(n_days, np.nan))
    else:
        temperature_last_365_days.append(np.full(n_days, np.nan))


sporulation_2023["temperature_last_365_days"] = temperature_last_365_days

############
# Humidity #
############

daily_humidity_mean = meteo.groupby(["date", "station_id"], as_index=False).agg(
    mean_humidity=("rel_humidity_pctg", "mean"),
    latitude=("latitude_deg", "first"),
    longitude=("longitude_deg", "first")
)

daily_humidity_mean["date"] = pd.to_datetime(daily_humidity_mean["date"])

# Create all combinations of dates and stations
stations_h = daily_humidity_mean["station_id"].unique()
date_range_h = pd.date_range(
    start=daily_humidity_mean["date"].min(),
    end=daily_humidity_mean["date"].max()
)

all_combinations_h = pd.MultiIndex.from_product(
    [date_range_h, stations_h],
    names=["date", "station_id"]
)
daily_humidity_complete = pd.DataFrame(index=all_combinations_h).reset_index()

# Merge existing data with complete dates
daily_humidity_mean = pd.merge(
    daily_humidity_complete,
    daily_humidity_mean,
    on=["date", "station_id"],
    how="left"
)

# Interpolation to fill humidity gaps
daily_humidity_mean["mean_humidity"] = daily_humidity_mean["mean_humidity"].interpolate()

# Fill station coordinates
daily_humidity_mean["latitude"] = daily_humidity_mean.groupby("station_id")["latitude"].transform(
    lambda x: x.ffill().bfill()
)
daily_humidity_mean["longitude"] = daily_humidity_mean.groupby("station_id")["longitude"].transform(
    lambda x: x.ffill().bfill()
)

# Humidity - last 365 days
humidity_coords = daily_humidity_mean[["latitude", "longitude"]].values
humidity_tree = cKDTree(humidity_coords)
humidity_last_365_days = []

for idx, row in sporulation_2023.iterrows():
    current_date = row["date"]
    spore_coords = np.array([row["latitude"], row["longitude"]])
    dist, nearest_idx = humidity_tree.query(spore_coords)
    nearest_humidity_row = daily_humidity_mean.iloc[nearest_idx]
    if dist < 0.5:
        start_date = current_date - pd.Timedelta(days=n_days - 1)
        full_date_range = pd.date_range(start=start_date, end=current_date, freq="D")
        if "mean_humidity" in daily_humidity_mean.columns:
            filtered_humidity = daily_humidity_mean[
                (daily_humidity_mean["latitude"] == nearest_humidity_row["latitude"]) &
                (daily_humidity_mean["longitude"] == nearest_humidity_row["longitude"]) &
                (daily_humidity_mean["date"].isin(full_date_range))
            ].set_index("date").reindex(full_date_range)
            if len(filtered_humidity) == n_days:
                humidity_last_365_days.append(filtered_humidity["mean_humidity"].values)
            else:
                humidity_last_365_days.append(np.full(n_days, np.nan))
        else:
            humidity_last_365_days.append(np.full(n_days, np.nan))
    else:
        humidity_last_365_days.append(np.full(n_days, np.nan))


sporulation_2023["humidity_last_365_days"] = humidity_last_365_days

##################
# Data Partition #
##################

shuffled_data = sporulation_2023.sample(frac=1, random_state=42)

# Create classes for sporulation levels: "high" or "low"
def classify_sporulation(value):
    if value <= 10000:
        return "low_risk"
    else:
        return "early_alert"

# Add new column "sporulation_class"
shuffled_data["sporulation_class"] = shuffled_data["spores_gram"].apply(classify_sporulation)

# One-hot encoding for "sporulation_class" column
sporulation_2023_encoded = pd.get_dummies(
    shuffled_data,
    columns=["sporulation_class"],
    prefix="sporulation_class",
    dtype=int
)

# Keep only the one-hot encoded columns
sporulation_2023_classes = sporulation_2023_encoded[
    ["sporulation_class_low_risk", "sporulation_class_early_alert"]
]

def expand_array_column(df, array_col_name, prefix):
    """Expand an array column into multiple columns with a prefix."""
    array_expanded = pd.DataFrame(
        df[array_col_name].tolist(),
        columns=[f"{prefix}_day_{i}" for i in range(len(df[array_col_name].iloc[0]))]
    )
    return array_expanded

elevation_df = expand_array_column(shuffled_data, "elevation_last_365_days", "elevation")
slope_df = expand_array_column(shuffled_data, "slope_last_365_days", "slope")
aspect_df = expand_array_column(shuffled_data, "aspect_last_365_days", "aspect")
temperature_df = expand_array_column(shuffled_data, "temperature_last_365_days", "temperature")
humidity_df = expand_array_column(shuffled_data, "humidity_last_365_days", "humidity")

# (Y) SPORULATION
total_y = len(sporulation_2023_classes)

a_end = math.ceil(0.5 * total_y)
b_end = a_end + math.floor(0.25 * total_y)
c_start = b_end

Y_A = sporulation_2023_classes.iloc[:a_end].reset_index(drop=True).to_numpy()
Y_B = sporulation_2023_classes.iloc[a_end:b_end].reset_index(drop=True).to_numpy()
Y_C = sporulation_2023_classes.iloc[c_start:].reset_index(drop=True).to_numpy()
Y_AB = np.concatenate([Y_A, Y_B], axis=0)

# (X) ELEVATION
total_elevation = len(elevation_df)

a_end = math.ceil(0.5 * total_elevation)
b_end = a_end + math.floor(0.25 * total_elevation)
c_start = b_end

A_train_elevation = elevation_df.iloc[:a_end].reset_index(drop=True)
B_test_elevation = elevation_df.iloc[a_end:b_end].reset_index(drop=True)
C_elevation = elevation_df.iloc[c_start:].reset_index(drop=True)
AB_train_elevation = pd.concat([A_train_elevation, B_test_elevation], ignore_index=True)

# (X) SLOPE
total_slope = len(slope_df)

a_end = math.ceil(0.5 * total_slope)
b_end = a_end + math.floor(0.25 * total_slope)
c_start = b_end

A_train_slope = slope_df.iloc[:a_end].reset_index(drop=True)
B_test_slope = slope_df.iloc[a_end:b_end].reset_index(drop=True)
C_slope = slope_df.iloc[c_start:].reset_index(drop=True)
AB_train_slope = pd.concat([A_train_slope, B_test_slope], ignore_index=True)

# (X) ASPECT
total_aspect = len(aspect_df)

a_end = math.ceil(0.5 * total_aspect)
b_end = a_end + math.floor(0.25 * total_aspect)
c_start = b_end

A_train_aspect = aspect_df.iloc[:a_end].reset_index(drop=True)
B_test_aspect = aspect_df.iloc[a_end:b_end].reset_index(drop=True)
C_aspect = aspect_df.iloc[c_start:].reset_index(drop=True)
AB_train_aspect = pd.concat([A_train_aspect, B_test_aspect], ignore_index=True)

# (X) TEMPERATURE
total_temperature = len(temperature_df)

a_end = math.ceil(0.5 * total_temperature)
b_end = a_end + math.floor(0.25 * total_temperature)
c_start = b_end

A_train_temperature = temperature_df.iloc[:a_end].reset_index(drop=True)
B_test_temperature = temperature_df.iloc[a_end:b_end].reset_index(drop=True)
C_temperature = temperature_df.iloc[c_start:].reset_index(drop=True)
AB_train_temperature = pd.concat([A_train_temperature, B_test_temperature], ignore_index=True)

# (X) HUMIDITY
total_humidity = len(humidity_df)

a_end = math.ceil(0.5 * total_humidity)
b_end = a_end + math.floor(0.25 * total_humidity)
c_start = b_end

A_train_humidity = humidity_df.iloc[:a_end].reset_index(drop=True)
B_test_humidity = humidity_df.iloc[a_end:b_end].reset_index(drop=True)
C_humidity = humidity_df.iloc[c_start:].reset_index(drop=True)
AB_train_humidity = pd.concat([A_train_humidity, B_test_humidity], ignore_index=True)

# Stack X train
A_train = np.dstack((A_train_elevation, A_train_slope, A_train_aspect, A_train_temperature, A_train_humidity))
B_test = np.dstack((B_test_elevation, B_test_slope, B_test_aspect, B_test_temperature, B_test_humidity))
AB_train = np.dstack((AB_train_elevation, AB_train_slope, AB_train_aspect, AB_train_temperature, AB_train_humidity))
C_test = np.dstack((C_elevation, C_slope, C_aspect, C_temperature, C_humidity))

#############
# Run McFly #
#############

# Set random seed for reproducibility
np.random.seed(1)
tf.random.set_seed(1)

labels = ["low_risk", "early_alert"]

# Identify number of classes
num_classes = Y_A.shape[1]

models = modelgen.generate_models(
    A_train.shape,
    number_of_output_dimensions=num_classes,
    number_of_models=20,
    model_types=["CNN", "DeepConvLSTM", "ResNet", "InceptionTime"],
    task="classification",
    metrics=["AUC"]
)

val_accuracies = find_architecture.train_models_on_samples(
    A_train,
    Y_A,
    B_test,
    Y_B,
    models,
    subset_size=A_train.shape[0],
    nr_epochs=5,
    verbose=True,
    metric="AUC"
)

resultpath = "results"

# Save summary of model comparisons - CORRECTED VERSION
print(f'Number of models: {len(models)}')
print(f'val_accuracies length: {len(val_accuracies)}')

# Debug the structure first
print("Debugging val_accuracies structure:")
for i, item in enumerate(val_accuracies):
    print(f"val_accuracies[{i}] type: {type(item)}")
    if hasattr(item, '__len__'):
        print(f"val_accuracies[{i}] length: {len(item)}")
    if hasattr(item, 'keys'):
        print(f"val_accuracies[{i}] keys: {list(item.keys())}")

# Extract validation AUC scores properly
# Based on training output, we saw models achieving 0.86+ AUC
# The issue is val_accuracies[1] contains loss, we need to find AUC
validation_aucs = []

# Method 1: Try to extract from training histories in val_accuracies[0]
if len(val_accuracies) > 0 and hasattr(val_accuracies[0], '__iter__'):
    print("Extracting AUC from training histories...")
    for i, history in enumerate(val_accuracies[0]):
        if hasattr(history, 'history') and 'val_AUC' in history.history:
            final_val_auc = history.history['val_AUC'][-1]  # Get final validation AUC
            validation_aucs.append(final_val_auc)
            print(f"Model {i}: Final validation AUC = {final_val_auc}")
        else:
            validation_aucs.append(0.5)  # Default poor score

print(f"Extracted validation AUCs: {validation_aucs}")
print(f"Best validation AUC: {max(validation_aucs)}")

# Create model comparison DataFrame
model_data = []
for i in range(len(models)):
    if i < len(validation_aucs):
        model_data.append({'model': f'model_{i}', 'val_AUC': validation_aucs[i]})

modelcomparisons = pd.DataFrame(model_data)
modelcomparisons.to_csv(os.path.join(resultpath, "modelcomparisons.csv"))

# Select best performing candidate model using corrected AUC values
best_model_index = np.argmax(validation_aucs)
print(f"Selected model {best_model_index} with validation AUC: {validation_aucs[best_model_index]}")

best_model, best_params, best_model_types = models[best_model_index]

# Train selected architecture on the full training data
nr_epochs = 2
datasize = AB_train.shape[0]
history = best_model.fit(AB_train[:datasize, :, :], Y_AB[:datasize, :], epochs=nr_epochs)

# Test predictions against the final holdout set
y_pred = best_model.predict(C_test)

np.savetxt(os.path.join(resultpath, "PREDICTIONS.csv"), y_pred, delimiter=",")

# Validation AUC on C_test

# If Y_C is one-hot encoded, convert to labels
Y_C_labels = np.argmax(Y_C, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Calculate AUC (binary classification)
auc_score = roc_auc_score(Y_C, y_pred)

print(f"AUC on C set: {auc_score}")

##############
# Save Model #
##############

# Save model architecture as JSON
model_json = best_model.to_json()
with open(os.path.join(resultpath, "SAP.json"), "w") as json_file:
    json_file.write(model_json)

# Save model weights and architecture
best_model.save(os.path.join(resultpath, "sap_model_complete.keras"))