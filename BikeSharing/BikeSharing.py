import pandas as pd
import os


# Column & DataFrame settings
def set_display_options():
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)  # Prevent DataFrame from breaking into multiple lines
    pd.set_option('display.width', 1000)  # Set the width large enough for DataFrame

# A. Assessing Data
# 1. Read path and load data
def load_data(path, hour_file="hour.csv", day_file="day.csv"):
    os.chdir(path)
    return pd.read_csv(hour_file), pd.read_csv(day_file)


# 2. Display DataFrame information
def show_info(df, name):
    print(f"----- Assessing Data: {name} -----")
    print(df.info())
    print()


# 3. Check for missing values
def check_missing_values(df, name):
    print(f"{name} missing values:\n", df.isna().sum())
    print()


# 4. Check for duplicated data
def check_duplicated_data(df, name):
    print(f"Sum of duplicated data in {name}: ", df.duplicated().sum())
    print()


# 5. Show statistical summary
def show_statistics(df, name):
    print(f"----- Statistical Summary for {name} -----")
    print(df.describe())
    print("----- End -----")
    print()


# # 6. Clean data by converting date columns
def clean_data(df):
    # Convert date columns from object to datetime
    if 'dteday' in df.columns:
        df['dteday'] = pd.to_datetime(df['dteday'])

    # Convert ordinal columns to category with ordering
    ordinal_cols = {
        'season': [1, 2, 3, 4],  # 1: spring, 2: summer, 3: fall, 4: winter
        'yr': [0, 1],  # 0: 2011, 1: 2012
        'mnth': list(range(1, 13)),  # 1 to 12
        'hr': list(range(24)),  # 0 to 23
        'weathersit': [1, 2, 3, 4]  # From clear to heavy rain
    }
    for col, categories in ordinal_cols.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=categories, ordered=True)

    # Convert nominal columns to category without ordering
    nominal_cols = ['holiday', 'weekday', 'workingday']
    for col in nominal_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Normalize the numerical columns back if needed
    if 'temp' in df.columns:
        df['temp'] = df['temp'] * (39 + 8) - 8  # Convert back to original temperature
    if 'atemp' in df.columns:
        df['atemp'] = df['atemp'] * (50 + 16) - 16  # Convert back to original feeling temperature
    if 'hum' in df.columns:
        df['hum'] = df['hum'] * 100  # Convert back to percentage
    if 'windspeed' in df.columns:
        df['windspeed'] = df['windspeed'] * 67  # Convert back to original scale

    print("Data cleaned successfully.")
    print(df.info(), "\n")
    return df

# 7. Show min and max of each column
def show_min_max(df, name):
    print(f"----- Min and Max Values for {name} DataFrame -----")
    print("Minimum values:\n", df.min(numeric_only=True))
    print("\nMaximum values:\n", df.max(numeric_only=True))
    print("----- End -----")
    print()


# Example usage:
if __name__ == "__main__":
    set_display_options()

    # Specify the path to your data

    # Load the data
    hour_df, day_df = load_data(data_path)

    # Assess the data
    show_info(hour_df, "Hour")
    check_missing_values(hour_df, "Hour")
    check_duplicated_data(hour_df, "Hour")
    show_statistics(hour_df, "Hour")

    show_info(day_df, "Day")
    check_missing_values(day_df, "Day")
    check_duplicated_data(day_df, "Day")
    show_statistics(day_df, "Day")

    # Clean the data
    datetime_columns = ["dteday"]
    print("----- Cleaning Data: Hour -----")
    clean_data(hour_df)

    print("----- Cleaning Data: Day -----")
    clean_data(day_df)

    # Show min and max values
    show_min_max(hour_df, "Hour")
    show_min_max(day_df, "Day")

    # Check for rows where casual + registered does not equal cnt
    discrepancies = hour_df[(hour_df['casual'] + hour_df['registered']) != hour_df['cnt']]
    print("Number of discrepancies:", len(discrepancies))
    print(discrepancies)
    print()

    # Show latest cleaned data
    print(hour_df.head())
    print()
    print(day_df.head())
    print()