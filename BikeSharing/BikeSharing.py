import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

sns.set_theme(style='white')

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


## 6. Clean data by converting date columns
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

# 3. EDA Steps & Dashboard
## Question 1: How Does Weather Impact Bike Rentals?
def analyze_weather_impact(df):
    all_df = pd.merge(

    )
    # Distribution of weather-related features
    plt.figure(figsize=(12, 6))
    sns.histplot(df['weathersit'], bins=4, kde=False)
    plt.title('Distribution of Weather Situations')
    plt.xlabel('Weather Situation')
    plt.ylabel('Frequency')
    plt.show()

    # Average bike rentals under different weather conditions
    avg_rentals_weather = df.groupby('weathersit')['cnt'].mean()
    avg_rentals_weather.plot(kind='bar', color='skyblue', title='Average Rentals by Weather Situation')
    plt.xlabel('Weather Situation')
    plt.ylabel('Average Rental Count')
    plt.show()

    # Scatter plots for temperature, humidity, and windspeed
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='temp', y='cnt', hue='season', palette='viridis')
    plt.title('Bike Rentals vs. Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Rental Count')
    plt.legend(title='Season')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='hum', y='cnt', hue='season', palette='coolwarm')
    plt.title('Bike Rentals vs. Humidity')
    plt.xlabel('Humidity')
    plt.ylabel('Rental Count')
    plt.legend(title='Season')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='windspeed', y='cnt', hue='season', palette='magma')
    plt.title('Bike Rentals vs. Windspeed')
    plt.xlabel('Windspeed')
    plt.ylabel('Rental Count')
    plt.legend(title='Season')
    plt.show()

## Question 2: What Are the Peak Hours and Days for Bike Rentals?
def analyze_peak_hours_days(df):
    if 'hr' not in df.columns:
        st.warning("The 'hr' column is not available in the selected data. Please use the hourly dataset.")
        return

    st.subheader("Analysis of Peak Hours and Days")

    # Average bike rentals by hour
    hour_avg = df.groupby('hr')['cnt'].mean()
    st.write("Average bike rentals by hour:")
    st.line_chart(hour_avg)

    # Average bike rentals by weather situation
    avg_rentals_weather = df.groupby('weathersit')['cnt'].mean()
    st.write("Average bike rentals by weather:")
    st.line_chart(avg_rentals_weather)

    # Average rentals by day of the week
    weekday_avg = df.groupby('weekday')['cnt'].mean()
    fig, ax = plt.subplots()
    ax.bar(weekday_avg.index, weekday_avg.values)
    ax.set_xlabel("Day of the Week")
    ax.set_ylabel("Average Rentals")
    ax.set_title("Average Bike Rentals by Day of the Week")
    st.pyplot(fig)

# Question 3: How is performance of bike rentals by seasons (holiday, weekday, working day) compare to casual users vs registered users?
def analyze_rentals_by_season(df):
    st.subheader("Bike Rentals by Season")

    # Check columns
    required_cols = {'holiday', 'weekday', 'workingday', 'casual', 'registered'}
    if not required_cols.issubset(df.columns):
        st.warning("One or more required columns are not available in the selected data.")
        return

    # Group by seasonality
    season_df = df.groupby(['holiday', 'weekday', 'workingday']).agg({
        'casual': 'mean', 'registered': 'mean'
    }).reset_index()

    # Plot line charts for holiday, weekday, and working day
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    sns.lineplot(data=season_df, x='holiday', y='casual', ax=ax[0], label="Casual Users", marker='o')
    sns.lineplot(data=season_df, x='holiday', y='registered', ax=ax[0], label="Registered Users", marker='s')
    ax[0].set_title("Bike Rentals by Holiday")
    ax[0].legend()

    sns.lineplot(data=season_df, x='weekday', y='casual', ax=ax[1], label="Casual Users", marker='o')
    sns.lineplot(data=season_df, x='weekday', y='registered', ax=ax[1], label="Registered Users", marker='s')
    ax[1].set_title("Bike Rentals by Weekday")
    ax[1].legend()

    sns.lineplot(data=season_df, x='workingday', y='casual', ax=ax[2], label="Casual Users", marker='o')
    sns.lineplot(data=season_df, x='workingday', y='registered', ax=ax[2], label="Registered Users", marker='s')
    ax[2].set_title("Bike Rentals by Working Day")
    ax[2].legend()

    st.pyplot(fig)

# Question 4: How is performance of bike rentals by weather situation compare to casual users vs registered users?
def analyze_rentals_by_weather(df):
    st.subheader("Bike Rentals by Weather Situation")

    # Check columns
    if 'weathersit' not in df.columns or 'casual' not in df.columns or 'registered' not in df.columns:
        st.warning("One or more required columns are not available in the selected data.")
        return

    # Group by weather situation
    weather_df = df.groupby('weathersit').agg({
        'casual': 'mean', 'registered': 'mean'
    }).reset_index()

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=weather_df, x='weathersit', y='casual', label="Casual Users", marker='o')
    sns.lineplot(data=weather_df, x='weathersit', y='registered', label="Registered Users", marker='s')
    ax.set_title("Average Bike Rentals by Weather Situation")
    ax.set_xlabel("Weather Situation")
    ax.set_ylabel("Average Rentals")
    ax.legend()

    st.pyplot(fig)

# Compare casual and registered users
def compare_casual_registered(df):
    st.subheader("Comparison of Casual and Registered Users")
    if 'hr' in df.columns:
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=df, x='hr', y='casual', label='Casual Users', marker='o')
        sns.lineplot(data=df, x='hr', y='registered', label='Registered Users', marker='s')
        plt.title('Hourly Bike Rentals - Casual vs. Registered Users')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Average Rental Count')
        plt.legend()
        plt.grid()
        st.pyplot(plt)

# Analyze time-series trends
def analyze_time_series(df):
    st.subheader("Time-Series Analysis of Bike Rentals Over Time")
    plt.figure(figsize=(14, 6))
    df.groupby('dteday')['cnt'].mean().plot(title='Average Daily Bike Rentals Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Rental Count')
    plt.grid()
    st.pyplot(plt)

# Filter by date
def filter_data_by_date(df, start_date, end_date):
    mask = (df['dteday'] >= pd.to_datetime(start_date)) & (df['dteday'] <= pd.to_datetime(end_date))
    return df.loc[mask]

# Filter by hour
def filter_data_by_hour(df, selected_hours):
    if selected_hours:
        mask = df['hr'].isin(selected_hours)
        return df.loc[mask]
    return df

# Display the filtered data
def display_filtered_data(filtered_df, start_date, end_date, selected_hours):
    if filtered_df is not None and not filtered_df.empty:
        st.write(f"Data from {start_date} to {end_date} for hour(s): {selected_hours}")
        st.write(filtered_df)
    else:
        st.write("No data available for the selected filters.")

def date_hour_picker(hour_df):
    min_date = hour_df['dteday'].min()
    max_date = hour_df['dteday'].max()

    # Date range picker for selecting the date range
    with st.sidebar:
        st.image("https://thumbs.dreamstime.com/b/public-city-bicycle-sharing-business-vector-flat-illustration-man-woman-pay-bike-rent-modern-automated-bike-rental-service-169415132.jpg")

        start_date, end_date = st.date_input(
            "Choose Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

        # Multi-select for hour selection
        all_hours = list(range(24))  # Assuming hours are from 0 to 23
        selected_hours = st.multiselect("Choose hour(s)", all_hours, default=all_hours)

    # Validate the selected date range
    if start_date > end_date:
        st.error("Error: End date must fall after start date.")
        return None, None, []

    return start_date, end_date, selected_hours

def perform_eda(filtered_df):
    st.subheader("Exploratory Data Analysis")
    analyze_weekly_trends(filtered_df)
    analyze_casual_registered_users(filtered_df)
    analyze_rentals_over_time(filtered_df)
    analyze_rentals_by_season(filtered_df)  # Added analysis by season
    analyze_rentals_by_weather(filtered_df)  # Added analysis by weather

# Visualize weekly trends
def analyze_weekly_trends(df):
    if 'weekday' not in df.columns:
        st.warning("The 'weekday' column is not available in the selected data.")
        return

    st.subheader("Average Bike Rentals by Day of the Week")
    weekday_avg = df.groupby('weekday')['cnt'].mean()
    st.bar_chart(weekday_avg)

# Visualize casual vs registered users
def analyze_casual_registered_users(df):
    if 'casual' not in df.columns or 'registered' not in df.columns:
        st.warning("The 'casual' or 'registered' column is not available in the selected data.")
        return

    st.subheader("Hourly Bike Rental: Regular User vs Registered User")
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df, x='hr', y='casual', label='Casual User', marker='o')
    sns.lineplot(data=df, x='hr', y='registered', label='Registered User', marker='s')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Rental Count')
    st.pyplot(plt)

# Analyze rentals over time
def analyze_rentals_over_time(df):
    if 'dteday' not in df.columns:
        st.warning("The 'dteday' column is not available in the selected data.")
        return

    st.subheader("Average Daily Bike Rental Over Time")
    daily_rentals = df.groupby('dteday')['cnt'].mean()
    st.line_chart(daily_rentals)

# Run the Streamlit dashboard
def main():
    # Load data
    hour_df = pd.read_csv("hour.csv", parse_dates=['dteday'])

    # Display date range picker and hour selection in the main content
    start_date, end_date, selected_hours = date_hour_picker(hour_df)

    # Filter data by selected date range and hours
    if start_date and end_date:
        filtered_df = filter_data_by_date(hour_df, start_date, end_date)
        perform_eda(filtered_df)

if __name__ == "__main__":
    main()