import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide")

# Load merged data
@st.cache_data
def load_data():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    hour_path = os.path.join(base_path, "hour.csv")
    day_path = os.path.join(base_path, "day.csv")

    hour_df = pd.read_csv(hour_path)
    day_df = pd.read_csv(day_path)

    merged_df = pd.merge(hour_df, day_df, how="left", on="dteday", suffixes=("_hour", "_day"))
    merged_df["dteday"] = pd.to_datetime(merged_df["dteday"])
    merged_df["weekday"] = merged_df["dteday"].dt.dayofweek

    # Binning based on cnt_hour
    bin_edges = [0, 100, 200, 300, 500, float("inf")]
    bin_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    merged_df['rental_level'] = pd.cut(merged_df['cnt_hour'], bins=bin_edges, labels=bin_labels)

    # Manual grouping based on user domination
    merged_df['user_group'] = merged_df.apply(
        lambda row: 'Dominated by Casual' if row['casual_hour'] / row['cnt_hour'] > 0.7 else (
            'Dominated by Registered' if row['registered_hour'] / row['cnt_hour'] > 0.7 else 'Balanced'
        ),
        axis=1
    )

    return merged_df

df = load_data()

# Sidebar
st.sidebar.image("https://thumbs.dreamstime.com/b/public-city-bicycle-sharing-business-vector-flat-illustration-man-woman-pay-bike-rent-modern-automated-bike-rental-service-169415132.jpg")
st.sidebar.markdown("### Choose Date Range")
start_date, end_date = st.sidebar.date_input("Date range", [df["dteday"].min(), df["dteday"].max()])

st.sidebar.markdown("### Choose hour(s)")
selected_hours = st.sidebar.multiselect(
    "Choose hour(s)", options=list(range(24)), default=list(range(24)), key="hour_selector"
)

# Filter data
df_filtered = df[
    (df["dteday"] >= pd.to_datetime(start_date)) &
    (df["dteday"] <= pd.to_datetime(end_date)) &
    (df["hr"].isin(selected_hours))
]

# Title
st.title("Bike Sharing")

# Plot 1 - Average Rentals by Day of Week with highlight
avg_by_day = df_filtered.groupby("weekday")["cnt_hour"].mean().reset_index()
max_val = avg_by_day["cnt_hour"].max()
min_val = avg_by_day["cnt_hour"].min()

def get_color(val):
    if val == max_val:
        return "blue"
    elif val == min_val:
        return "blue"
    else:
        return "rgba(160, 180, 210, 0.3)"

avg_by_day["color"] = avg_by_day["cnt_hour"].apply(get_color)

fig1 = px.bar(
    avg_by_day,
    x="weekday",
    y="cnt_hour",
    color="color",
    color_discrete_map="identity",  # biar warna pakai kolom 'color' langsung
    labels={'weekday': 'Day of the Week (0=Mon, 6=Sun)', 'cnt_hour': 'Average Rental Count'},
    title="Average Bike Rentals by Day of the Week"
)

st.plotly_chart(fig1)

# Plot 2 – Distribution of Hours on Highest Rents Day
top_day = df_filtered.groupby("dteday")["cnt_hour"].sum().idxmax()
top_day_data = df_filtered[df_filtered["dteday"] == top_day].sort_values("hr")

fig2 = px.bar(
    top_day_data,
    x="hr",
    y="cnt_hour",
    labels={"hr": "Hour", "cnt_hour": "Number of Rents"},
    title=f"Hourly Rents Distribution on {top_day.date()} (Highest Day)"
)
st.plotly_chart(fig2)

# Plot 3 - Casual vs Registered
avg_user = df_filtered.groupby("hr")[["casual_hour", "registered_hour"]].mean().reset_index()
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=avg_user["hr"], y=avg_user["casual_hour"],
    mode="lines+markers", name="Casual User",
    hovertemplate="Hour: %{x}<br>Casual: %{y}<extra></extra>"
))
fig3.add_trace(go.Scatter(
    x=avg_user["hr"], y=avg_user["registered_hour"],
    mode="lines+markers", name="Registered User",
    hovertemplate="Hour: %{x}<br>Registered: %{y}<extra></extra>"
))
fig3.update_layout(
    title="Hourly Bike Rental: Regular User vs Registered User",
    xaxis_title="Hour of the Day",
    yaxis_title="Average Rental Count"
)

st.plotly_chart(fig3)

# Plot 4 - Weekday vs Holiday hourly pattern
avg_cnt_by_hour = df_filtered.groupby(["hr", "holiday_day"])[
    "cnt_hour"
].mean().reset_index()

# Labeling for legend
avg_cnt_by_hour["Holiday Label"] = avg_cnt_by_hour["holiday_day"].map({0: "Weekday", 1: "National Holiday"})

fig4 = px.line(
    avg_cnt_by_hour,
    x="hr",
    y="cnt_hour",
    color="Holiday Label",
    labels={"hr": "Hour", "cnt_hour": "Average Number of Rents"},
    title="Hourly Rents Distribution: Weekdays vs Holidays"
)
fig4.update_traces(mode="lines+markers")
st.plotly_chart(fig4)

# Plot 5 – Casual vs Registered Pattern on High Day
fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=top_day_data["hr"], y=top_day_data["casual_hour"],
    mode="lines+markers", name="Casual",
    marker=dict(symbol="circle"),
    hovertemplate="Hour: %{x}<br>Casual: %{y}<extra></extra>"
))
fig5.add_trace(go.Scatter(
    x=top_day_data["hr"], y=top_day_data["registered_hour"],
    mode="lines+markers", name="Registered",
    marker=dict(symbol="square"),
    hovertemplate="Hour: %{x}<br>Registered: %{y}<extra></extra>"
))
fig5.update_layout(
    title=f"Casual vs Registered Rents Patterns on {top_day.date()}",
    xaxis_title="Hour",
    yaxis_title="Rents Amount"
)
st.plotly_chart(fig5)

# Plot 6 - Average Daily Bike Rental Over Time
avg_daily = df_filtered.groupby("dteday")["cnt_hour"].sum().reset_index()

# Clean up data
avg_daily = avg_daily.dropna(subset=["dteday", "cnt_hour"])
avg_daily["dteday"] = pd.to_datetime(avg_daily["dteday"])

# Plotly line chart
fig6 = px.line(
    avg_daily,
    x="dteday",
    y="cnt_hour",
    labels={"dteday": "Date", "cnt_hour": "Total Daily Rentals"},
    title="Average Daily Bike Rental Over Time",
    markers=True
)
fig6.update_traces(hovertemplate="Date: %{x|%Y-%m-%d}<br>Total: %{y}<extra></extra>")
st.plotly_chart(fig6)

# Plot 7 - Distribution of Rents Level (rental_level)
fig7 = px.histogram(
    df_filtered,
    x="rental_level",
    color="rental_level",
    color_discrete_sequence=px.colors.sequential.Blues,
    title="Distribution of Rental Volume Level (Binning Based on cnt_hour)",
    labels={"rental_level": "Rental Volume Level"}
)
df_filtered["hour_group"] = pd.cut(df_filtered["hr"], bins=[-1, 6, 11, 16, 21, 24], labels=["Night", "Morning", "Noon", "Afternoon", "night Again"])
df_filtered["day_type"] = df_filtered["weekday"].apply(lambda x: "Weekday" if x < 5 else "Weekend")

# Combibne time & user type dominance
df_filtered["user_time_group"] = df_filtered.apply(
    lambda row: f"{row['user_group']} @ {row['hour_group']} ({row['day_type']})", axis=1
)

def temp_level(temp):
    if temp < 0.3:
        return "Cold"
    elif temp < 0.6:
        return "Mild"
    else:
        return "Hot"

df_filtered["temp_level"] = df_filtered["temp_day"].apply(temp_level)

# Combine
df_filtered["weather_rent_seg"] = df_filtered["temp_level"] + " | " + df_filtered["rental_level"].astype(str)

st.plotly_chart(fig7, use_container_width=True)

# Plot 8 - Distribution of User Type Domination
fig8 = px.pie(
    df_filtered,
    names="user_group",
    title="User Group Dominance (Manual Grouping)",
    hole=0.4,
    color_discrete_sequence=px.colors.qualitative.Pastel
)
st.plotly_chart(fig8, use_container_width=True)

# Plot 9 - User-Time Grouping
fig9 = px.histogram(
    df_filtered,
    x="user_time_group",
    color="user_group",
    title="User Type Dominance by Hour Group and Day Type",
    labels={"user_time_group": "Time Segment"},
    category_orders={"user_time_group": sorted(df_filtered["user_time_group"].unique())}
)
fig9.update_xaxes(tickangle=45)
st.plotly_chart(fig9, use_container_width=True)

# Plot 10 - Weather dan Rentals Level
fig10 = px.bar(
    df_filtered.groupby("weather_rent_seg")["cnt_hour"].count().reset_index(),
    x="weather_rent_seg",
    y="cnt_hour",
    title="Rental Distribution by Weather Condition and Rental Volume Level",
    labels={"cnt_hour": "Count", "weather_rent_seg": "Weather + Rental Level"}
)
st.plotly_chart(fig10, use_container_width=True)
