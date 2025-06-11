import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide")

# Load merged data
@st.cache_data
def load_data():
    hour_df = pd.read_csv("hour.csv")
    day_df = pd.read_csv("day.csv")
    merged_df = pd.merge(hour_df, day_df, how="left", on="dteday", suffixes=("_hour", "_day"))
    merged_df["dteday"] = pd.to_datetime(merged_df["dteday"])
    merged_df["weekday"] = merged_df["dteday"].dt.dayofweek
    return merged_df

df = load_data()

# Sidebar
st.sidebar.image("https://thumbs.dreamstime.com/b/public-city-bicycle-sharing-business-vector-flat-illustration-man-woman-pay-bike-rent-modern-automated-bike-rental-service-169415132.jpg", use_container_width=True)
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
