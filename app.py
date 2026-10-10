from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path("delivery_operations_data.csv")

st.set_page_config(
    page_title="Logistics Analytics Dashboard",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Logistics Operations Analytics")
st.caption("Exploratory analysis, performance KPIs, prediction and delivery segmentation")

if not DATA_FILE.exists():
    st.error("Dataset not found. Run `python create_dataset.py` first.")
    st.stop()

df = pd.read_csv(DATA_FILE)
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

numeric_cols = [
    "delivery_time_minutes",
    "distance_km",
    "fuel_cost_usd",
    "package_weight_kg",
    "traffic_index"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["order_date", "delivery_time_minutes", "distance_km"]
)
df = df[
    (df["delivery_time_minutes"] > 0)
    & (df["distance_km"] > 0)
].copy()

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

df["avg_speed_kmh"] = (
    df["distance_km"] / (df["delivery_time_minutes"] / 60)
)

# Sidebar filters
st.sidebar.header("Filters")

min_date = df["order_date"].min().date()
max_date = df["order_date"].max().date()

date_range = st.sidebar.date_input(
    "Order date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

filtered = df.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        filtered["order_date"].dt.date.between(start, end)
    ]

if "zone" in filtered.columns:
    zones = sorted(filtered["zone"].dropna().unique().tolist())
    selected_zones = st.sidebar.multiselect(
        "Delivery zones", zones, default=zones
    )
    filtered = filtered[filtered["zone"].isin(selected_zones)]

if "vehicle_type" in filtered.columns:
    vehicles = sorted(filtered["vehicle_type"].dropna().unique().tolist())
    selected_vehicles = st.sidebar.multiselect(
        "Vehicle type", vehicles, default=vehicles
    )
    filtered = filtered[filtered["vehicle_type"].isin(selected_vehicles)]

if filtered.empty:
    st.warning("No records match these filters. Change the filters.")
    st.stop()

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Deliveries", f"{len(filtered):,}")
c2.metric("Avg Delivery Time", f"{filtered['delivery_time_minutes'].mean():.1f} min")
c3.metric("Avg Distance", f"{filtered['distance_km'].mean():.1f} km")
c4.metric("Avg Fuel Cost", f"${filtered['fuel_cost_usd'].mean():.2f}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Data Explorer", "Prediction", "Clustering"
])

with tab1:
    st.subheader("Delivery time trend")
    trend = (
        filtered.assign(day=filtered["order_date"].dt.date)
        .groupby("day")["delivery_time_minutes"]
        .mean()
    )
    st.line_chart(trend)

    if "zone" in filtered.columns:
        st.subheader("Average delivery time by zone")
        zone_avg = (
            filtered.groupby("zone")["delivery_time_minutes"]
            .mean().sort_values(ascending=False)
        )
        st.bar_chart(zone_avg)

    st.subheader("Correlation matrix")
    st.dataframe(
        filtered[numeric_cols].corr().round(2),
        use_container_width=True
    )

with tab2:
    st.subheader("Filtered delivery records")
    st.dataframe(filtered, use_container_width=True)

    st.download_button(
        "Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_delivery_data.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("Estimate delivery time")

    distance = st.number_input(
        "Distance (km)", min_value=0.1, max_value=1000.0, value=20.0
    )
    weight = st.number_input(
        "Package weight (kg)", min_value=0.1, max_value=500.0, value=5.0
    )
    traffic = st.slider("Traffic index", min_value=1, max_value=10, value=5)
    fuel = st.number_input(
        "Estimated fuel cost (USD)",
        min_value=0.01, max_value=10000.0, value=3.0
    )

    model_features = [
        "distance_km",
        "package_weight_kg",
        "traffic_index",
        "fuel_cost_usd"
    ]

    if st.button("Predict delivery time"):
        if len(df) < 10:
            st.error("At least 10 valid records are required.")
        else:
            X = df[model_features]
            y = df["delivery_time_minutes"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = make_pipeline(
                StandardScaler(),
                RandomForestRegressor(n_estimators=150, random_state=42)
            )
            model.fit(X_train, y_train)

            input_data = pd.DataFrame([{
                "distance_km": distance,
                "package_weight_kg": weight,
                "traffic_index": traffic,
                "fuel_cost_usd": fuel
            }])

            prediction = model.predict(input_data)[0]
            st.success(f"Estimated delivery time: {prediction:.1f} minutes")
            st.caption(
                "This estimate is based on the available dataset and is not a guaranteed delivery time."
            )

with tab4:
    st.subheader("Delivery profiles using K-Means")

    cluster_features = [
        "distance_km",
        "delivery_time_minutes",
        "package_weight_kg",
        "traffic_index"
    ]

    if len(filtered) < 3:
        st.info("Select at least three records to perform clustering.")
    else:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        scaled = StandardScaler().fit_transform(
            filtered[cluster_features]
        )
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clustered = filtered.copy()
        clustered["cluster"] = kmeans.fit_predict(scaled)

        st.write("Cluster means")
        st.dataframe(
            clustered.groupby("cluster")[cluster_features]
            .mean().round(2),
            use_container_width=True
        )

        st.bar_chart(
            clustered.groupby("cluster")["delivery_time_minutes"].mean()
        )

        st.download_button(
            "Download clustered data",
            data=clustered.to_csv(index=False).encode("utf-8"),
            file_name="clustered_delivery_data.csv",
            mime="text/csv"
        )

st.divider()
st.caption(
    "Academic demonstration using synthetic sample data. "
    "Model performance should be validated on real delivery records before operational use."
)