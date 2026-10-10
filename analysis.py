from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path("delivery_operations_data.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

REQUIRED = [
    "order_date",
    "delivery_time_minutes",
    "distance_km",
    "fuel_cost_usd",
    "package_weight_kg",
    "traffic_index"
]

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Dataset missing. Run: python create_dataset.py"
        )

    df = pd.read_csv(DATA_FILE)

    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["order_date"] = pd.to_datetime(
        df["order_date"], errors="coerce"
    )

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
        subset=[
            "order_date",
            "delivery_time_minutes",
            "distance_km"
        ]
    )

    df = df[
        (df["delivery_time_minutes"] > 0)
        & (df["distance_km"] > 0)
    ].copy()

    if df.empty:
        raise ValueError("No valid records remain after cleaning.")

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    df["avg_speed_kmh"] = (
        df["distance_km"] / (df["delivery_time_minutes"] / 60)
    )

    df["delivery_day"] = df["order_date"].dt.day_name()
    df["delivery_month"] = df["order_date"].dt.to_period("M").astype(str)

    # KPI summary
    kpis = pd.DataFrame({
        "Metric": [
            "Total deliveries",
            "Average delivery time (minutes)",
            "Average distance (km)",
            "Average fuel cost (USD)",
            "Average calculated speed (km/h)"
        ],
        "Value": [
            len(df),
            round(df["delivery_time_minutes"].mean(), 2),
            round(df["distance_km"].mean(), 2),
            round(df["fuel_cost_usd"].mean(), 2),
            round(df["avg_speed_kmh"].mean(), 2)
        ]
    })

    print("\n=== Logistics KPI Summary ===")
    print(kpis.to_string(index=False))
    kpis.to_csv(OUTPUT_DIR / "kpi_summary.csv", index=False)

    print("\n=== Descriptive Statistics ===")
    print(df[numeric_cols].describe().round(2))
    df[numeric_cols].describe().round(2).to_csv(
        OUTPUT_DIR / "descriptive_statistics.csv"
    )

    # Correlation heatmap
    features = [
        "delivery_time_minutes",
        "distance_km",
        "package_weight_kg",
        "traffic_index",
        "fuel_cost_usd"
    ]

    plt.figure(figsize=(9, 6))
    sns.heatmap(
        df[features].corr(),
        annot=True,
        cmap="Blues",
        fmt=".2f",
        vmin=-1,
        vmax=1
    )
    plt.title("Correlation Matrix of Delivery Factors")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # Delivery time distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["delivery_time_minutes"], kde=True)
    plt.title("Delivery Time Distribution")
    plt.xlabel("Delivery time (minutes)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "delivery_time_distribution.png", dpi=150)
    plt.close()

    # Average delivery time by zone, if available
    if "zone" in df.columns:
        zone_stats = (
            df.groupby("zone")["delivery_time_minutes"]
            .agg(["count", "mean", "median"])
            .round(2)
        )
        zone_stats.to_csv(OUTPUT_DIR / "zone_performance.csv")

        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=df,
            x="zone",
            y="delivery_time_minutes",
            errorbar=None
        )
        plt.title("Average Delivery Time by Zone")
        plt.ylabel("Minutes")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "zone_performance.png", dpi=150)
        plt.close()

    # Machine learning: predict delivery time
    model_features = [
        "distance_km",
        "package_weight_kg",
        "traffic_index",
        "fuel_cost_usd"
    ]

    X = df[model_features]
    y = df["delivery_time_minutes"]

    if len(df) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=150,
                random_state=42
            )
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rmse = mean_squared_error(y_test, predictions) ** 0.5
        metrics = {
            "MAE_minutes": round(mean_absolute_error(y_test, predictions), 2),
            "RMSE_minutes": round(rmse, 2),
            "R2_score": round(r2_score(y_test, predictions), 3)
        }

        print("\n=== Prediction Model Evaluation ===")
        print(metrics)
        pd.DataFrame([metrics]).to_csv(
            OUTPUT_DIR / "model_metrics.csv", index=False
        )

        pd.DataFrame({
            "actual_delivery_time": y_test,
            "predicted_delivery_time": predictions
        }).to_csv(OUTPUT_DIR / "predictions.csv", index=False)

        # Fit on all available records for later dashboard use
        model.fit(X, y)
        import joblib
        joblib.dump(model, OUTPUT_DIR / "delivery_time_model.joblib")

    # K-Means clustering for delivery profiles
    cluster_features = [
        "distance_km",
        "delivery_time_minutes",
        "package_weight_kg",
        "traffic_index"
    ]

    if len(df) >= 3:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[cluster_features])

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["delivery_cluster"] = kmeans.fit_predict(scaled)

        cluster_summary = (
            df.groupby("delivery_cluster")[cluster_features]
            .mean()
            .round(2)
        )
        cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv")

    df.to_csv(OUTPUT_DIR / "cleaned_delivery_data.csv", index=False)

    print("\nAnalysis completed successfully.")
    print(f"Output files saved in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()