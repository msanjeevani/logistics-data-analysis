import random
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

def create_dataset(records=500):
    start_date = pd.Timestamp("2026-01-01")
    rows = []

    zones = ["North", "South", "East", "West", "Central"]
    vehicles = ["Bike", "Van", "Truck"]

    for i in range(records):
        order_date = start_date + pd.Timedelta(
            days=random.randint(0, 180),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        distance = round(random.uniform(1, 100), 2)
        traffic = random.randint(1, 10)
        weight = round(random.uniform(0.5, 30), 2)
        vehicle = random.choice(vehicles)
        zone = random.choice(zones)

        # Simulated delivery time in minutes
        speed = random.uniform(0.25, 0.65)  # km per minute
        delivery_time = (
            distance / speed
            + traffic * random.uniform(1, 3)
            + weight * random.uniform(0.1, 0.5)
        )

        if vehicle == "Truck":
            delivery_time *= 1.10

        fuel_cost = distance * random.uniform(0.08, 0.20)

        rows.append({
            "order_id": f"ORD{i + 1:04d}",
            "order_date": order_date,
            "zone": zone,
            "vehicle_type": vehicle,
            "delivery_time_minutes": round(delivery_time, 2),
            "distance_km": distance,
            "fuel_cost_usd": round(fuel_cost, 2),
            "package_weight_kg": weight,
            "traffic_index": traffic
        })

    df = pd.DataFrame(rows)

    output = Path("delivery_operations_data.csv")
    df.to_csv(output, index=False)

    print(f"Dataset saved: {output.resolve()}")
    print(f"Total records: {len(df)}")
    print(df.head())

if __name__ == "__main__":
    create_dataset()