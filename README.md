# 🚚 Logistics Operations Analytics & Delivery Time Prediction

A Python-based data analytics and machine learning project designed to explore logistics operations, evaluate delivery performance, identify operational patterns, and estimate delivery times through an interactive dashboard.

## 📌 Project Overview

Efficient logistics management plays an important role in reducing delivery delays, controlling transportation costs, and improving operational performance.

This project analyzes delivery-related data to identify trends and relationships among delivery time, distance, package weight, traffic conditions, and fuel costs. It combines Exploratory Data Analysis (EDA), data visualization, machine learning, and an interactive Streamlit dashboard in a single application.

**Project Type:** Data Analytics & Machine Learning  
**Project Stage:** Week 1 – Strategic Planning and Data Exploration  
**Programming Language:** Python

## 🎯 Objectives

- Clean and validate logistics datasets.
- Analyze delivery performance using Key Performance Indicators (KPIs).
- Explore relationships between delivery time, distance, traffic, and package weight.
- Visualize delivery trends and operational patterns.
- Build a Random Forest model to estimate delivery duration.
- Apply K-Means clustering to identify delivery profiles.
- Present analytical results through an interactive dashboard.

## ✨ Key Features

### 1. Data Preparation
- Dataset validation and missing-value handling.
- Date and numerical data conversion.
- Removal of invalid delivery records.
- Export of cleaned data for further analysis.

### 2. Exploratory Data Analysis
- Descriptive statistics for delivery metrics.
- Correlation heatmap of numerical variables.
- Delivery-time distribution visualization.
- Delivery-zone performance comparison.

### 3. Logistics KPI Analysis
The project calculates:

- Total number of deliveries.
- Average delivery duration.
- Average delivery distance.
- Average fuel cost.
- Calculated average delivery speed.

### 4. Delivery Time Prediction
A Random Forest regression model estimates delivery duration based on:

- Distance travelled.
- Package weight.
- Traffic index.
- Estimated fuel cost.

Model evaluation includes Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.

### 5. Delivery Profile Clustering
K-Means clustering groups delivery records according to distance, delivery duration, package weight, and traffic conditions.

These groups can help explore different delivery profiles and support further operational investigation.

### 6. Interactive Streamlit Dashboard
- Date-range filtering.
- Delivery-zone and vehicle-type filters.
- KPI cards and trend charts.
- Delivery record exploration.
- Delivery-time prediction interface.
- Cluster summaries.
- CSV downloads.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and clustering |
| Streamlit | Interactive dashboard |
| Git & GitHub | Version control and project hosting |

## 📁 Project Structure

```text
logistics-data-analysis/
│
├── app.py
├── analysis.py
├── create_dataset.py
├── delivery_operations_data.csv
├── requirements.txt
├── README.md
├── .gitignore
│
└── outputs/
    ├── kpi_summary.csv
    ├── descriptive_statistics.csv
    ├── correlation_heatmap.png
    ├── delivery_time_distribution.png
    ├── zone_performance.csv
    ├── zone_performance.png
    ├── model_metrics.csv
    ├── predictions.csv
    ├── cluster_summary.csv
    └── cleaned_delivery_data.csv
```

*Note: Output files are generated when the analysis script runs. Some outputs depend on whether the dataset contains optional columns such as `zone`.*

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10 or a compatible Python version.
- Visual Studio Code or another Python editor.
- Git installed on your system.

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/logistics-data-analysis.git
cd logistics-data-analysis
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Generate the sample dataset

```bash
python create_dataset.py
```

This command generates `delivery_operations_data.csv` containing 500 simulated delivery records.

### Step 5: Run the data analysis

```bash
python analysis.py
```

The analysis results and visualizations are saved in the `outputs/` directory.

### Step 6: Launch the interactive dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## 📊 Model Evaluation

The Random Forest regression model is evaluated using:

- **MAE:** Average absolute difference between actual and predicted delivery times.
- **RMSE:** Penalizes larger prediction errors more heavily.
- **R² Score:** Indicates how well the model explains variation in delivery time.

Actual metric values are generated by running the analysis script and saved in `outputs/model_metrics.csv`. Results may vary with different datasets.

## 📂 Dataset Information

The included dataset generator creates synthetic delivery records with the following fields:

| Column | Description |
|---|---|
| `order_id` | Unique delivery identifier |
| `order_date` | Delivery order date and time |
| `zone` | Delivery zone |
| `vehicle_type` | Vehicle category |
| `delivery_time_minutes` | Simulated delivery duration |
| `distance_km` | Delivery distance |
| `fuel_cost_usd` | Simulated fuel cost |
| `package_weight_kg` | Package weight |
| `traffic_index` | Simulated traffic score from 1 to 10 |

**Data Disclaimer:** The generated dataset is synthetic and intended for educational demonstration. It does not represent actual logistics company operations. Model results should not be interpreted as validated real-world performance.

## 🔮 Future Enhancements

- Integrate verified real-world logistics datasets.
- Add route mapping and geographic visualization.
- Compare Random Forest with other regression models.
- Introduce delivery-delay classification.
- Add model persistence and prediction history.
- Deploy the dashboard to a cloud hosting platform.
- Add automated reporting and operational alerts.

## 🎓 Learning Outcomes

This project demonstrates practical experience in:

- Data preprocessing and validation.
- Exploratory Data Analysis.
- KPI development and interpretation.
- Statistical visualization.
- Regression model development and evaluation.
- Unsupervised learning with K-Means.
- Interactive dashboard development.
- GitHub-based project documentation.

## 👩‍💻 Author

**Sanjee Vani M.**  
Final-Year B.E. Computer Science and Engineering Student

GitHub: [@msanjeevani](https://github.com/msanjeevani)

LinkedIn: [Sanjee Vani M.](https://www.linkedin.com/in/sanjee-vani-m-83034740a/)

## 📄 License

This project can be shared for educational and portfolio purposes. Add an appropriate open-source license, such as the MIT License, if you intend to permit reuse and redistribution.

---

*Developed as a learning project focused on logistics data exploration, delivery performance analytics, and machine learning.*
