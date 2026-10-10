# Logistics Operations Analytics

## Project Overview

This project explores delivery operations using Python, data analysis,
visualization, machine learning and an interactive Streamlit dashboard.

## Features

- Sample dataset generation
- Data cleaning and validation
- Logistics KPI analysis
- Correlation heatmap
- Delivery time prediction using Random Forest
- Delivery profile clustering using K-Means
- Interactive filters and CSV downloads

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit

## Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the sample dataset:

```bash
python create_dataset.py
```

Run batch analysis:

```bash
python analysis.py
```

Launch the interactive dashboard:

```bash
streamlit run app.py
```

## Dataset

The included generator creates synthetic records for educational
demonstration. The data does not represent an actual logistics company.

## Limitations

Prediction quality depends on data quality and sample representativeness.
The synthetic data is suitable for demonstrating the workflow, not for
real-world operational decisions.

## Future Enhancements

- Use verified real-world delivery records
- Add route and delivery-zone maps
- Compare additional machine-learning models
- Add shipment status and delay classification
- Deploy the dashboard online
