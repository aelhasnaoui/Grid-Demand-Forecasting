# Atalaia Grid Demand Forecasting Framework (Live Edition)

This framework provides a sophisticated solution for predicting grid demand using **real-time data**, specifically optimized for manage spinning reserves.

## Key Features Demonstrated

- **Real-Time Integration**: Leverages `gridstatus` to fetch live electricity load data directly from major ISOs (CAISO, PJM, ERCOT), eliminating the need for static historical CSVs.
- **Bi-Directional Attention Architecture**: Implements a `ResidualGridNet` (Bi-LSTM with Attention mechanism) that captures complex temporal dependencies in both forward and backward directions.
- **Multi-Horizon Forecasting**: Includes dedicated output heads for multiple forecasting horizons (24h, 48h, and 72h), essential for grid stability mapping.
- **Robust Training Pipeline**: Uses `RobustScaler` and residual learning to ensure model convergence and resilience against live sensor noise.

## Setup & Installation

The project uses `pyproject.toml` for modern dependency management.

```bash
# Install the required dependencies
make install
```

## Running the Forecast

The core logic is now encapsulated in a high-interactivity Jupyter Notebook:

1. Open **`GridForecast.ipynb`**.
2. Run the cells to fetch the latest 60 days of live grid data.
3. The model will automatically initialize and train on the current grid conditions.

## Architecture details

- **RealTimeGridDataset**: Dynamically connects to ISO APIs, handles hourly resampling, and performs robust scaling on live streams.
- **ResidualGridNet**: A deep learning architecture combining Bi-LSTM layers with Cross-Attention to focalize on critical historical load peaks.
- **Domain-Specific Loss**: Applies a weighted MSE loss that prioritizes 24h accuracy (critical for Spinning Reserves) while maintaining long-range stability for 72h projections.
