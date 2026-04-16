# Grid Demand Forecasting

We are trying to predict grid demand with a specific focus on spinning reserve management.

## Key Features Demonstrated

- **Scale**: Handles high-dimensionality data using vectorized tensor operations.
- **Modularity**: Clean separation of concerns between data loading (`GridDataset`), architecture (`ResidualGridNet`), and multi-task loss calculation.
- **Domain Specificity**: Tailored for spinning reserves with a Multi-Horizon output head covering 24h, 48h, and 72h prediction windows.
- **Robustness**: Utilizes `RobustScaler` and residual connections to ensure stable model training and prevent divergence during sensor noise or anomalies.

## Setup & Installation

The project uses a standard `pyproject.toml` for dependency management.

```bash
# Install the required dependencies
make install
```

## Running the Model

Trigger the training process with a single command:

```bash
make train
```

*(Note: Ensure your grid data CSV is located in the appropriate path as required by `TFT.py`.)*

## Architecture details

- **GridDataset**: Handles sequence creation and dynamic scaling via `RobustScaler`.
- **ResidualGridNet**: A Bi-LSTM network with Cross-Attention and residual connections modeling temporal dependencies in both directions.
- **Dynamic Loss Weighting**: The `train_step` applies domain-specific importance weighting for each prediction horizon (prioritizing the 24h horizon for immediate spinning reserves).
