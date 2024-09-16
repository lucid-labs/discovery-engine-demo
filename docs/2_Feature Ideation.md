


# Feature Ideation Table Template

| **Feature Name** | **Description** | **Data Source** | **Data Type** | **Unit/Scale** | **Transformation/Preprocessing Required** | **Validation Criteria** | **Business Justification** |
|------------------|-----------------|-----------------|---------------|----------------|--------------------------------------------|-------------------------|----------------------------|
| Feature 1        | Brief description of the feature. | API, Dataset, or Source of Data (e.g., Uniswap API) | Categorical, Numerical, Text, etc. | Define the units or scale (e.g., USD, log scale) | Define preprocessing steps like normalization, log transformation, etc. | How to validate this feature (e.g., Check for missing values, data distribution). | Why this feature is useful for solving the problem or achieving the business objective. |
| Feature 2        | Brief description of the feature. | Data Source (e.g., Ethereum mainnet) | Numerical | E.g., Percentage or a defined range | E.g., Min-max normalization, one-hot encoding for categorical | Validate against benchmark data or manually spot check | Reason the feature is crucial to problem-solving (e.g., improves accuracy in predictions). |
| Feature 3        | Brief description of the feature. | Data Source (e.g., CoinGecko API) | Time-series | E.g., USD | Smoothing, windowing, or other techniques. | Ensure data is non-missing, stationarity check for time series | The expected business impact (e.g., helps estimate market trends accurately). |

## Example

| **Feature Name**  | **Description**                      | **Data Source**       | **Data Type** | **Unit/Scale** | **Transformation/Preprocessing Required**  | **Validation Criteria**      | **Business Justification**                  |
|-------------------|--------------------------------------|-----------------------|---------------|----------------|--------------------------------------------|--------------------------------|--------------------------------------------|
| Trading Volume     | 24h Trading volume of a token pair   | CoinGecko API          | Numerical     | USD            | Log transformation                         | Validate against historical data| Important for understanding market liquidity|
| Liquidity Pool Size| Total value locked in liquidity pools| Uniswap API            | Time-series   | USD            | Smoothing, rolling window                  | Ensure no missing values and steady flow | Helps predict liquidity fluctuations in DeFi |
| Price Volatility   | Price fluctuations over time         | Ethereum Blockchain    | Numerical     | Percentage      | Normalization                              | Validate variance over time, remove outliers | Predictive of market risk in lending protocols|
