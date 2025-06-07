# Hull-White Interest Rate Model Calibration

This project implements the calibration of the Hull-White interest rate model using US Treasury market data. The implementation focuses on fitting the model to the current term structure of interest rates and estimating the model's volatility parameters.

## Project Overview

The Hull-White model is a one-factor interest rate model that extends the Vasicek model by allowing the mean reversion level to be time-dependent. This implementation:

1. Fetches historical 3-month Treasury rates to calibrate the volatility parameter
2. Processes current Treasury bills and bonds data
3. Bootstraps the zero-coupon yield curve
4. Calibrates the Hull-White model parameters to match market prices
5. Visualizes the fitted term structure against market rates

## Requirements

- Required Python packages:
  - numpy
  - pandas
  - pandas_datareader
  - matplotlib
  - scipy

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install numpy pandas pandas_datareader matplotlib scipy
```

## Data Files

The project uses the following data files:
- `bills2024-03-05.txt`: Treasury bills data
- `bonds2024-03-05.txt`: Treasury bonds data
- `hull_white_params.txt`: Calibrated model parameters
- `hull_white_zeta.txt`: Calibrated zeta function values

## Usage

Run the main script:
```bash
python "Hull-White-Interest-Rate-Model-Calibration.py"
```

The script will:
1. Fetch historical Treasury rates
2. Calibrate the Hull-White volatility parameter
3. Process current Treasury data
4. Bootstrap the zero-coupon yield curve
5. Calibrate the model to match market prices
6. Generate a plot comparing market rates with model rates
7. Save the calibrated parameters to files

## Output

The script generates:
1. A plot showing the fitted term structure against market rates
2. `hull_white_params.txt` containing the calibrated parameters (sigma and a)
3. `hull_white_zeta.txt` containing the calibrated zeta function values

## Model Details

The Hull-White model is implemented with the following components:
- Short rate dynamics: dr(t) = (θ(t) - a*r(t))dt + σdW(t)
- Bond price formula: P(t,T) = A(t,T)exp(-B(t,T)r(t))
- Calibration to market prices using numerical optimization

## Notes

- The implementation uses a polynomial fit to smooth the bootstrapped discount factors
- The calibration process matches both the current term structure and historical volatility
- The model parameters are calibrated to minimize the difference between model and market prices



