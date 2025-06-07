

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import datetime
import dateutil
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Calibrate Hull-White volatility parameter 
today = datetime.date(2024, 3, 5)
start_date = today - relativedelta(years=10)
end_date = today

# Fetch historical 3-month treasury rates
print("Fetching historical 3-month treasury rates...")
try:
    df = web.DataReader('DGS3MO', 'fred', start_date, end_date)
    if isinstance(df, pd.Series):
        df = df.to_frame(name='Rf')
    else:
        df = df.rename(columns={'DGS3MO': 'Rf'})
    
    df = df.fillna(method='ffill')  
    df = df.dropna()  
    
    # Calculate daily changes and volatility using Vasicek regression
    delta = 1/252  
    df['Rf'] = df['Rf'] / 100 
    df['dR'] = df['Rf'].diff()
    df = df.dropna()
    
    # Vasicek regression for Hull-White calibration
    X = np.column_stack([np.ones(len(df)-1), df['Rf'].values[:-1]])
    y = df['dR'].values[1:]
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    residuals = y - X @ beta
    sigma = np.std(residuals) / np.sqrt(delta)
    a = -beta[1] / delta  
    r0 = df['Rf'].iloc[-1]  
    
    print(f"Calibrated Hull-White parameters:")
    print(f"sigma: {sigma:.6f}")
    print(f"a: {a:.6f}")
    print(f"Initial short rate (r0): {r0:.6f}")
except Exception as e:
    print(f"Error fetching data: {str(e)}")
    print("Using default values")
    sigma = 0.0078  
    a = 0.1        
    r0 = 0.0525   

# Read and process treasury data for term structure
print("\nProcessing treasury data...")
f1 = 'bills2024-03-05.txt'
f2 = 'bonds2024-03-05.txt'


bills = pd.read_csv(f1, sep='\t')
bonds = pd.read_csv(f2, sep='\t')

# Rename columns for consistency
bills = bills.rename(columns={
    'MATURITY': 'Maturity',
    'BID': 'Bid',
    'ASKED': 'Asked',
    'CHG': 'Chg',
    'ASKED YIELD': 'Askedyield'
})

bonds = bonds.rename(columns={
    'MATURITY': 'Maturity',
    'COUPON': 'Coupon',
    'BID': 'Bid',
    'ASKED': 'Asked',
    'CHG': 'Chg',
    'ASKED YIELD': 'Askedyield'
})

# Clean data 
bills = bills[~bills.Asked.isin(['n.a.', 'n.a'])]
bonds = bonds[~bonds.Asked.isin(['n.a.', 'n.a'])]

# Convert bond prices (decimal as fraction of 32)
bonds.Asked = pd.to_numeric(bonds.Asked)
bonds.Asked = bonds.Asked.apply(lambda x: int(x) + (x % 1) * 100/32)

# Set today's date
today = pd.to_datetime('2024-03-05')
yrlen = 365

# Convert dates and yields to proper format
bonds.Maturity = pd.to_datetime(bonds.Maturity)
bills.Maturity = pd.to_datetime(bills.Maturity)
bonds.Askedyield = pd.to_numeric(bonds.Askedyield)
bills.Askedyield = pd.to_numeric(bills.Askedyield)

# Filter bonds and reset indices
bonds = bonds[(bonds.Maturity-datetime.timedelta(yrlen)) > today]
bonds = bonds[bonds.Maturity != bonds.Maturity.shift(1)]
bills = bills[bills.Maturity != bills.Maturity.shift(1)]
bonds.index = np.arange(1, len(bonds)+1)
bills.index = np.arange(1, len(bills)+1)

# Calculate time to maturity and prices
bills['Ttm'] = pd.to_numeric((bills.Maturity-today)/datetime.timedelta(yrlen))
bonds['Ttm'] = pd.to_numeric((bonds.Maturity-today)/datetime.timedelta(yrlen))
bills['Price'] = 1./(1.+(bills.Askedyield/100)*bills.Ttm)

# Bootstrap zero-curve for bonds
bonds['ZeroPrice'] = pd.to_numeric(bonds.Asked)/100
bonds.Coupon = pd.to_numeric(bonds.Coupon)

print("\nBootstrapping zero-curve...")
for i in range(1, len(bonds)+1):
    s = np.floor(pd.to_numeric((bonds.Maturity[i]-today)/datetime.timedelta(yrlen))*2)
    while ((bonds.Maturity[i]-relativedelta(months=int(s*6)) > today) & 
           (bonds.Maturity[i]-relativedelta(months=int(s*6)) < bonds.Maturity[i])):
        cpndate = bonds.Maturity[i]-relativedelta(months=int(s*6))
        if pd.to_numeric((cpndate-today)/datetime.timedelta(yrlen)) < 1:
            absdif = abs(bills.Maturity-cpndate)
            df = bills.Price[absdif.idxmin()]
        else:
            absdif = abs(bonds.Maturity-cpndate)
            df = bonds.ZeroPrice[absdif.idxmin()]
            
        if s == np.floor(pd.to_numeric((bonds.Maturity[i]-today)/datetime.timedelta(yrlen))*2):
            bonds.ZeroPrice[i] = bonds.ZeroPrice[i] + ((bonds.Coupon[i]/100)/2)*(1-pd.to_numeric((cpndate-today)/datetime.timedelta(30*6)))
        bonds.ZeroPrice[i] = bonds.ZeroPrice[i] - ((bonds.Coupon[i]/100)/2)*df
        s = s-1
    bonds.ZeroPrice[i] = bonds.ZeroPrice[i]/(1+((bonds.Coupon[i]/100)/2))
    if i > 1 and (bonds.ZeroPrice[i]/bonds.ZeroPrice[i-1]-1) > 0.01:
        bonds.ZeroPrice[i] = 1/((1+1/(bonds.ZeroPrice[i-1]**(1/bonds.Ttm[i-1]))-1)**bonds.Ttm[i])

# Calculate zero yields
bonds['ZeroYield'] = (1/(bonds.ZeroPrice**(1/bonds.Ttm))-1)*100

# Combine bills and bonds data
zeros = pd.DataFrame((bills.Askedyield)._append(bonds.ZeroYield))
zeros.columns = ['Yield']
zeros['Price'] = (bills.Price)._append(bonds.ZeroPrice)
zeros['Maturity'] = (bills.Maturity)._append(bonds.Maturity)
zeros['Ttm'] = pd.to_numeric((zeros.Maturity-today)/datetime.timedelta(yrlen))
zeros.index = np.arange(1, len(zeros)+1)

# Filter for maturities >= 3 months
zeros = zeros[zeros.Ttm >= 0.25]

# Fit polynomial to bootstrapped discount factors
print("\nFitting polynomial to discount factors...")
poly_fit = np.polyfit(zeros.Ttm, zeros.Price, 9)
zeros['PolyFit'] = np.polyval(poly_fit, zeros.Ttm)

# Function to calculate Hull-White bond price
def hull_white_bond_price(t, T, r0, a, sigma, zeta_t):
    tau = T - t
    B = (1 - np.exp(-a*tau))/a
    A = np.exp(zeta_t - (sigma**2/(4*a**3))*(1 - np.exp(-2*a*tau)) - (sigma**2*B**2)/(4*a))
    return A * np.exp(-B*r0)

# Function to calculate forward rate
def forward_rate(t, P):
    return -(np.log(P[1:]) - np.log(P[:-1]))/(t[1:] - t[:-1])

# Calculate initial forward rate curve
ttm_grid = np.arange(0.25, 10.0 + 1/12, 1/12)
P_market = np.polyval(poly_fit, ttm_grid)
f_market = forward_rate(ttm_grid, P_market)

# Calibrate zeta(t) to match market prices
def calibrate_zeta(ttm_grid, P_market, r0, a, sigma):
    zeta = np.zeros_like(ttm_grid)
    for i in range(len(ttm_grid)):
        def obj(z):
            return (hull_white_bond_price(0, ttm_grid[i], r0, a, sigma, z) - P_market[i])**2
        res = minimize(obj, x0=0.0, method='Nelder-Mead')
        zeta[i] = res.x[0]
    return zeta

print("\nCalibrating zeta(t)...")
zeta = calibrate_zeta(ttm_grid, P_market, r0, a, sigma)

# Calculate model prices and rates
model_prices = np.array([hull_white_bond_price(0, t, r0, a, sigma, z) 
                        for t, z in zip(ttm_grid, zeta)])
market_rates = -np.log(zeros.PolyFit)/zeros.Ttm * 100
model_rates = -np.log(model_prices)/ttm_grid * 100

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(zeros.Ttm, market_rates, 'b.', label='Market Rates')
plt.plot(ttm_grid, model_rates, 'r-', label='Hull-White Model Rates')
plt.title('Term Structure of Interest Rates (2024-03-05)')
plt.xlabel('Time to Maturity (years)')
plt.ylabel('Interest Rate (%)')
plt.legend()
plt.grid(True)
plt.show()

# Save calibrated parameters
params = np.array([sigma, a])
np.savetxt('hull_white_params.txt', params)
np.savetxt('hull_white_zeta.txt', zeta)
print("\nCalibration complete. Parameters saved to files.") 