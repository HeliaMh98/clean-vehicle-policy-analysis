# Data

Raw data files are **not included** in this repository. All datasets used in this project are publicly available. Follow the instructions below to download and prepare them for analysis.

---

## Required Datasets

### 1. AFDC Policy Database — Alternative Fuel Vehicle Laws & Incentives

**Source:** U.S. Department of Energy, Alternative Fuels Data Center  
**URL:** https://afdc.energy.gov/laws  
**Access:** Use the AFDC Laws & Incentives API or download via the "Download Data" button on the portal.

**Download steps:**
1. Navigate to https://afdc.energy.gov/laws/search
2. Filter by: Country = United States; Technology = All; Category = All
3. Click "Download" → select CSV format
4. Save as `data/raw/afdc_policies_raw.csv`

**Key columns used:**
| Column | Description |
|---|---|
| `id` | Unique policy identifier |
| `title` | Short policy title |
| `text` | Full policy description text (input to LLM classifier) |
| `state` | Two-letter state abbreviation |
| `enacted_date` | Year policy was enacted |
| `expired_date` | Year policy expired (if applicable) |
| `categories` | AFDC's own category tags (not used in classification) |
| `vehicle_types` | Applicable vehicle technologies |

---

### 2. AFV Registration Data

**Source:** AFDC Vehicle Registration Counts by State  
**URL:** https://afdc.energy.gov/vehicle-registration  

**Download steps:**
1. Navigate to https://afdc.energy.gov/vehicle-registration
2. Select "All Fuels" and download yearly data for 2013–2020
3. Save each year as `data/raw/registrations_{year}.csv`

**Alternatively**, use the AFDC API:
```bash
# Example: fetch 2019 registration data
curl "https://developer.nrel.gov/api/afdc/v1/vehicle-registration.json?api_key=YOUR_KEY&year=2019" \
  -o data/raw/registrations_2019.json
```
Free API key available at: https://developer.nrel.gov/signup/

**Key columns used:**
| Column | Description |
|---|---|
| `state` | State FIPS code or abbreviation |
| `fuel` | Fuel type (BEV, PHEV, HEV, Hydrogen, Propane, etc.) |
| `registration_count` | Number of registered vehicles |
| `total_vehicles` | Total vehicle registrations in state-year |

Compute `share_{fuel} = registration_count / total_vehicles * 100` as the outcome variable.

---

### 3. State Socioeconomic Covariates

These are merged from three federal sources:

#### GDP per Capita
**Source:** U.S. Bureau of Economic Analysis (BEA)  
**URL:** https://apps.bea.gov/regional/downloadzip.cfm  
**Dataset:** SAGDP — Annual GDP by State  
Download `SAGDP1__ALL_AREAS_1997_2023.csv` and filter to 2013–2020.

#### Electricity Price
**Source:** U.S. Energy Information Administration (EIA)  
**URL:** https://www.eia.gov/electricity/data/state/  
**File:** `avgprice_annual.xlsx` — Average retail price of electricity by state and sector  
Use the "Residential" sector column.

#### Population Density, Median Age, Employment Rate
**Source:** U.S. Census Bureau / American Community Survey (ACS)  
**URL:** https://data.census.gov  
**Table:** `B01002` (Median Age), `B23025` (Employment Status), `B01003` (Total Population)  
Filter to 5-year ACS estimates for each year; divide population by state area (sq mi) for density.

---

## Panel Construction

After downloading all files, run `notebooks/01_data_cleaning.ipynb` to:

1. **Parse AFDC policy text** into a state-year binary treatment matrix (one column per mechanism, one row per state-year)
2. **Construct AFV adoption shares** from registration counts
3. **Merge covariates** on `(state_fips, year)`
4. **Export** the final panel as `panel_data.csv` (not committed to this repo)

### Expected panel structure

```
state_fips | year | share_bev | share_phev | share_hev | ... | policy_upfront_cost | policy_access_convenience | ... | electricity_price | gdp_per_capita | ...
-----------|------|-----------|------------|-----------|-----|---------------------|---------------------------|-----|-------------------|----------------|
01         | 2013 | 0.12      | 0.08       | 1.43      | ... | 0                   | 1                         | ... | 11.2              | 42100          | ...
...
```

**Panel dimensions:** 51 states × 8 years = 408 observations per vehicle type.

---

## Notes on Policy-Year Assignment

A policy is considered "active" in a given state-year if:
- `enacted_date ≤ year`, and
- `expired_date` is null OR `expired_date > year`

This produces a balanced binary treatment indicator for each mechanism in each state-year.
