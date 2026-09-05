# Bank Stock Sensitivity to Treasury Yield Changes

## Research Question
Are bank stock returns more sensitive to changes in Treasury yields than the broader stock market?

## Hypotheses
- H1: Bank stock returns are significantly associated with changes in Treasury yields.
- H2: Bank stocks exhibit greater sensitivity to Treasury yield changes than the broader market.
- H3: Interest rate sensitivity differs across JPM, BAC, and GS.
- H4: Bank interest rate sensitivity changes over time.

## Data
- Stock returns: JPM, BAC, GS, S&P 500 (2015-2026, daily)
- Interest rate: 10 Year Treasury yield (DGS10), 2 Year Treasury yield (DGS2) as robustness
- Market uncertainty: VIX

## Methodology
Multiple regression with HAC standard errors to estimate interest rate betas, followed by 252 day rolling beta analysis to examine time variation.

## Project Structure
- data/ : raw and processed data
- notebooks/ : Jupyter notebooks for analysis
- src/ : Python scripts for data collection and processing
- paper/ : methodology, results, and writeup
- figures/ : generated plots and charts

## Status
Completed. All four hypotheses tested using OLS regression with HAC
standard errors (2015-2026 daily data). Key finding: bank stocks show
statistically significant interest rate sensitivity 7-15x larger than
the broader market, with sensitivity varying substantially over time
(near zero in 2020, peaking during the 2021-2022 rate hiking cycle).
See paper/results.md for full results.