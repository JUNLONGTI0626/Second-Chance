# Project goal
Study time-varying connectedness among AI, electricity, gas, oil, gold, and coal-related variables in the U.S. market.

# Variables
- AI: AI index
- ELEC: electricity index
- COAL: coal index
- GAS: Henry Hub price
- WTI: WTI crude oil price
- GOLD: gold price

# Required methods
1. DCC-GARCH as auxiliary evidence
2. TVP-VAR-SV as main model
3. TVP-VAR-BK as frequency-domain extension

# Output requirements
- Save cleaned data to data/processed/
- Save tables to outputs/tables/
- Save figures to outputs/figures/
- Always log sample period, missing values, and variable transformations
