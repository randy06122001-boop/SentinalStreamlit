What Is SENTINEL?
SENTINEL is a composite geopolitical fear index (0-100) that combines four orthogonal signals to quantify how much geopolitical risk the market is pricing in at any given moment. It was built to answer one question: can satellite-detected military buildups predict market returns before a shot is fired?

The index draws on academic research by Caldara and Iacoviello (American Economic Review, 2022), who decomposed geopolitical risk into a "Threats" component (military buildups, war threats, nuclear threats) and an "Acts" component (beginning of war, escalation, terrorist acts). Their key finding — confirmed by Goncalves et al. (2025) — is that only the threat component is consistently priced across assets. Realized acts have a "much weaker link to risk premia."

SENTINEL operationalizes this insight by weighting the threat signal most heavily and combining it with real-time volatility and prediction market data.

GPR data updates weekly, not daily. Intraday geopolitical shocks are captured by VIX and Polymarket but not GPR.

The backtest uses the full sample for calibration. A proper out-of-sample test would require a train/test split (train 1990-2015, test 2016-2026).

Polymarket liquidity on some geopolitical contracts is thin, making probabilities noisy.

OVX data only goes back to 2007, so the 15% weight uses a shorter history.

Commercial satellite imagery restrictions (as seen during Iran 2026) can degrade the OSINT-to-market pipeline in ways the index cannot capture.

Future additions: Brent crude call/put skew as a 5th signal, sector rotation overlays, and automated threshold alerts.

Link
https://sentinalapp.streamlit.app/
