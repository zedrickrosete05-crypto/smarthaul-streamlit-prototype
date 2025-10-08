# SmartHaul Streamlit Prototype

Streamlit app for route optimization and live monitoring.

## Features
- Upload orders (CSV), validate columns
- Greedy VRPTW-ish optimizer (optional OR-Tools)
- Live monitoring panel with basic risk alerts
- KPI view: on-time %, ETA MAE placeholder

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
