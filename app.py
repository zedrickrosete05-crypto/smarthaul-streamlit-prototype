import streamlit as st
from smarthaul_core.dataio import load_and_validate_orders
from smarthaul_core.distance import get_distance_matrix
from smarthaul_core.solver import solve_greedy

st.write("✅ All imports worked! Now re-enable main app logic.")
