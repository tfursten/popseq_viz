#!/usr/bin/env python3
"""Main entry point for the nested SNP explorer and the original tree viewer."""
from pathlib import Path
import sys

import streamlit as st

# Keep sibling imports working for normal launches and Streamlit page execution.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

page = st.navigation([
    st.Page("tree_app.py", title="Original tree viewer", default=True),
    st.Page("nested_app.py", title="Nested SNP Explorer (Beta)"),
])
with st.spinner("Updating visualization…"):
    page.run()
