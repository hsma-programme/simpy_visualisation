import streamlit as st
import pandas as pd
import gc

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

gc.collect()

st.title("Visual Interactive Simulation (VIS) - Demonstration")

st.markdown(
    """
This streamlit app demonstrates the use of a visual interactive simulation (VIS) package for showing the position of queues and resource utilisation in a manner understandable to stakeholders.

It is also valuable for developers, as the functioning of the simulation can be more easily monitored.

It is designed for integration with simpy - however, in theory, it could be integrated with different simulation packages in Python or other languages.

Please use the tabs on the left hand side to view different examples of how this package can be used.
    """
)
# st.divider()

# st.subheader("Technical Notes")

# st.markdown(
#     """
# This
#     """
# )

st.divider()

st.subheader("Models used as examples")

st.markdown(
    """
The underlying code for the emergency department model:
Monks.T, Harper.A, Anagnoustou. A, Allen.M, Taylor.S. (2022) Open Science for Computer Simulation
https://github.com/TomMonks/treatment-centre-sim    
""")

with st.expander("Licence: Treatment Centre Model"):

    st.markdown(
        """
    MIT License

    Copyright (c) 2021 Tom Monks

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
        """
)

st.markdown(
"""
The layout code for the emergency department model:


The hospital efficiency project model:
Harper, A., & Monks, T. Hospital Efficiency Project Orthopaedic Planning Model Discrete-Event Simulation [Computer software]. 
https://doi.org/10.5281/zenodo.7951080
https://github.com/AliHarp/HEP/tree/main 
    """
)

with st.expander("Licence: HEP"):
    st.markdown(
    """
MIT License

Copyright (c) 2022 AliHarp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
    """
    )