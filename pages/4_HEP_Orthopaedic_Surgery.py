import streamlit as st
import pandas as pd
from examples.ex_3_theatres_beds.simulation_execution_functions import single_run, multiple_replications
from examples.ex_3_theatres_beds.model_classes import Scenario, Schedule
from output_animation_functions import animate_activity_log
import gc

st.set_page_config(layout="wide", 
                   initial_sidebar_state="expanded",
                   page_title="Forced Overcrowding - Simple ED")

gc.collect()

st.title("Orthopaedic Ward - Hospital Efficiency Project")

st.markdown(
    """
This is the orthopaedic surgery model developed as part of the hospital efficiency project. 

    """
)

button_run_pressed = st.button("Run simulation")

if button_run_pressed:

    args = Scenario(schedule=Schedule())
        

    # model = TreatmentCentreModel(args)

    # st.subheader("Single Run")

    # results_df = single_run(args)

    # st.dataframe(results_df)

    # st.subheader("Multiple Runs")

    results = multiple_replications(
                    return_detailed_logs=True,
                    scenario=args
                )
    
    st.dataframe(results[0])

