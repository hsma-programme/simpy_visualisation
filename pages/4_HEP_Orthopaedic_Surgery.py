import streamlit as st
import pandas as pd
from examples.ex_3_theatres_beds.simulation_execution_functions import single_run, multiple_replications
from examples.ex_3_theatres_beds.model_classes import Scenario, Schedule
from output_animation_functions import animate_activity_log
import gc

st.set_page_config(layout="wide", 
                   initial_sidebar_state="expanded",
                   page_title="Orthopaedic Ward - HEP")

gc.collect()

st.title("Orthopaedic Ward - Hospital Efficiency Project")

st.markdown(
    """
This is the orthopaedic surgery model developed as part of the hospital efficiency project. 

    """
)

TRACE = True

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
    
    st.subheader("Summary of Results")
    st.dataframe(results[0])

    # st.dataframe(results[1])

    # st.dataframe(results[2])

    # st.dataframe(results[3])

    st.subheader("Event Log")
    # st.dataframe(results[4])

    # Join the event log with a list of patients to add a column that will determine
    # the icon set used for a patient (in this case, we want to distinguish between the 
    # knee/hip patients)

    event_log = results[4]
    event_log = event_log[event_log['rep'] == 1]

    primary_patients = results[2]
    primary_patients = primary_patients[primary_patients['rep'] == 1]
    primary_patients['patient class'] = primary_patients['patient class'].str.title()

    revision_patients = results[3]
    revision_patients = revision_patients[revision_patients['rep'] == 1]
    revision_patients['patient class'] = revision_patients['patient class'].str.title()

    full_log_with_patient_details = event_log.merge(pd.concat([primary_patients, revision_patients]), 
                                                     how="left",
                                                    left_on=["patient", "pathway"],
                                                    right_on=["ID", "patient class"])
    
    st.dataframe(full_log_with_patient_details)
    
    event_position_df = pd.DataFrame([
                # {'event': 'arrival', 'x':  10, 'y': 250, 'label': "Arrival" },
                
                # Triage - minor and trauma                
                {'event': 'enter_queue_for_bed', 
                 'x':  200, 'y': 400, 'label': "Waiting for<br>Bed" },
                {'event': 'post_surgery_stay_begins', 
                 'x':  625, 'y': 300, 'resource':'n_beds', 'label': "In bed" },
                {'event': 'exit', 
                 'x':  670, 'y': 400, 'label': "Exit"}

                ])
    st.plotly_chart(
            animate_activity_log(
                event_log=event_log,
                event_position_df= event_position_df,
                scenario=args,
                debug_mode=True,
                every_x_time_units=1,
                include_play_button=True,
                return_df_only=False,
                gap_between_entities=8,
                gap_between_rows=20,
                plotly_height=700,
                plotly_width=900,
                override_x_max=700,
                override_y_max=550,
                icon_and_text_size=14,
                wrap_queues_at=10,
                step_snapshot_max=50,
                frame_duration=1000,
                # time_display_units="dhm",
                display_stage_labels=True,
                limit_duration=42,
                # add_background_image="https://raw.githubusercontent.com/hsma-programme/Teaching_DES_Concepts_Streamlit/main/resources/Full%20Model%20Background%20Image%20-%20Horizontal%20Layout.drawio.png",
            ), use_container_width=False,
                config = {'displayModeBar': False}
        )                                               

    