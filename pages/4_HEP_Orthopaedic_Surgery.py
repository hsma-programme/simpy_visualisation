import streamlit as st
import pandas as pd
from examples.ex_3_theatres_beds.simulation_execution_functions import multiple_replications
from examples.ex_3_theatres_beds.model_classes import Scenario, Schedule
from output_animation_functions import reshape_for_animations, generate_animation_df, generate_animation, animate_activity_log
import gc
import time

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
debug_mode=True

schedule = Schedule()

st.markdown(
    """
4 theatres

5 day/week

Each theatre has three sessions per day:

Morning: 1 revision OR 2 primary

Afternoon: 1 revision OR 2 primary

Evening: 1 primary

40 ring-fenced beds for recovery from these operations
    """
)

st.dataframe(
    pd.DataFrame.from_dict(schedule.sessions_per_weekday, orient="index")
    .rename(columns={0: "Sessions"}).merge(

    pd.DataFrame.from_dict(schedule.theatres_per_weekday, orient="index")
        .rename(columns={0: "Theatre Capacity"}), 
        left_index=True, right_index=True

    ).merge(

    pd.DataFrame.from_dict(schedule.allocation, orient="index"), 
    left_index=True, right_index=True

    )
    )

args = Scenario(schedule=schedule)

button_run_pressed = st.button("Run simulation")


if button_run_pressed:

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
    event_log['patient'] = event_log['patient'].astype('str') + event_log['pathway']

    primary_patients = results[2]
    primary_patients = primary_patients[primary_patients['rep'] == 1]
    primary_patients['patient class'] = primary_patients['patient class'].str.title()
    primary_patients['ID'] = primary_patients['ID'].astype('str') + primary_patients['patient class']

    revision_patients = results[3]
    revision_patients = revision_patients[revision_patients['rep'] == 1]
    revision_patients['patient class'] = revision_patients['patient class'].str.title()
    revision_patients['ID'] = revision_patients['ID'].astype('str') + revision_patients['patient class']


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

                {'event': 'no_bed_available', 
                 'x':  400, 'y': 400, 'label': "No Bed<br>Available" },

                {'event': 'post_surgery_stay_begins', 
                 'x':  650, 'y': 300, 'resource':'n_beds', 'label': "In bed" },

                {'event': 'discharged_after_stay', 
                 'x':  670, 'y': 100, 'label': "Discharged from hospital after stay"}
                # {'event': 'exit', 
                #  'x':  670, 'y': 100, 'label': "Exit"}

                ])
    
    full_patient_df = reshape_for_animations(full_log_with_patient_details, 
                                             every_x_time_units=1,
                                             limit_duration=42,
                                             step_snapshot_max=50,
                                             debug_mode=debug_mode
                                             )
    
    st.dataframe(full_patient_df)
    
    if debug_mode:
        print(f'Reshaped animation dataframe finished construction at {time.strftime("%H:%M:%S", time.localtime())}')
    
    full_patient_df_plus_pos = generate_animation_df(
                                full_patient_df=full_patient_df,
                                event_position_df=event_position_df,
                                wrap_queues_at=20,
                                step_snapshot_max=50,
                                gap_between_entities=10,
                                gap_between_resources=15,
                                gap_between_rows=50,
                                debug_mode=debug_mode
                        )
    
    st.dataframe(full_patient_df_plus_pos)
    
    def set_icon(row):
        if row["surgery type"] == "p_knee":
            return "🦵<br>1️⃣<br> "
        elif row["surgery type"] == "r_knee":
            return "🦵<br>🔁<br> "
        elif row["surgery type"] == "p_hip":
            return "🕺<br>1️⃣<br> "
        elif row["surgery type"] == "r_hip":
            return "🕺<br>🔁<br> "
        elif row["surgery type"] == "uni_knee":
            return "🦵<br>✳️<br> "
        else:
            return f"CHECK<br>{row['icon']}"

    full_patient_df_plus_pos = full_patient_df_plus_pos.assign(icon=full_patient_df_plus_pos.apply(set_icon, axis=1))

    st.plotly_chart(
        generate_animation(
            full_patient_df_plus_pos=full_patient_df_plus_pos,
            event_position_df=event_position_df,
            scenario=args,
            plotly_height=700,
            plotly_width=1000,
            override_x_max=800,
            override_y_max=550,
            icon_and_text_size=14,
            gap_between_resources=15,
            include_play_button=True,
            add_background_image=None,
            display_stage_labels=True,
            time_display_units="d",
            start_date="2022-06-27",
            setup_mode=False,
            frame_duration=1000, #milliseconds
            frame_transition_duration=600, #milliseconds
            debug_mode=False
        )
    )

    # st.plotly_chart(
    #         animate_activity_log(
    #             event_log=event_log,
    #             event_position_df= event_position_df,
    #             scenario=args,
    #             debug_mode=True,
    #             every_x_time_units=1,
    #             include_play_button=True,
    #             gap_between_entities=8,
    #             gap_between_rows=20,
    #             plotly_height=700,
    #             plotly_width=900,
    #             override_x_max=700,
    #             override_y_max=550,
    #             icon_and_text_size=14,
    #             wrap_queues_at=10,
    #             step_snapshot_max=50,
    #             frame_duration=1000,
    #             # time_display_units="dhm",
    #             display_stage_labels=True,
    #             limit_duration=42,
    #             # add_background_image="https://raw.githubusercontent.com/hsma-programme/Teaching_DES_Concepts_Streamlit/main/resources/Full%20Model%20Background%20Image%20-%20Horizontal%20Layout.drawio.png",
    #         ), use_container_width=False,
    #             config = {'displayModeBar': False}
    #     )                                               

    