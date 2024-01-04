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
st.caption(
    """
@software{Harper_Hospital_Efficiency_Project,

author = {Harper, Alison and Monks, Thomas},

license = {MIT},

title = {{Hospital Efficiency Project  Orthopaedic Planning Model Discrete-Event Simulation}},

url = {https://github.com/AliHarp/HEP}
} 
""")

st.markdown(
    """
It has been used as a test case here to allow the development and testing of several key features of the event log animations:
    
- adding of logging to a model from scratch

- ensuring the requirement to use simpy stores instead of simpy resources doesn't prevent the uses of certain common modelling patterns (in this case, conditional logic where patients will leave the system if a bed is not available within a specified period of time)

- displaying different icons for different classes of patients

- displaying custom resource icons

- displaying additional static information as part of the icon (in this case, whether the client's discharge is delayed)

- displaying information that updates with each animation step as part of the icon (in this case, the LoS of the patient at each time point)
    """
)

st.divider()

TRACE = True
debug_mode=True

schedule = Schedule()

col_a, col_b = st.columns(2)

with col_a:
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

with col_b:
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
    

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown('# Model Parameters')
		
    st.markdown('## Ring-fenced beds:')
    n_beds = st.slider('Beds', 10, 80, 40, 1)

    st.markdown('## Mean lengths-of-stay for each type of surgery:')
    primary_hip_los = st.slider('Primary Hip LoS', 1.0, 10.0, 4.4, 0.1)

    primary_knee_los = st.slider('Primary Knee LoS', 1.0, 10.0, 4.7, 0.1)

    revision_hip_los = st.slider('Revision Hip LoS', 2.0, 10.0, 6.9 , 0.1)

    revision_knee_los = st.slider('Revision Knee LoS', 2.0, 10.0, 7.2, 0.1)

    unicompart_knee_los = st.slider('Unicompart knee LoS', 1.0, 10.0,2.9, 0.1)

with col2:
    st.markdown('## Mean length of delayed discharge:')
    los_delay = st.slider('Mean length of delay', 2.0, 10.0,16.5, 0.1)
    los_delay_sd = st.slider('Variation of delay (standard deviation)', 1.0, 25.0,15.2, 0.1)
		
    st.markdown('## Proportion of patients with a discharge delay:')
    prop_delay = st.slider('Proportion delayed', 0.00, 1.00, 0.076, 0.01)
		
    st.markdown('## :green[Model execution]')
    replications = st.slider(':green[Number of runs]', 1, 50, 30)
    runtime = st.slider(':green[Runtime (days)]', 30, 100, 60)
    warmup=st.slider(':green[Warmup (days)]', 1, 14, 7)

button_run_pressed = st.button("Run simulation")

args = Scenario(schedule=schedule,
                primary_hip_mean_los=primary_hip_los,
                primary_knee_mean_los=primary_knee_los,
                revision_hip_mean_los=revision_hip_los,
                revision_knee_mean_los=revision_knee_los,
                unicompart_knee_mean_los=unicompart_knee_los,
                prob_ward_delay=prop_delay,
                n_beds=n_beds,
                delay_post_los_mean=los_delay,
                delay_post_los_sd=los_delay_sd
                )

if button_run_pressed:

    results = multiple_replications(
                    return_detailed_logs=True,
                    scenario=args,
                    n_reps=replications,
                    results_collection=runtime
                )
    
    st.subheader("Summary of Results")
    st.dataframe(results[0])

    
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
                                                    right_on=["ID", "patient class"]).reset_index(drop=True).drop(columns="ID")
    
    pid_table = full_log_with_patient_details[['patient']].drop_duplicates().reset_index(drop=True).reset_index(drop=False).rename(columns={'index': 'pid'})

    full_log_with_patient_details = full_log_with_patient_details.merge(pid_table, how='left', on='patient').drop(columns='patient').rename(columns={'pid':'patient'})
    
    event_position_df = pd.DataFrame([
                # {'event': 'arrival', 'x':  10, 'y': 250, 'label': "Arrival" },
                
                # Triage - minor and trauma                
                {'event': 'enter_queue_for_bed', 
                 'x':  200, 'y': 450, 'label': "Waiting for<br>Availability of<br>Bed to be Confirmed<br>Before Surgery" },

                {'event': 'no_bed_available', 
                 'x':  600, 'y': 450, 'label': "No Bed<br>Available:<br>Surgery Cancelled" },

                {'event': 'post_surgery_stay_begins', 
                 'x':  650, 'y': 200, 'resource':'n_beds', 'label': "In Bed:<br>Recovering from<br>Surgery" },

                {'event': 'discharged_after_stay', 
                 'x':  670, 'y': 50, 'label': "Discharged from Hospital<br>After Recovery"}
                # {'event': 'exit', 
                #  'x':  670, 'y': 100, 'label': "Exit"}

                ])
    
    full_patient_df = reshape_for_animations(full_log_with_patient_details, 
                                             every_x_time_units=1,
                                             limit_duration=runtime,
                                             step_snapshot_max=50,
                                             debug_mode=debug_mode
                                             )
    
    if debug_mode:
        print(f'Reshaped animation dataframe finished construction at {time.strftime("%H:%M:%S", time.localtime())}')
    
    full_patient_df_plus_pos = generate_animation_df(
                                full_patient_df=full_patient_df,
                                event_position_df=event_position_df,
                                wrap_queues_at=20,
                                step_snapshot_max=50,
                                gap_between_entities=15,
                                gap_between_resources=15,
                                gap_between_rows=50,
                                debug_mode=debug_mode
                        )
    
    def set_icon(row):
        if row["surgery type"] == "p_knee":
            return "🦵<br>1️⃣<br> "
        elif row["surgery type"] == "r_knee":
            return "🦵<br>♻️<br> "
        elif row["surgery type"] == "p_hip":
            return "🕺<br>1️⃣<br> "
        elif row["surgery type"] == "r_hip":
            return "🕺<br>♻️<br> "
        elif row["surgery type"] == "uni_knee":
            return "🦵<br>✳️<br> "
        else:
            return f"CHECK<br>{row['icon']}"

    full_patient_df_plus_pos = full_patient_df_plus_pos.assign(icon=full_patient_df_plus_pos.apply(set_icon, axis=1))

    # TODO: Check why this doesn't seem to be working quite right for the 'discharged after stay'
    # step. e.g. 194Primary is discharged on 28th July showing a LOS of 1 but prior to this shows a LOS of 9.
    def add_los_to_icon(row):
        if row["event"] == "post_surgery_stay_begins":
            return f'{row["icon"]}<br>{row["minute"]-row["time"]:.0f}' 
        elif row["event"] == "discharged_after_stay":
            return f'{row["icon"]}<br>{row["los"]:.0f}' 
        else:
            return row["icon"] 
        
    full_patient_df_plus_pos = full_patient_df_plus_pos.assign(icon=full_patient_df_plus_pos.apply(add_los_to_icon, axis=1))

    
    def indicate_delay_via_icon(row):
        if row["delayed discharge"] is True:
            return f'{row["icon"]}<br>*'
        else:
            return f'{row["icon"]}<br> '

    full_patient_df_plus_pos = full_patient_df_plus_pos.assign(icon=full_patient_df_plus_pos.apply(indicate_delay_via_icon, axis=1))


    with st.expander("Click here to view detailed event dataframes"):
        st.subheader("Event Log")
        st.subheader("Data - After merging full log with patient details")
        st.dataframe(full_log_with_patient_details)
        st.subheader("Dataframe - Reshaped for animation (step 1)")
        st.dataframe(full_patient_df)
        st.subheader("Dataframe - Reshaped for animation (step 2)")
        st.dataframe(full_patient_df_plus_pos)

    cancelled_due_to_no_bed_available = len(full_log_with_patient_details[full_log_with_patient_details['event'] == "no_bed_available"]["patient"].unique())
    total_patients = len(full_log_with_patient_details["patient"].unique())

    cancelled_perc = cancelled_due_to_no_bed_available/total_patients

    st.markdown(f"Surgeries cancelled due to no bed being available in time: {cancelled_perc:.2%} ({cancelled_due_to_no_bed_available} of {total_patients})")

    st.markdown(
        """
        **Key**: 
        
        🦵1️⃣: Primary Knee
        
        🦵♻️: Revision Knee
        
        🕺1️⃣: Primary Hip
        
        🕺♻️: Revision Hip
        
        🦵✳️: Primary Unicompartment Knee
        
        An asterisk (*) indicates that the patient has a delayed discharge from the ward.

        The numbers below patients indicate their length of stay.
        """
    )
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
            custom_resource_icon="🛏️",
            time_display_units="d",
            start_date="2022-06-27",
            setup_mode=False,
            frame_duration=1500, #milliseconds
            frame_transition_duration=1000, #milliseconds
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

    