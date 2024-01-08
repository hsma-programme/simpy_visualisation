import gc
import time
import math
import datetime as dt
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from examples.ex_4_community.model_classes import Scenario, generate_seed_vector
from examples.ex_4_community.simulation_execution_functions import single_run
from examples.ex_4_community.simulation_summary_functions import results_summary
from output_animation_functions import reshape_for_animations, generate_animation_df, generate_animation, animate_activity_log
# from plotly.subplots import make_subplots

st.set_page_config(layout="wide", 
                   initial_sidebar_state="expanded",
                   page_title="Mental Health - Booking Model")

gc.collect()

st.title("Mental Health - Appointment Booking Model")

# args = Scenario()

#example solution...

st.subheader("Weekly Slots")
shifts = pd.read_csv("examples/ex_4_community/data/shifts.csv")
shifts_edited = st.data_editor(shifts)

#depending on settings and CPU this model takes around 15-20 seconds to run 

button_run_pressed = st.button("Run simulation")

if button_run_pressed:

    # add a spinner and then display success box
    with st.spinner('Simulating the community booking system...'):

        #5 day a week model = 260 days a year
        RESULTS_COLLECTION = 260 * 1

        #We will learn about warm-up periods in a later lab.  We use one
        #because the model starts up empty which doesn't reflect reality
        WARM_UP = 260 * 3
        RUN_LENGTH = RESULTS_COLLECTION + WARM_UP

        #set up the scenario for the model to run.
        scenarios = {}
        scenarios['as-is'] = Scenario(RUN_LENGTH, WARM_UP, 
                                      seeds=generate_seed_vector(),
                                      slots_file=shifts_edited)
        scenarios['pooled'] = Scenario(RUN_LENGTH, WARM_UP, pooling=True,
                                        seeds=generate_seed_vector(),
                                      slots_file=shifts_edited)
        scenarios['no_carve_out'] = Scenario(RUN_LENGTH, WARM_UP, pooling=True, 
                                                prop_carve_out=0.0, 
                                                seeds=generate_seed_vector(),
                                      slots_file=shifts_edited)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("As-is")
            results_all_as_is, results_low_as_is, results_high_as_is, event_log_as_is = single_run(scenarios['as-is'])
            st.dataframe(results_summary(results_all_as_is, results_low_as_is, results_high_as_is))

        # with col2:
        #     st.subheader("Pooled")
        #     results_all_pooled, results_low_pooled, results_high_pooled, event_log_pooled = single_run(scenarios['pooled'])
        #     st.dataframe(results_summary(results_all_pooled, results_low_pooled, results_high_pooled))

        # with col3:
        #     st.subheader("Pooled with no carve out")
        #     results_all_no_carve_out, results_low_no_carve_out, results_high_no_carve_out, event_log_no_carve_out = single_run(scenarios['no_carve_out'])
        #     st.dataframe(results_summary(results_all_no_carve_out, results_low_no_carve_out, results_high_no_carve_out))


        event_log_as_is_df = pd.DataFrame(event_log_as_is)


        event_log_as_is_df['event_original'] = event_log_as_is_df['event']
        event_log_as_is_df['event'] = event_log_as_is_df.apply(lambda x: f"{x['event']}{f'_{int(x.booked_clinic)}' if pd.notna(x['booked_clinic']) else ''}", axis=1)

        full_patient_df = reshape_for_animations(event_log_as_is_df,
                                                 limit_duration=180,
                                                 every_x_time_units=1,
                                                 step_snapshot_max=100)

        clinics =  [x for x in event_log_as_is_df['booked_clinic'].sort_values().unique().tolist() if not math.isnan(x)]

        clinic_waits = [{'event': f'appointment_booked_waiting_{int(clinic)}', 
          'y':  950-(clinic+1)*80, 
          'x': 625, 
          'label': f"Booked into<br>clinic {int(clinic)}"} 
          for clinic in clinics]
        
        clinic_attends = [{'event': f'have_appointment_{int(clinic)}', 
          'y':  950-(clinic+1)*80, 
          'x': 850, 
          'label': f"Attending appointment<br>at clinic {int(clinic)}"} 
          for clinic in clinics]
        
        event_position_df = pd.concat([pd.DataFrame(clinic_waits),(pd.DataFrame(clinic_attends))])

        # event_position_df = pd.concat([
        #     event_position_df, 
        #     pd.DataFrame([{'event': 'exit', 'x':  270, 'y': 70, 'label': "Exit"}])]) .reset_index(drop=True)

        full_patient_df_plus_pos = generate_animation_df(
                            full_patient_df=full_patient_df,
                            event_position_df=event_position_df,
                            wrap_queues_at=50,
                            step_snapshot_max=100,
                            gap_between_entities=13,
                            gap_between_resources=15,
                            gap_between_rows=15,
                            debug_mode=True
                    )
        
        def show_priority_icon(row):
            if row["pathway"] == 2:
                return "🚨"
            else:
                return row["icon"] 

        def add_los_to_icon(row):
            if row["event_original"] == "have_appointment":
                return f'{row["icon"]}<br>{int(row["wait"])}'
            else:
                return row["icon"] 

        full_patient_df_plus_pos = full_patient_df_plus_pos.assign(icon=full_patient_df_plus_pos.apply(show_priority_icon, axis=1))

        full_patient_df_plus_pos = full_patient_df_plus_pos.assign(icon=full_patient_df_plus_pos.apply(add_los_to_icon, axis=1))

        
        fig = generate_animation(
            full_patient_df_plus_pos=full_patient_df_plus_pos,
            event_position_df=event_position_df,
            scenario=None,
            plotly_height=850,
            plotly_width=1100,
            override_x_max=1000,
            override_y_max=1000,
            icon_and_text_size=10,
            # gap_between_resources=15,
            include_play_button=True,
            add_background_image=None,
            display_stage_labels=True,
            time_display_units="d",
            start_date="2022-06-27",
            setup_mode=False,
            frame_duration=1500, #milliseconds
            frame_transition_duration=1000, #milliseconds
            debug_mode=False
        )

        st.plotly_chart(fig)

    # fig.show()
        
#TODO
# Add in additional trace that shows the number of available slots per day
# using the slot df
        
#TODO
# Pooled booking version where being in non-home clinic makes you one colour
# and home clinic makes you another
        
#TODO
# Investigate adding a priority attribute to event log
# that can be considered when ranking queues if present
