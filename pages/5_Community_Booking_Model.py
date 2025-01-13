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
from vidigi.prep import reshape_for_animations, generate_animation_df
from vidigi.animation import generate_animation
# from plotly.subplots import make_subplots

st.set_page_config(layout="wide",
                   initial_sidebar_state="expanded",
                   page_title="Mental Health - Booking Model")

gc.collect()

st.title("Mental Health - Appointment Booking Model")

st.markdown(
    """
This model looks at a simple mental health pathway.

In this model, we are only concerned with the booking of an initial appointment.

By default, the model uses an appointment book with some slots held back for high-priority patients.
Each patient in the default scenario can only go to their 'home'/most local clinic.

However, it is possible to switch to other scenarios
- a 'pooling' system where patients can choose between one of several linked clinics in their local area (with the assumption that they will choose the clinic of the group with the soonest available appointment)
- the pooling system described above, but with no slots held back for high-priority patients (i.e. no 'carve-out')
    """
)

# args = Scenario()

#example solution...

st.subheader("Weekly Slots")
st.markdown("Edit the number of daily slots available per clinic by clicking in the boxes below, or leave as the default schedule")
shifts = pd.read_csv("examples/ex_4_community/data/shifts.csv")
shifts_edited = st.data_editor(shifts)

scenario_choice = st.selectbox(
    'Choose a Scenario',
    ('As-is', 'With Pooling', 'With Pooling - No Carve-out'))

if scenario_choice == "As-is" or scenario_choice == "With Pooling":
    prop_carve_out = st.slider("Select proportion of carve-out", 0.0, 0.9, 0.15, 0.01)

#depending on settings and CPU this model takes around 15-20 seconds to run

button_run_pressed = st.button("Run simulation")

if button_run_pressed:

    # add a spinner and then display success box
    with st.spinner('Simulating the community booking system...'):

        RESULTS_COLLECTION = 365 * 1

        #We will learn about warm-up periods in a later lab.  We use one
        #because the model starts up empty which doesn't reflect reality
        WARM_UP = 365 * 1
        RUN_LENGTH = RESULTS_COLLECTION + WARM_UP

        #set up the scenario for the model to run.
        scenarios = {}

        scenarios['as-is'] = Scenario(RUN_LENGTH,
                                      WARM_UP,
                                      prop_carve_out=prop_carve_out,
                                      seeds=generate_seed_vector(),
                                      slots_file=shifts_edited)

        scenarios['pooled'] = Scenario(RUN_LENGTH,
                                       WARM_UP,
                                       prop_carve_out=prop_carve_out,
                                       pooling=True,
                                       seeds=generate_seed_vector(),
                                       slots_file=shifts_edited)

        scenarios['no_carve_out'] = Scenario(RUN_LENGTH,
                                             WARM_UP,
                                             pooling=True,
                                             prop_carve_out=0.0,
                                             seeds=generate_seed_vector(),
                                             slots_file=shifts_edited)

        col1, col2, col3 = st.columns(3)

        if scenario_choice == "As-is":
            st.subheader("As-is")
            results_all, results_low, results_high, event_log = single_run(scenarios['as-is'])
            st.dataframe(results_summary(results_all, results_low, results_high))
        elif scenario_choice == "With Pooling":
            st.subheader("With Pooling")
            results_all, results_low, results_high, event_log = single_run(scenarios['pooled'])
            st.dataframe(results_summary(results_all, results_low, results_high))

        elif scenario_choice == "With Pooling - No Carve-out":
            st.subheader("Pooled with no carve out")
            results_all, results_low, results_high, event_log = single_run(scenarios['no_carve_out'])
            st.dataframe(results_summary(results_all, results_low, results_high))


        event_log_df = pd.DataFrame(event_log)


        event_log_df['event_original'] = event_log_df['event']
        event_log_df['event'] = event_log_df.apply(lambda x: f"{x['event']}{f'_{int(x.booked_clinic)}' if pd.notna(x['booked_clinic']) else ''}", axis=1)

        full_patient_df = reshape_for_animations(event_log_df,
                                                 limit_duration=WARM_UP+180,
                                                 every_x_time_units=1,
                                                 step_snapshot_max=50)

        # Remove the warm-up period from the event log
        full_patient_df = full_patient_df[full_patient_df["minute"] >= WARM_UP]


        clinics =  [x for x in event_log_df['booked_clinic'].sort_values().unique().tolist() if not math.isnan(x)]

        clinic_waits = [{'event': f'appointment_booked_waiting_{int(clinic)}',
          'y':  950-(clinic+1)*80,
          'x': 625,
          'label': f"Booked into<br>clinic {int(clinic)}",
          'clinic': int(clinic)}
          for clinic in clinics]

        clinic_attends = [{'event': f'have_appointment_{int(clinic)}',
          'y':  950-(clinic+1)*80,
          'x': 850,
          'label': f"Attending appointment<br>at clinic {int(clinic)}"}
          for clinic in clinics]

        event_position_df = pd.concat([pd.DataFrame(clinic_waits),(pd.DataFrame(clinic_attends))])

        referred_out = [{'event': f'referred_out_{int(clinic)}',
          'y':  950-(clinic+1)*80,
          'x': 125,
          'label': f"Referred Out From <br>clinic {int(clinic)}"}
          for clinic in clinics]

        event_position_df = pd.concat([event_position_df,(pd.DataFrame(referred_out))])

        # event_position_df = pd.concat([
        #     event_position_df,
        #     pd.DataFrame([{'event': 'exit', 'x':  270, 'y': 70, 'label': "Exit"}])]) .reset_index(drop=True)

        clinic_lkup_df = pd.DataFrame([
            {'clinic': 0, 'icon': "🟠"},
            {'clinic': 1, 'icon': "🟡"},
            {'clinic': 2, 'icon': "🟢"},
            {'clinic': 3, 'icon': "🔵"},
            {'clinic': 4, 'icon': "🟣"},
            {'clinic': 5, 'icon': "🟤"},
            {'clinic': 6, 'icon': "⚫"},
            {'clinic': 7, 'icon': "⚪"},
            {'clinic': 8, 'icon': "🔶"},
            {'clinic': 9, 'icon': "🔷"},
            {'clinic': 10, 'icon': "🟩"}
        ])


        if scenario_choice == "With Pooling" or scenario_choice == "With Pooling - No Carve-out":
            event_position_df = event_position_df.merge(clinic_lkup_df, how="left")
            event_position_df["label"] = event_position_df.apply(lambda x: f"{x['label']} {x['icon']}" if pd.notna(x['icon']) else x['label'], axis=1)
            event_position_df = event_position_df.drop(columns="icon")

        event_position_df = event_position_df.drop(columns="clinic")

        full_patient_df_plus_pos = generate_animation_df(
                            full_patient_df=full_patient_df,
                            event_position_df=event_position_df,
                            wrap_queues_at=25,
                            step_snapshot_max=50,
                            gap_between_entities=15,
                            gap_between_resources=15,
                            gap_between_rows=15,
                            debug_mode=True
                    )



        if scenario_choice == "With Pooling" or scenario_choice == "With Pooling - No Carve-out":
            def show_home_clinic(row):
                if "more" not in row["icon"]:
                    if row["home_clinic"] == 0:
                        return "🟠"
                    if row["home_clinic"] == 1:
                        return "🟡"
                    if row["home_clinic"] == 2:
                        return "🟢"
                    if row["home_clinic"] == 3:
                        return "🔵"
                    if row["home_clinic"] == 4:
                        return "🟣"
                    if row["home_clinic"] == 5:
                        return "🟤"
                    if row["home_clinic"] == 6:
                        return "⚫"
                    if row["home_clinic"] == 7:
                        return "⚪"
                    if row["home_clinic"] == 8:
                        return "🔶"
                    if row["home_clinic"] == 9:
                        return "🔷"
                    if row["home_clinic"] == 10:
                        return "🟩"
                    else:
                        return row["icon"]
                else:
                    return row["icon"]

            full_patient_df_plus_pos = full_patient_df_plus_pos.assign(icon=full_patient_df_plus_pos.apply(show_home_clinic, axis=1))


        def show_priority_icon(row):
            if "more" not in row["icon"]:
                if row["pathway"] == 2:
                    if scenario_choice == "As-is":
                        return "🚨"
                    else:
                        return f"{row['icon']}*"
                else:
                    return row["icon"]
            else:
                return row["icon"]

        def add_los_to_icon(row):
            if row["event_original"] == "have_appointment":
                return f'{row["icon"]}<br>{int(row["wait"])}'
            else:
                return row["icon"]

        full_patient_df_plus_pos = full_patient_df_plus_pos.assign(
            icon=full_patient_df_plus_pos.apply(show_priority_icon, axis=1)
            )

        full_patient_df_plus_pos = full_patient_df_plus_pos.assign(
            icon=full_patient_df_plus_pos.apply(add_los_to_icon, axis=1)
            )

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
