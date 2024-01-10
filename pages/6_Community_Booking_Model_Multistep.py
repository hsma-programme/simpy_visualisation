import gc
import time
import math
import datetime as dt
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from examples.ex_5_community_follow_up.model_classes import Scenario, generate_seed_vector
from examples.ex_5_community_follow_up.simulation_execution_functions import single_run
from examples.ex_5_community_follow_up.simulation_summary_functions import results_summary
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
st.markdown("Edit the number of daily slots available per clinic by clicking in the boxes below, or leave as the default schedule")
shifts = pd.read_csv("examples/ex_5_community_follow_up/data/shifts.csv")
shifts_edited = st.data_editor(shifts)

annual_demand = st.slider("Select average annual demand", 1, 3000, 1500, 10)

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

        scenarios['pooled'] = Scenario(RUN_LENGTH,
                                       WARM_UP,
                                       prop_carve_out=prop_carve_out,
                                       pooling=True,
                                       seeds=generate_seed_vector(),
                                       slots_file=shifts_edited,
                                      annual_demand=annual_demand)

        col1, col2, col3 = st.columns(3)

        st.subheader("With Pooling")
        st.markdown("### Wait for initial appointment")
        results_all, results_low, results_high, event_log = single_run(scenarios['pooled'])
        st.dataframe(results_summary(results_all, results_low, results_high))

        event_log_df = pd.DataFrame(event_log)


        event_log_df['event_original'] = event_log_df['event']
        event_log_df['event'] = event_log_df.apply(lambda x: f"{x['event']}{f'_{int(x.booked_clinic)}' if pd.notna(x['booked_clinic']) else ''}", axis=1)

        full_patient_df = reshape_for_animations(event_log_df,
                                                 limit_duration=WARM_UP+180,
                                                 every_x_time_units=1,
                                                 step_snapshot_max=30)

        # Remove the warm-up period from the event log
        full_patient_df = full_patient_df[full_patient_df["minute"] >= WARM_UP]

        #####################################################
        # Create the positioning dataframe for the animation
        #####################################################

        # Create a list of clinics
        clinics =  [x for x in event_log_df['booked_clinic'].sort_values().unique().tolist() if not math.isnan(x)]

        # Create a column of positions for people waiting for their initial appointment with the clinic
        clinic_waits = [{'event': f'appointment_booked_waiting_{int(clinic)}',
          'y':  950-(clinic+1)*80,
          'x': 560,
          'label': f"Booked for<br>assessment with<br>clinician {int(clinic)}",
          'clinic': int(clinic)}
          for clinic in clinics]

        # Create a column of positions for people having an appointment with the clinic
        clinic_attends = [{'event': f'have_appointment_{int(clinic)}',
          'y':  950-(clinic+1)*80,
          'x': 725,
          'label': f"Attending appointment<br>with clinician {int(clinic)}"}
          for clinic in clinics]

        # Join these dataframes
        event_position_df = pd.concat([pd.DataFrame(clinic_waits),(pd.DataFrame(clinic_attends))])

        # Create a column of positions for people who are put on a waiting list before being given their future
        # appointment
        wait_for_booking = [{'event': 'waiting_appointment_to_be_scheduled',
          'y':  250,
          'x': 225,
          'label': f"Waiting to be<br>scheduled with <br>clinician "}]

        event_position_df = pd.concat([event_position_df,(pd.DataFrame(wait_for_booking))])

        # Create a column of positions for people being referred to another service (triaged as inappropriate
        # for this service after their initial referral and before an appointment is booked)
        referred_out = [{'event': 'referred_out',
          'y':  700,
          'x': 225,
          'label': f"Referred Out:<br>Unsuitable for Service"}]

        event_position_df = pd.concat([event_position_df,(pd.DataFrame(referred_out))])

        # Create a column of positions for people who have had their initial appointment and are now waiting for a
        # booked follow-up appointment to take place
        follow_up_waiting = [{'event': f'follow_up_appointment_booked_waiting_{int(clinic)}',
          'y':  950-(clinic+1)*80,
          'x': 1100,
          'label': f"On books - awaiting <br>next appointment<br>with clinician {int(clinic)}"}
          for clinic in clinics]

        event_position_df = pd.concat([event_position_df,(pd.DataFrame(follow_up_waiting))])

        # event_position_df = pd.concat([
        #     event_position_df,
        #     pd.DataFrame([{'event': 'exit', 'x':  270, 'y': 70, 'label': "Exit"}])]) .reset_index(drop=True)

        # clinic_lkup_df = pd.DataFrame([
        #     {'clinic': 0, 'icon': "🟠"},
        #     {'clinic': 1, 'icon': "🟡"},
        #     {'clinic': 2, 'icon': "🟢"},
        #     {'clinic': 3, 'icon': "🔵"},
        #     {'clinic': 4, 'icon': "🟣"},
        #     {'clinic': 5, 'icon': "🟤"},
        #     {'clinic': 6, 'icon': "⚫"},
        #     {'clinic': 7, 'icon': "⚪"},
        #     {'clinic': 8, 'icon': "🔶"},
        #     {'clinic': 9, 'icon': "🔷"},
        #     {'clinic': 10, 'icon': "🟩"}
        # ])



        # event_position_df = event_position_df.merge(clinic_lkup_df, how="left")
        # event_position_df["label"] = event_position_df.apply(lambda x: f"{x['label']} {x['icon']}" if pd.notna(x['icon']) else x['label'], axis=1)
        # event_position_df = event_position_df.drop(columns="icon")

        event_position_df = event_position_df.drop(columns="clinic")

        full_patient_df_plus_pos = generate_animation_df(
                            full_patient_df=full_patient_df,
                            event_position_df=event_position_df,
                            wrap_queues_at=15,
                            step_snapshot_max=30,
                            gap_between_entities=15,
                            gap_between_resources=15,
                            gap_between_rows=15,
                            debug_mode=True
                    )



        # def show_home_clinic(row):
        #     if "more" not in row["icon"]:
        #         if row["home_clinic"] == 0:
        #             return "🟠"
        #         if row["home_clinic"] == 1:
        #             return "🟡"
        #         if row["home_clinic"] == 2:
        #             return "🟢"
        #         if row["home_clinic"] == 3:
        #             return "🔵"
        #         if row["home_clinic"] == 4:
        #             return "🟣"
        #         if row["home_clinic"] == 5:
        #             return "🟤"
        #         if row["home_clinic"] == 6:
        #             return "⚫"
        #         if row["home_clinic"] == 7:
        #             return "⚪"
        #         if row["home_clinic"] == 8:
        #             return "🔶"
        #         if row["home_clinic"] == 9:
        #             return "🔷"
        #         if row["home_clinic"] == 10:
        #             return "🟩"
        #         else:
        #             return row["icon"]
        #     else:
        #         return row["icon"]

        # full_patient_df_plus_pos = full_patient_df_plus_pos.assign(icon=full_patient_df_plus_pos.apply(show_home_clinic, axis=1))


        def show_priority_icon(row):
            if "more" not in row["icon"]:
                if row["pathway"] == 2:
                        return "🚨"
                else:
                    return f"{row['icon']}"
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

        # full_patient_df_plus_pos = full_patient_df_plus_pos.assign(
        #     icon=full_patient_df_plus_pos.apply(add_los_to_icon, axis=1)
        #     )

        fig = generate_animation(
            full_patient_df_plus_pos=full_patient_df_plus_pos,
            event_position_df=event_position_df,
            scenario=None,
            plotly_height=1000,
            plotly_width=1200,
            override_x_max=1200,
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

        st.dataframe(event_log_df)

        # Average interval for low intensity and high intensity
        st.subheader("Are the intervals between appointments correct?")
        st.markdown("""
        Goal:

        LOW_INTENSITY_FOLLOW_UP_TARGET_INTERVAL = 14
        HIGH_INTENSITY_FOLLOW_UP_TARGET_INTERVAL = 7
        """)

        st.dataframe(
            event_log_df
            .dropna(subset='follow_up_intensity')
            .query('event_original == "have_appointment"')
            .groupby('follow_up_intensity')['interval']
            .describe()
            .T
        )
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
