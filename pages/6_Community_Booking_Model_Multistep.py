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
from output_animation_functions import reshape_for_animations, generate_animation_df, generate_animation
# from plotly.subplots import make_subplots
# from helper_functions import d2


st.set_page_config(layout="wide",
                   initial_sidebar_state="expanded",
                   page_title="Mental Health - Booking Model")

gc.collect()

st.title("Mental Health - Appointment Booking Model")

st.subheader("Weekly Slots")
st.markdown("Edit the number of daily slots available per clinician by clicking in the boxes below, or leave as the default schedule")
shifts = pd.read_csv("examples/ex_5_community_follow_up/data/shifts.csv")

number_of_clinicians = st.number_input("Number of Clinicians (caution: changing this will reset any changes you've made to shifts below)",
                min_value=1, max_value=11, value=11, step=1)
shifts_edited = st.data_editor(shifts.iloc[:,:number_of_clinicians])

CASELOAD_TARGET_MULTIPLIER = st.slider(label = "What factor should target caseload be adjusted by?",
                                       min_value=0.75, max_value=2.0, step=0.01, value=1.0)

# Adjust caseload target
st.markdown(
  """
  The default is to aim to have as many people on caseload as you have maximum theoretical slots.
  This can be adjusted up or down to see the impact of changing the policy.
  """
)

# Total caseload slots available
caseload_default_adjusted = pd.concat(
        [shifts_edited.sum(),
         shifts_edited.sum() * CASELOAD_TARGET_MULTIPLIER],
         axis=1
         )
caseload_default_adjusted.columns = ["Default Caseload (Total Slots)", "Adjusted Caseload"]
st.write(
  caseload_default_adjusted
)

st.write(f"Total caseload slots available: {round(shifts_edited.sum().sum() * CASELOAD_TARGET_MULTIPLIER,1)}")

annual_demand = st.slider("Select average annual demand", 100, 5000, 1200, 10)
prop_high_priority = st.slider("Select proportion of high priority patients (will go to front of booking queue)", 0.0, 0.9, 0.03, 0.01)
# prop_carve_out = st.slider("Select proportion of carve-out (slots reserved for high-priority patients)", 0.0, 0.9, 0.0, 0.01)
# Note - need to check if carve-out still works before reintegrating - it may be that changes to the way appointments are booked means that
# high-priority patients are no longer able to access them
# Will also need to update the creation of the scenario object to reintroduce it there

WARM_UP = st.slider(label = "How many days should the simulation warm-up for before collecting results?",
                               min_value=0, max_value=365*2,
                               step=5, value=365)


RESULTS_COLLECTION = st.slider(label = "How many days should results be collected for?",
                               min_value=100, max_value=365*5,
                               step=5, value=365*3)

RUN_LENGTH = RESULTS_COLLECTION + WARM_UP

button_run_pressed = st.button("Run simulation")

if button_run_pressed:

    # add a spinner and then display success box
    with st.spinner('Simulating the community booking system...'):
        #set up the scenario for the model to run.
        scenarios = {}

        caseload = pd.read_csv("examples/ex_5_community_follow_up/data/caseload.csv").iloc[:,:number_of_clinicians+1]
        pooling = pd.read_csv("examples/ex_5_community_follow_up/data/partial_pooling.csv").iloc[:number_of_clinicians,:number_of_clinicians+1]
        referrals = pd.read_csv("examples/ex_5_community_follow_up/data/referrals.csv").iloc[:number_of_clinicians]

        scenarios['pooled'] = Scenario(RUN_LENGTH,
                                       WARM_UP,
                                      #  prop_carve_out=prop_carve_out,
                                       seeds=generate_seed_vector(),
                                       slots_file=shifts_edited,
                                       pooling_file=pooling,
                                       existing_caseload_file=caseload,
                                       caseload_multiplier=CASELOAD_TARGET_MULTIPLIER,
                                       prop_high_priority=prop_high_priority,
                                       demand_file=referrals,
                                       annual_demand=annual_demand)

        results_all, results_low, results_high, event_log, bookings, available_slots, daily_caseload_snapshots = single_run(args = scenarios['pooled'])
        st.subheader("Clinic Simulation")

        event_log_df = pd.DataFrame(event_log)


        event_log_df['event_original'] = event_log_df['event']
        event_log_df['event'] = event_log_df.apply(
            lambda x: f"{x['event']}{f'_{int(x.booked_clinic)}'if pd.notna(x['booked_clinic']) and x['event'] != 'waiting_appointment_to_be_scheduled' else ''}",
            axis=1
            )

        full_patient_df = reshape_for_animations(event_log_df,
                                                 limit_duration=WARM_UP+RESULTS_COLLECTION,
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

        tab_summary, tab1,  tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
            ["Results Summary", "Animated Event Log", "Queue Sizes","Time From Referral to Booking",
             "Wait for Assessment", "Follow-up Appointment Stats", "Utilisation",
             "Caseload Sizes", "Full Event Log"]
        )

        with tab_summary:
            st.markdown("Placeholder")
            # Waiting list size over time (is it steady-state or out of control?)

            # Distribution of assessment waits by priority (do they meet target)

            # Distribution of inter-appointment waits (are they in tolerances)

            # Utilisation of appointment slots

            # Caseload sizes over time




        with tab4:
            st.markdown("### Wait for initial appointment")
            st.dataframe(
                results_summary(results_all, results_low, results_high)
                )

        with tab1:
            st.plotly_chart(fig)

        with tab8:
            st.dataframe(event_log_df)

        with tab5:
            # Average interval for low intensity and high intensity
            st.subheader("Are the intervals between appointments correct?")
            st.markdown("""
            Goal:

            LOW_INTENSITY_FOLLOW_UP_TARGET_INTERVAL = 14

            HIGH_INTENSITY_FOLLOW_UP_TARGET_INTERVAL = 7
            """)

            # Look at time from joining waiting list to booking

            # Look at average number of appointments (distribution)


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

#
            st.subheader("Number of follow-up appointments per client")
            st.markdown(
              """
              1 = initially triaged as low priority
              2 = initially triaged as high priority

              high = high-intensity follow-ups recommended after assessment (7 day interval)
              low = low-intensity follow-ups recommended after assessment (7 day interval)
              """
            )
            st.dataframe(
                event_log_df
                .dropna(subset='follow_ups_intended')
                .drop_duplicates(subset='patient')
                .groupby(['pathway','follow_up_intensity'])['follow_ups_intended']
                .describe()
                .T
            )

        # st.write(
        #     event_log_df
        #       .dropna(subset='follow_ups_intended')
        #       .drop_duplicates(subset='patient')[['pathway','follow_ups_intended']]
        #       .value_counts()
        # )

            st.plotly_chart(
                px.bar(
                event_log_df
                  .dropna(subset='follow_ups_intended')
                  .drop_duplicates(subset='patient')[['pathway','follow_ups_intended']]
                  .value_counts()
                  .reset_index(drop=False),
                x="follow_ups_intended", y="count",facet_row="pathway"
                )
            )

        # st.plotly_chart(
        #     px.bar(
        #     event_log_df
        #       .dropna(subset='follow_ups_intended')
        #       .drop_duplicates(subset='patient'),
        #       x=,
        #       y=
        # )
        # )

        with tab3:
            st.subheader("Time from referral to appointment booking")

            st.write(
                event_log_df
                .dropna(subset='assessment_booking_wait')
                .drop_duplicates(subset='patient')
                .groupby('pathway')['assessment_booking_wait']
                .describe()
                .T
            )

        # st.write(
        #     event_log_df
        #     .dropna(subset='assessment_booking_wait')
        #     .drop_duplicates(subset='patient')
        #     .groupby('pathway')[['pathway','assessment_booking_wait']]
        #     .value_counts()
        # )

            st.plotly_chart(
                px.bar(
                event_log_df
                .dropna(subset='assessment_booking_wait')
                .drop_duplicates(subset='patient')
                .groupby('pathway')[['pathway','assessment_booking_wait']]
                .value_counts()
                .reset_index(drop=False),
                x="assessment_booking_wait", y="count", facet_row="pathway"
                )
            )

        # st.subheader("Bookings")

        # st.write(bookings.iloc[WARM_UP:RUN_LENGTH,])

        # st.subheader("Remaining Slots")

        # st.write(available_slots.iloc[WARM_UP:RUN_LENGTH,])

        with tab6:
            st.subheader("Slot Utilisation - Slots Remaining")

            st.write(
                ((bookings.iloc[WARM_UP:RUN_LENGTH,]).sum() /
                ((bookings.iloc[WARM_UP:RUN_LENGTH,]) + available_slots.iloc[WARM_UP:RUN_LENGTH,]).sum()).T
                    )

        with tab7:


            st.subheader("Daily Caseload Snapshots")

            cl = pd.DataFrame(daily_caseload_snapshots["caseload_day_end"].tolist())
            cl_filtered = cl.iloc[WARM_UP:RUN_LENGTH,:]

            st.write(cl_filtered)

            st.plotly_chart(
                px.line(
                cl_filtered.reset_index(drop=False).melt(id_vars=["index"], var_name="clinician", value_name="caseload"),
                x="index",
                y= "caseload",
                color="clinician"
                )
            )

        with tab2:
            st.subheader("Waiting List over Time")

            daily_position_counts = []

            for day in range(RUN_LENGTH):
                # First limit to anyone who hasn't left the system yet
                departed = event_log_df[
                    (event_log_df["time"] <= day) &
                    (event_log_df["event"] == "depart")]["patient"].tolist()
                # Filter down to events that have occurred at or before this day
                upto_now = event_log_df[(event_log_df["time"] <= day)
                                        & (event_log_df["event"] != "arrival")
                                        & (~event_log_df["patient"].isin(departed))]
                # Now take the latest event for each person
                latest_event_upto_now = upto_now.sort_values("time").groupby("patient").tail(1)
                for event_type in event_log_df["event_original"].unique():
                    snapshot_count = len(latest_event_upto_now[(latest_event_upto_now["event_original"] == event_type)])
                    daily_position_counts.append(
                        {"day": day,
                        "event": event_type,
                        "count": snapshot_count}
                    )

            daily_position_counts = pd.DataFrame(daily_position_counts)

            st.plotly_chart(
                px.bar(
                    daily_position_counts[daily_position_counts["event"] != "depart"],
                    x="event",
                    y="count",
                    animation_frame="day",
                    range_y=[0, max(daily_position_counts["count"])]
                ),
                use_container_width=True
            )

            st.subheader("Waiting list sizes over time")
            #TODO: Add box indicating warm-up period
            st.plotly_chart(
                px.line(
                    daily_position_counts[(daily_position_counts["event"] == "waiting_appointment_to_be_scheduled") |
                                          (daily_position_counts["event"] == "appointment_booked_waiting") |
                                          (daily_position_counts["event"] == "follow_up_appointment_booked_waiting")],
                  x="day",
                  y="count",
                  color="event"
                ),
                use_container_width=True

            )

            st.subheader("Balance between people arriving in the system and departing")

            st.write(event_log_df[(event_log_df["event"] == "arrival") |
                                  (event_log_df["event"] == "depart")][["time", "event"]].value_counts().sort_values('time'))

            st.plotly_chart(
                px.line(
                    event_log_df[(event_log_df["event"] == "arrival") |
                                  (event_log_df["event"] == "depart")][["time", "event"]].value_counts().reset_index(drop=False).sort_values('time'),
                  x="time",
                  y="count",
                  color="event"
                ),
                use_container_width=True

            )


        # st.plotly_chart(
        #     px.line(
        #     event_log_df
        #       .dropna(subset='follow_ups_intended')
        #       .drop_duplicates(subset='patient')[['pathway','follow_ups_intended']]
        #       .value_counts()
        #       .reset_index(drop=False),
        #     x="follow_ups_intended", y="count",facet_row="pathway"
        #     )
        # )
