import gc
import math
import random
# Package for webpage development
import streamlit as st
# Packages for data manipulation
import numpy as np
import pandas as pd
# Packages for graphing
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Model functions
from examples.ex_5_community_follow_up.model_classes import Scenario, generate_seed_vector
from examples.ex_5_community_follow_up.simulation_execution_functions import single_run
from examples.ex_5_community_follow_up.simulation_summary_functions import results_summary
# Animation functions
from vidigi.prep import reshape_for_animations, generate_animation_df
from vidigi.animation import generate_animation
# Flowchart display function
from helper_functions import mermaid

st.set_page_config(layout="wide",
                   initial_sidebar_state="expanded",
                   page_title="Mental Health - Booking Model")

gc.collect()

st.title("Community Service - Repeat Appointment Booking Model with Variable Follow-ups")

st.markdown(
  """
  This model is designed to mimic a simple community-based appointment service where clients have an initial appointment and then a variable number of follow-ups over an extended period of time.

  A client can have their first referral with any clinician - in practice, whoever has capacity and the soonest appointment - but all follow-on appointments will be with the same clinician.

  Instead of using simpy resources, an appointment book is set up. The model looks for an appointment that meets criteria, then books this in, reducing the available slots as appropriate.
  This allows for continuity of care in a way that is more difficult to achieve with a simpy resource, as well as allowing finer control over the number of appointments a clinician can undertake in a day.

  The process a patient arriving at this clinic will follow is shown below:
  """
)

mermaid(height=450, width=1250, code="""
%%{ init: {  'theme': 'base', 'themeVariables': {'lineColor': '#b4b4b4', background: '#636363'} } }%%
flowchart LR
    A[Arrival\nGenerated] -->B((<b>SINK</b>\nDischarged as\ninappropriate for\nservice))
    A --> B2[Arrival\nAccepted]
    B2 --> C{Triaged as\nHigh Priority}
    B2 --> D{Triaged as\nLow Priority}
    C --> |Go to front of queue\nbut behind other\nhigh priority patients\nwho arrived first| E[In Queue\nfor Booking\n\nWaiting for\nClinician With\nCapacity]
    D --> |Go to back of queue |E
    E --> |Caseload slot\nbecomes available\nwhen another client\nleaves caseload| F(Patient at front of queue booked\nin for soonest available\nappointment\n\nIf more than one caseload\nslot opens up, book next\npatients in queue too)
    F --> |Wait until\nday of\nAssessment\nAppointment|G[Have\nAssessment\nAppointment]
    G --> H((<b>SINK</b>\nDischarged as\nno follow-ups\nrequired))
    G --> I{Requires\nHigh Intensity\nFollow-up\n\nWeekly\n1 caseload\nslot}
    G --> J{Requires\nLow Intensity\nFollow-up\n\nFortnightly\n0.5 caseload\nslots}
    J --> K
    I --> K[Attend Follow-up Appointments]
    K --> K
    K --> L((<b>SINK</b>\nDepart After\nAll Follow-ups\nComplete))
      """)


with st.expander("Click here for additional details about this model"):

    st.markdown(
        """
        There are a range of ways this model could be adapted further:
          - The logic/criteria used to determine whether patients are booked in or held on a waiting list prior to booking could be changed
          - The number of priorities or potential ongoing intensities could be altered
          - It could be incorporated into a larger model where different resources - i.e. different sets of appointment books - are involved in different steps
          - It could be scaled up to look at multiple clinics, including a step where patients are potentially routed to a different clinic depending on availability
          - Different appointment types could take up different numbers of slots (e.g. an assessment appointment could be 1.5 slots while a regular appointment is 1 slot, or telephone appointments could take up 1 slot while in-person appointmetns take 3)
          - A preference for weekend vs weekday appointments could be introduced for certain clients
          - The distinction between high and low itensity appointment frequencies could be removed, with all clients being seen at a standard frequency
          - The interval between appointments could be randomised to some extent
          - The interval between appointments could be set to increase over time
          - The number of appointments could be set to a fixed amount, or capped at a maximum, model a system where only a fixed number of appointments are offered as standard
          (e.g. in first-line IAPT pyschological wellbeing services in the UK, only a certain number of appointments are offered at tier two before users are either discharged
          or escalated to tier three support if clinically appropriate)
        """
    )

st.warning("Note that at present this only runs a single replication of the simulation model. Multiple replications should be run with different random seeds to ensure a good picture of potential variability is created.")

st.error("Note that some issues seem to be present with the caseload calculations, leading to some unexpected behaviour within the model over time.")

st.divider()

st.subheader("Set up Model Parameters")

shifts = pd.read_csv("examples/ex_5_community_follow_up/data/shifts.csv")

number_of_clinicians = st.number_input("Number of Clinicians (caution: changing this will reset any changes you've made to shifts below)",
                min_value=1, max_value=20, value=8, step=1)

set_weekly_slots = st.toggle("Toggle this to set clinician slots per week instead of per day")


if not set_weekly_slots:
    st.subheader("Clinician Appointment Book - Set Slots Available Per Day")
    st.markdown("Edit the number of daily slots available per clinician by clicking in the boxes below, or leave as the default schedule")

    shifts_edited = shifts.copy()
    shifts_edited.index = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    shifts_edited = st.data_editor(
        shifts_edited.iloc[:,:number_of_clinicians]
        )

if set_weekly_slots:
    st.subheader("Clinician Appointment Book - Set Slots Available Per Week")
    st.markdown("""
        Note that behind the scenes, the code will randomly distribute each clinician's appointments across different days of the week, giving them a maximum of 5 working days
        """
        )

    shifts_daily = pd.DataFrame(shifts.iloc[:,:number_of_clinicians].sum()).reset_index()
    # Add column to help with pivoting
    shifts_daily['reindex'] = 0
    shifts_daily = (shifts_daily
         .pivot(index='reindex', columns="index", values=0)
         .reset_index(drop=True))

    shifts_edited_daily = st.data_editor(
        shifts_daily
        )
    # Ensure we have a dataframe and not a series
    shifts_edited_daily = pd.DataFrame(shifts_edited_daily)

    # Approach taken from https://stackoverflow.com/questions/54353083/distribute-an-integer-amount-by-a-set-of-slots-as-evenly-as-possible
    def distribute_shifts(shifts, days):
        base, extra = divmod(shifts, days)
        return [base + (i < extra) for i in range(days)]

    all_clinican_shifts = []

    for clinician in range(number_of_clinicians):
        # Get the total number of slots for this clinician from the dataframe
        # st.write(shifts_edited_daily.columns.tolist())
        # st.write(clinician)
        clinician_id = shifts_edited_daily.columns[clinician]
        clinician_total_shifts = shifts_edited_daily.iloc[:,clinician].values[0]
        # Distribute the number of shifts evenly across a list of length 7
        # To begin with, distribute across a number of days in the week - let's go for either 4 or 5
        clinician_shifts = distribute_shifts(clinician_total_shifts, random.randint(4, 5))
        # Now randomly place days off amongst this to ensure that the final length is 7
        days_to_pad = 7-len(clinician_shifts)

        for i in range(days_to_pad):
            position = random.randint(0, 6)
            clinician_shifts.insert(position, 0)

        all_clinican_shifts.append(clinician_shifts)

    shifts_edited = pd.DataFrame(all_clinican_shifts).T.copy()

    shifts_edited.columns = (shifts.iloc[:,:number_of_clinicians]).columns
    shifts_edited.index = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                           "Saturday", "Sunday"]

    with st.expander("Click here to view the randomly distributed shifts that the simulation will use"):
        st.write(shifts_edited)

# Change index back to 0 to 6 for further steps
shifts_edited.index = [0,1,2,3,4,5,6]


# Total caseload slots available
with st.expander("Click here to adjust caseload targets"):
    CASELOAD_TARGET_MULTIPLIER = st.slider(
        label = "What factor should target caseload be adjusted by?",
        min_value=0.5, max_value=4.0, step=0.01, value=1.3
        )

    # Adjust caseload target
    st.markdown(
      """
      The default is to aim to have as many people on caseload as you have maximum theoretical slots.
      This can be adjusted up or down to see the impact of changing the policy.

      Note that low intensity patients in this model take up 0.5 slots. High intensity patients take up 1 slot.
      """
    )

    caseload_default_adjusted = pd.concat(
            [shifts_edited.sum(),
            np.floor(shifts_edited.sum() * CASELOAD_TARGET_MULTIPLIER)],
            axis=1
            )
    caseload_default_adjusted.columns = ["Default Caseload (Total Slots Per Week)",
                                         "Adjusted Caseload"]
    st.write(
      caseload_default_adjusted
    )

st.divider()

st.write(f"Total appointment slots available per week: {np.floor(shifts_edited.sum()).sum()}")
st.write(f"Total caseload slots available after adjustment by multiplier ({CASELOAD_TARGET_MULTIPLIER}):" \
         f"{np.floor(shifts_edited.sum() * CASELOAD_TARGET_MULTIPLIER).sum()}")

col_setup_1, col_setup_2 = st.columns(2)
with col_setup_1:
    ANNUAL_DEMAND= st.slider("Select average annual demand", 100, 5000, 700, 10)
with col_setup_2:
    PROP_HIGH_PRIORITY = st.slider(
        "Select proportion of high priority patients (will go to front of booking queue)",
        0.0, 0.9, 0.03, 0.01
        )

# prop_carve_out = st.slider("Select proportion of carve-out (slots reserved for high-priority patients)", 0.0, 0.9, 0.0, 0.01)
# Note - need to check if carve-out still works before reintegrating - it may be that changes to the way appointments are booked means that
# high-priority patients are no longer able to access them
# Will also need to update the creation of the scenario object to reintroduce it there

col_setup_3, col_setup_4 = st.columns(2)
with col_setup_3:
    WARM_UP = st.slider(label = "How many days should the simulation warm-up for before collecting results?",
                                  min_value=0, max_value=365*2,
                                  step=5, value=365)

with col_setup_4:
    RESULTS_COLLECTION = st.slider(label = "How many days should results be collected for?",
                                  min_value=100, max_value=365*5,
                                  step=5, value=365*3)

RUN_LENGTH = RESULTS_COLLECTION + WARM_UP

col_setup_5, col_setup_6 = st.columns(2)

with col_setup_5:
    PROP_REFERRED_OUT = st.slider("Set Proportion of Arrivals Referred Out as Unsuitable", 0.0, 0.99, 0.12, 0.01)

with col_setup_6:
    SEED = st.slider("Set Seed", 0, 1000, 42, 1)

with st.expander("Set follow-up appointment settings:"):
    col_setup_8, col_setup_9, col_setup_10, col_setup_11 = st.columns(4)

    with col_setup_8:
        PROP_HIGH_PRIORITY_ONGOING_APPOINTMENTS = st.slider(
            label = "What proportion of high-priority patients have ongoing appointments after assessment?",
            min_value=0.0, max_value=1.0,
            step=0.01, value=0.95)
        PROP_LOW_PRIORITY_ONGOING_APPOINTMENTS = st.slider(
            label = "What proportion of low-priority patients have ongoing appointments after assessment?",
            min_value=0.0, max_value=1.0,
            step=0.01, value=0.8)

    with col_setup_9:
        # What proportion of people initially graded as *high*
        # priority go on to have high intensity therapy?
        PROP_HIGH_PRIORITY_HIGH_INTENSITY = st.slider(
                label = "What proportion of people initially graded as *high* priority go on to have high intensity therapy?",
                min_value=0.0, max_value=1.0,
                step=0.01, value=0.7)
        # What proportion of people initially graded as *low*
        # priority go on to have high intensity therapy?
        PROP_LOW_PRIORITY_HIGH_INTENSITY = st.slider(
                label = "What proportion of people initially graded as *low* priority go on to have high intensity therapy?",
                min_value=0.0, max_value=1.0,
                step=0.01, value=0.2)

        st.markdown("Note that the remainder of patients in both cases will go on to have low-intensity therapy")

    with col_setup_10:
        MEAN_FOLLOW_UPS_HIGH_INTENSITY = st.slider(label = "How many follow-ups do high-intensity patients have on average?",
                                      min_value=1, max_value=50,
                                      step=5, value=10)
        MEAN_FOLLOW_UPS_LOW_INTENSITY = st.slider(label = "How many follow-ups do low-intensity patients have on average?",
                                      min_value=1, max_value=50,
                                      step=5, value=6)


    with col_setup_11:
        SD_FOLLOW_UPS_HIGH_INTENSITY = st.slider(label = "How many follow-ups do high-intensity patients have on average?",
                                      min_value=1, max_value=50,
                                      step=5, value=18)
        st.caption(f"95% of high intensity patients have between {MEAN_FOLLOW_UPS_HIGH_INTENSITY-SD_FOLLOW_UPS_HIGH_INTENSITY if (MEAN_FOLLOW_UPS_HIGH_INTENSITY-SD_FOLLOW_UPS_HIGH_INTENSITY) >= 0 else 0} and {MEAN_FOLLOW_UPS_HIGH_INTENSITY+SD_FOLLOW_UPS_HIGH_INTENSITY} follow-up appointments")
        SD_FOLLOW_UPS_HIGH_INTENSITY = SD_FOLLOW_UPS_HIGH_INTENSITY/3
        SD_FOLLOW_UPS_LOW_INTENSITY = st.slider(label = "How much do the number of appointments for low intensity patients tend to vary by?",
                                      min_value=1, max_value=50,
                                      step=5, value=9)
        st.caption(f"95% of high intensity patients have between {MEAN_FOLLOW_UPS_LOW_INTENSITY-SD_FOLLOW_UPS_LOW_INTENSITY if (MEAN_FOLLOW_UPS_LOW_INTENSITY-SD_FOLLOW_UPS_LOW_INTENSITY) >= 0 else 0} and {MEAN_FOLLOW_UPS_LOW_INTENSITY+SD_FOLLOW_UPS_LOW_INTENSITY} follow-up appointments")
        SD_FOLLOW_UPS_LOW_INTENSITY = SD_FOLLOW_UPS_LOW_INTENSITY/3

# These calculations don't appear to be quite right, or don't appear to be that useful at the moment. Need to further consider how it interacts with the 'caseload'.
# slots_required_calc =  \
#   # High priority patients who go on to have high intensity follow-ups
#   ((ANNUAL_DEMAND * (1-PROP_REFERRED_OUT)) * (PROP_HIGH_PRIORITY * PROP_HIGH_PRIORITY_ONGOING_APPOINTMENTS * PROP_HIGH_PRIORITY_HIGH_INTENSITY) * MEAN_FOLLOW_UPS_HIGH_INTENSITY) + \
#   # High priority patients who go on to have low intensity follow-ups
#   ((ANNUAL_DEMAND * (1-PROP_REFERRED_OUT)) * (PROP_HIGH_PRIORITY * PROP_HIGH_PRIORITY_ONGOING_APPOINTMENTS * (1-PROP_HIGH_PRIORITY_HIGH_INTENSITY)) * MEAN_FOLLOW_UPS_LOW_INTENSITY) + \
#   # Low priority patients who go on to have high intensity follow-ups
#   ((ANNUAL_DEMAND * (1-PROP_REFERRED_OUT)) * ((1-PROP_HIGH_PRIORITY) * PROP_LOW_PRIORITY_ONGOING_APPOINTMENTS * PROP_LOW_PRIORITY_HIGH_INTENSITY) * MEAN_FOLLOW_UPS_HIGH_INTENSITY) + \
#   # Low priority patients who go on to have low intensity follow-ups
#   ((ANNUAL_DEMAND * (1-PROP_REFERRED_OUT)) * ((1-PROP_HIGH_PRIORITY) * PROP_LOW_PRIORITY_ONGOING_APPOINTMENTS * (1-PROP_HIGH_PRIORITY_HIGH_INTENSITY)) * MEAN_FOLLOW_UPS_LOW_INTENSITY)

# st.markdown(
#     f"Approximately {round(slots_required_calc,0):.0f} slots required to meet an annual demand of {ANNUAL_DEMAND} with currently set parameters. {(shifts_edited.sum().sum())*52} appointments are available per year."
# )

button_run_pressed = st.button("Run simulation")

if button_run_pressed:

    # add a spinner and then display success box
    with st.spinner('Simulating the community booking system...'):
        #set up the scenario for the model to run.
        scenarios = {}

        caseload = (pd.read_csv("examples/ex_5_community_follow_up/data/caseload.csv")
                   .iloc[:,:number_of_clinicians+1])
        pooling = (pd.read_csv("examples/ex_5_community_follow_up/data/partial_pooling.csv")
                   .iloc[:number_of_clinicians,:number_of_clinicians+1])
        referrals = (pd.read_csv("examples/ex_5_community_follow_up/data/referrals.csv")
                     .iloc[:number_of_clinicians])

        scenarios['pooled'] = Scenario(RUN_LENGTH,
                                       WARM_UP,
                                      #  prop_carve_out=prop_carve_out,
                                       seeds=generate_seed_vector(SEED),
                                       slots_file=shifts_edited,
                                       pooling_file=pooling,
                                       existing_caseload_file=caseload,
                                       demand_file=referrals,
                                       caseload_multiplier=CASELOAD_TARGET_MULTIPLIER,
                                       prop_high_priority=PROP_HIGH_PRIORITY,
                                       prop_high_priority_ongoing_appointments=PROP_HIGH_PRIORITY_ONGOING_APPOINTMENTS,
                                       prop_low_priority_ongoing_appointments=PROP_LOW_PRIORITY_ONGOING_APPOINTMENTS,
                                       prop_high_priority_assessed_high_intensity=PROP_HIGH_PRIORITY_HIGH_INTENSITY,
                                       prop_low_priority_assessed_high_intensity=PROP_LOW_PRIORITY_HIGH_INTENSITY,
                                       mean_follow_ups_high_intensity=MEAN_FOLLOW_UPS_HIGH_INTENSITY,
                                       sd_follow_ups_high_intensity=SD_FOLLOW_UPS_HIGH_INTENSITY,
                                       mean_follow_ups_low_intensity=MEAN_FOLLOW_UPS_LOW_INTENSITY,
                                       sd_follow_ups_low_intensity=SD_FOLLOW_UPS_LOW_INTENSITY,
                                       annual_demand=ANNUAL_DEMAND,
                                       prop_referred_out=PROP_REFERRED_OUT)

        # Run the model and unpack the outputs
        results_all, results_low, results_high, event_log, \
        bookings, available_slots, daily_caseload_snapshots, \
        daily_waiting_for_booking_snapshots, \
        daily_arrivals = single_run(args = scenarios['pooled'])

        st.subheader("Clinic Simulation")

        event_log_df = pd.DataFrame(event_log)


        event_log_df['event_original'] = event_log_df['event']
        event_log_df['event'] = event_log_df.apply(
            lambda x: f"{x['event']}{f'_{int(x.booked_clinic)}'if pd.notna(x['booked_clinic']) and x['event'] != 'waiting_appointment_to_be_scheduled' else ''}",
            axis=1
            )

        full_patient_df = reshape_for_animations(event_log_df,
                                                 limit_duration=WARM_UP+RESULTS_COLLECTION,
                                                 entity_col_name="patient",
                                                 every_x_time_units=1,
                                                 step_snapshot_max=30)

        # Remove the warm-up period from the event log
        full_patient_df = full_patient_df[full_patient_df["snapshot_time"] >= WARM_UP]

        #####################################################
        # Create the positioning dataframe for the animation
        #####################################################

        # Create a list of clinics
        clinics =  [x for x
                    in event_log_df['booked_clinic'].sort_values().unique().tolist()
                    if not math.isnan(x)]

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
        event_position_df = pd.concat(
            [pd.DataFrame(clinic_waits),
             (pd.DataFrame(clinic_attends))
             ])

        # Create a column of positions for people who are put on a waiting list before being given their future
        # appointment
        wait_for_booking = [{
          'event': 'waiting_appointment_to_be_scheduled',
          'y':  250,
          'x': 225,
          'label': "Waiting to be<br>scheduled with <br>clinician "
          }]

        event_position_df = pd.concat([event_position_df,(pd.DataFrame(wait_for_booking))])

        # Create a column of positions for people being referred to another service (triaged as inappropriate
        # for this service after their initial referral and before an appointment is booked)
        referred_out = [{
          'event': 'referred_out',
          'y':  700,
          'x': 225,
          'label': "Referred Out:<br>Unsuitable for Service"
          }]

        event_position_df = pd.concat([event_position_df,(pd.DataFrame(referred_out))])

        # Create a column of positions for people who have had their initial appointment and are now waiting for a
        # booked follow-up appointment to take place
        follow_up_waiting = [{
          'event': f'follow_up_appointment_booked_waiting_{int(clinic)}',
          'y':  950-(clinic+1)*80,
          'x': 1100,
          'label': f"On books - awaiting <br>next appointment<br>with clinician {int(clinic)}"
          } for clinic in clinics]

        event_position_df = pd.concat([event_position_df,(pd.DataFrame(follow_up_waiting))])

        event_position_df = event_position_df.drop(columns="clinic")

        full_patient_df_plus_pos = generate_animation_df(
                            full_entity_df=full_patient_df,
                            event_position_df=event_position_df,
                            entity_col_name="patient",
                            wrap_queues_at=15,
                            step_snapshot_max=30,
                            gap_between_entities=15,
                            gap_between_resources=15,
                            gap_between_queue_rows=15,
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
            full_entity_df_plus_pos=full_patient_df_plus_pos,
            event_position_df=event_position_df,
            scenario=None,
            entity_col_name="patient",
            plotly_height=1000,
            plotly_width=1200,
            override_x_max=1200,
            override_y_max=1000,
            text_size=10,
            entity_icon_size=10,
            include_play_button=True,
            add_background_image=None,
            display_stage_labels=True,
            time_display_units="d",
            simulation_time_unit="days",
            start_date="2022-06-27",
            setup_mode=False,
            frame_duration=1500, #milliseconds
            frame_transition_duration=1000, #milliseconds
            debug_mode=False
        )

        daily_position_counts = []

        for day in range(RUN_LENGTH):
            # First limit to anyone who hasn't left the system yet
            # Get a list of all people who have departed on or before the day
            # of interest as we can then remove them from the dataframe
            # at the next step
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

        tab_summary, tab1,  tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
            ["Results Summary", "Animated Event Log", "Queue Sizes","Time From Referral to Booking",
             "Wait for Assessment", "Follow-up Appointment Stats", "Utilisation",
             "Caseload Sizes", "Full Event Log"]
        )

        with tab_summary:
            col1, col2 = st.columns(2)
            # Waiting list size over time (is it steady-state or out of control?)
            with col1:
                st.subheader("Waiting lists over time")

                st.markdown("""
                            This looks at the number of people at each point in the system over the simulation's duration.
                            """)
                fig_daily_position_counts = px.line(daily_position_counts[(daily_position_counts["event"] == "waiting_appointment_to_be_scheduled") |
                                          (daily_position_counts["event"] == "appointment_booked_waiting") |
                                          (daily_position_counts["event"] == "follow_up_appointment_booked_waiting")  |
                                          (daily_position_counts["event"] == "have_appointment")],
                  x="day",
                  y="count",
                  color="event"
                )
                fig_daily_position_counts.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ))

                st.plotly_chart(
                    fig_daily_position_counts,
                    use_container_width=True
                )

                st.subheader("Balance between arrivals and departures")
                arrival_depart_df = event_log_df[(event_log_df["event"] == "arrival") |
                                      (event_log_df["event"] == "depart")][["time", "event"]].value_counts().reset_index(drop=False).sort_values('time')

                arrival_depart_df_pivot = arrival_depart_df.pivot(index="time", columns="event", values="count")
                arrival_depart_df_pivot["difference (arrival-depart) - positive is more more arriving than departing"] = arrival_depart_df_pivot["arrival"] - arrival_depart_df_pivot["depart"]

                arrival_depart_balance_fig = px.scatter(
                      arrival_depart_df,
                      x="time",
                      y="count",
                      color="event",
                      trendline="rolling",
                      color_discrete_sequence=['#636EFA', '#EF553B'],
                      opacity=0.1,
                      trendline_options=dict(window=100)
                    )

                st.plotly_chart(arrival_depart_balance_fig, use_container_width=True)


            # Time from referral to booking
            with col2:
                st.subheader("Booking Waits")

                st.markdown("""
                            This looks at the time from arriving in the system to having an appointment booked,
                            which will only happen when a clinician has a low enough caseload to take on a new patient
                            """)

                assessment_booking_waits = (event_log_df
                      .dropna(subset='assessment_booking_wait')
                      .drop_duplicates(subset='patient')
                      [['time','pathway', 'assessment_booking_wait']]
                      )

                st.write(f"Average wait for booking (high priority):" \
                         f" {round(np.mean(assessment_booking_waits[assessment_booking_waits['pathway'] == 2]['assessment_booking_wait']),1)}" \
                         f" (Target: 7 days, Longest " \
                         f" {round(max(assessment_booking_waits[assessment_booking_waits['pathway'] == 2]['assessment_booking_wait']),1)} days)")

                st.write(f"Average wait for booking (low priority):" \
                         f" {round(np.mean(assessment_booking_waits[assessment_booking_waits['pathway'] == 1]['assessment_booking_wait']),1)}" \
                         f" (Target: 14 days, Longest" \
                         f" {round(max(assessment_booking_waits[assessment_booking_waits['pathway'] == 1]['assessment_booking_wait']),1)} days)")

                st.plotly_chart(
                    px.box(
                      assessment_booking_waits,
                      y="assessment_booking_wait", x="pathway", color="pathway"
                          ), use_container_width=True
                )

                st.subheader("Average Booking Waits by Pathway Over Time")

                st.plotly_chart(
                    px.line(
                      assessment_booking_waits,
                      y="assessment_booking_wait", x="time", color="pathway", line_group="pathway"
                    ), use_container_width=True
                )

            # Distribution of assessment waits by priority (do they meet target)
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Assessment Waits")
                st.markdown("""
                            This looks at the time from arriving in the system to having an assessment (first appointment).
                            This is the time from arrival to booking + the time from booking to appointment.
                            """)

                # print(results_high)
                # print(results_low)

                if results_high:
                    st.write(f"Average wait for assessment (high priority):" \
                            f" {round(np.mean(results_high),1)} (Target: 7 days, Longest {round(max(results_high),1)} days)")

                if results_low:
                    st.write(f"Average wait for assessment (high priority):" \
                            f" {round(np.mean(results_low),1)} (Target: 14 days, Longest {round(max(results_low),1)} days)")

                st.plotly_chart(
                    px.box(
                event_log_df
                .dropna(subset='wait')
                .drop_duplicates(subset='patient')[['pathway', 'wait']],
                y="wait", x="pathway", color="pathway"
                    ), use_container_width=True
                    )
                st.subheader("Average Assessment Waits by Pathway Over Time")
                st.plotly_chart(
                    px.line(
                      event_log_df
                      .dropna(subset='wait')
                      .drop_duplicates(subset='patient')[['time','pathway', 'wait']],
                      y="wait", x="time", color="pathway", line_group="pathway"
                    ), use_container_width=True
                    )
            # Distribution of inter-appointment waits (are they in tolerances)
            with col4:
                st.subheader("Inter-appointment waits")
                st.markdown("""
                            This looks at the time between repeat appointments after the assessment appointment
                            has taken place. Ideally intervals should remain consistent and around the target.
                            Intervals that are much longer than the target indicate that the system is overloaded
                            (clinician's appointment books are filling up too much in the short term) -
                            too many clients are on caseload at once to be sustainable.
                            """)

                inter_appointment_gaps = (event_log_df
                .dropna(subset='interval')
                .drop_duplicates('patient')
                # .query('event_original == "have_appointment"')
                [['time', 'follow_up_intensity','interval']]
                )

                # st.write(inter_appointment_gaps)

                st.write(f"Average appointment interval (high priority): {round(np.mean(inter_appointment_gaps[inter_appointment_gaps['follow_up_intensity'] == 'high']['interval']),1)} (Target: 7 days, Longest {round(max(inter_appointment_gaps[inter_appointment_gaps['follow_up_intensity'] == 'high']['interval']),1)} days)")
                st.write(f"Average appointment interval (high priority): {round(np.mean(inter_appointment_gaps[inter_appointment_gaps['follow_up_intensity'] == 'low']['interval']),1)} (Target: 14 days, Longest {round(max(inter_appointment_gaps[inter_appointment_gaps['follow_up_intensity'] == 'low']['interval']),1)} days)")

                st.plotly_chart(
                    px.box(
                      inter_appointment_gaps,
                      y="interval", x="follow_up_intensity", color="follow_up_intensity"
                          ), use_container_width=True
                )

                st.markdown("""
                            This looks at whether the inter-appointment arrival has varied significantly across the simulation's duration.
                            We would expect to see some constant variation, but large peaks at certain points indicate the system may not have
                            enough spare capacity to consistently deal with peaks in demand/new arrivals.
                            """)

                st.plotly_chart(
                    px.line(
                      inter_appointment_gaps,
                      y="interval", x="time", color="follow_up_intensity", line_group="follow_up_intensity"
                    ), use_container_width=True
                )

            # Utilisation of appointment slots
            col5, col6 = st.columns(2)
            with col5:
                st.subheader("Slot Utilisation - % of Slots Used")
                st.markdown("""
                            This looks at the % of slots that clincian's have that end up having a booking.
                            This does not include data from the warm-up period.
                            """)
                st.write("**Total % of slots used:** {}%".format(
                   round(
                        (((bookings.iloc[WARM_UP:RUN_LENGTH,]).sum().sum() /
                          ((bookings.iloc[WARM_UP:RUN_LENGTH,]) + available_slots.iloc[WARM_UP:RUN_LENGTH,]).sum().sum())
                          )*100,
                          1
                        )
                      )
                )


                st.write(
                    round(
                        (((bookings.iloc[WARM_UP:RUN_LENGTH,]).sum() /
                          ((bookings.iloc[WARM_UP:RUN_LENGTH,]) + available_slots.iloc[WARM_UP:RUN_LENGTH,]).sum()).T
                          )*100,
                          1
                        )
                      )



            # Caseload sizes over time
            with col6:
                st.subheader("Daily Arrivals (Including Rejected)")
                st.markdown("""
                            This looks at the number of arrivals to the system per day, including those who were rejected
                            before reaching the assessment booking stage as they were triaged as being inappropriate for
                            the service. This can be useful to determine if the random seed for this simulation has resulted
                            in any particular large peaks or troughs in the number of arrivals that may have tested the
                            resilience of the clinic.
                            """)
                # st.write(daily_arrivals)
                # Want two trendlines on the same fig
                # Using answer from
                # https://community.plotly.com/t/displaying-2-trendlines-for-1-set-of-data-with-plotly/68972/2
                fig_arrivals = go.Figure(make_subplots(rows=1, cols=1))
                fig_arrivals_1 = px.scatter(
                        pd.DataFrame(pd.Series(daily_arrivals).value_counts()).reset_index(drop=False),
                        x="index",
                        y="count",
                        trendline="rolling",
                        opacity=0.4,
                        trendline_options=dict(window=7)#,
                    )
                fig_arrivals_2 = px.scatter(
                        pd.DataFrame(pd.Series(daily_arrivals).value_counts()).reset_index(drop=False),
                        x="index",
                        y="count",
                        trendline="rolling",
                        trendline_options=dict(window=60),
                        color_discrete_sequence=['red']
                    )
                fig_arrivals_2.data = [t for t in fig_arrivals_2.data if t.mode == "lines"]
                fig_trace = []

                for trace in range(len(fig_arrivals_1["data"])):
                    fig_trace.append(fig_arrivals_1["data"][trace])
                for trace in range(len(fig_arrivals_2["data"])):
                    fig_trace.append(fig_arrivals_2["data"][trace])

                for traces in fig_trace:
                    fig_arrivals.append_trace(traces, row=1, col=1)

                st.plotly_chart(
                    fig_arrivals, use_container_width=True
                )

        with tab4:
            st.markdown("### Wait for initial appointment")
            st.dataframe(
                results_summary(results_all, results_low, results_high)
                )

            st.markdown("### Wait from booking to initial appointment")

        with tab1:
            st.subheader("Animated Event Log")
            st.plotly_chart(fig)

        with tab8:
            st.subheader("Full Event Log")
            st.dataframe(event_log_df)
            # AgGrid(event_log_df)

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
            st.subheader("Slot Utilisation - % of Slots Used")

            st.write(
                    round(
                        (((bookings.iloc[WARM_UP:RUN_LENGTH,]).sum().sum() /
                          ((bookings.iloc[WARM_UP:RUN_LENGTH,]) + available_slots.iloc[WARM_UP:RUN_LENGTH,]).sum().sum())
                          )*100,
                          1
                        )
                      )

            st.write(
                round((((bookings.iloc[WARM_UP:RUN_LENGTH,]).sum() /
                ((bookings.iloc[WARM_UP:RUN_LENGTH,]) + available_slots.iloc[WARM_UP:RUN_LENGTH,]).sum()).T)*100,1)
                    )

        with tab7:
            st.subheader("Daily Caseload Snapshots")

            cl = pd.DataFrame(daily_caseload_snapshots["caseload_day_end"].tolist())
            cl_filtered = cl.iloc[WARM_UP:RUN_LENGTH,:]

            st.write(cl_filtered)
            cl_plotting = cl_filtered.reset_index(drop=False).melt(id_vars=["index"], var_name="clinician", value_name="caseload")
            st.plotly_chart(
                px.line(
                cl_plotting,
                x="index",
                y= "caseload",
                color="clinician",
                range_y=[0, max(cl_plotting["caseload"])]
                )
            )

            st.subheader("Total Caseload in Use By Day")
            st.write(cl_filtered.sum(axis=1))

            st.subheader("% Caseload in Use By Day")
            st.write(cl_filtered.sum(axis=1)/(np.floor(shifts_edited.sum() * CASELOAD_TARGET_MULTIPLIER).sum()))

            px.line((cl_filtered.sum(axis=1)/(np.floor(shifts_edited.sum() * CASELOAD_TARGET_MULTIPLIER).sum())).reset_index(),
                    x="index", y=0)

            # st.plotly_chart(
            #     px.line(
            #         cl_filtered.sum(axis=1),
            #         x=
            #     )
            # )

        with tab2:
            st.subheader("Waiting List over Time")

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
                fig_daily_position_counts,
                use_container_width=True

            )

            st.subheader("Daily Waiting List Snapshots - Alternate method")

            bq = pd.DataFrame(daily_waiting_for_booking_snapshots["booking_queue_size_day_end"].tolist())
            bq_filtered = bq.iloc[WARM_UP:RUN_LENGTH,:]

            # st.write(bq_filtered)
            daily_wl_fig_alternate = px.line(
                bq_filtered.reset_index(drop=False),
                x="index",
                y= 0
                )

            col2a, col2b = st.columns([1,3])
            with col2a:
                st.write(bq_filtered)

            with col2b:
                st.plotly_chart(daily_wl_fig_alternate, use_container_width=True)

            st.subheader("Balance between people arriving in the system and departing")

            st.markdown("""
                This looks at the number of people arriving in the system - including those who are
                rejected before assessment as being inappropriate for the service - and the number
                departing at any point in their journey. The coloured lines are the rolling 100 day
                average. Ideally the two lines should roughly overlap, though some points of minor
                divergence are to be expected due to the variation in the number of arrivals, the number
                being rejected, the number leaving after assessment without ongoing treatment, and the
                number of follow-up appointments required per client.
                """)

            st.plotly_chart(
                arrival_depart_balance_fig,
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
