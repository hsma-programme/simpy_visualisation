import streamlit as st
import pandas as pd
import plotly.express as px
from examples.ex_2_branching_and_optional_paths.simulation_execution_functions import single_run, multiple_replications
from examples.ex_2_branching_and_optional_paths.model_classes import Scenario, TreatmentCentreModel
from vidigi.animation import animate_activity_log
import gc

st.set_page_config(layout="wide",
                   initial_sidebar_state="expanded",
                   page_title="Complex ED")

st.title("Simple Interactive Treatment Step")

st.markdown(
    """
This interactive simulation builds on the previous examples to demonstrate a multi-step, branching pathway with some optional steps.
    """
)

gc.collect()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Triage")
    n_triage = st.slider("👨‍⚕️👩‍⚕️ Number of Triage Cubicles", 1, 10, step=1, value=4)
    prob_trauma = st.slider("🚑 Probability that a new arrival is a trauma patient",
                            0.0, 1.0, step=0.01, value=0.3,
                            help="0 = No arrivals are trauma patients\n\n1 = All arrivals are trauma patients")

with col2:
    st.subheader("Trauma Pathway")
    n_trauma = st.slider("👨‍⚕️👩‍⚕️ Number of Trauma Bays for Stabilisation", 1, 10, step=1, value=6)
    n_cubicles_2 = st.slider("👨‍⚕️👩‍⚕️ Number of Treatment Cubicles for Trauma", 1, 10, step=1, value=6)

with col3:
    st.subheader("Non-Trauma Pathway")
    n_reg = st.slider("👨‍⚕️👩‍⚕️ Number of Registration Cubicles", 1, 10, step=1, value=3)
    n_exam = st.slider("👨‍⚕️👩‍⚕️ Number of Examination Rooms for non-trauma patients", 1, 10, step=1, value=3)

with col4:
    st.subheader("Non-Trauma Treatment")
    n_cubicles_1 = st.slider("👨‍⚕️👩‍⚕️ Number of Treatment Cubicles for Non-Trauma", 1, 10, step=1, value=2)
    non_trauma_treat_p = st.slider("🤕 Probability that a non-trauma patient will need treatment",
                                    0.0, 1.0, step=0.01, value=0.7,
                                    help="0 = No non-trauma patients need treatment\n\n1 = All non-trauma patients need treatment")


col5, col6 = st.columns(2)
with col5:
    st.write("Total rooms in use is {}".format(n_cubicles_1+n_cubicles_2+n_exam+n_trauma+n_triage+n_reg))
with col6:
    with st.expander("Advanced Parameters"):
        seed = st.slider("🎲 Set a random number for the computer to start from",
                        1, 1000,
                        step=1, value=42)

        n_reps = st.slider("🔁 How many times should the simulation run? WARNING: Fast/modern computer required to take this above 5 replications.",
                        1, 10,
                        step=1, value=3)

        run_time_days = st.slider("🗓️ How many days should we run the simulation for each time?",
                    1, 60,
                    step=1, value=5)


# A user must press a streamlit button to run the model
button_run_pressed = st.button("Run simulation")

if button_run_pressed:

    # add a spinner and then display success box
    with st.spinner('Simulating the department...'):

        args = Scenario(
                random_number_set=seed,
                        n_triage=n_triage,
                        n_reg=n_reg,
                        n_exam=n_exam,
                        n_trauma=n_trauma,
                        n_cubicles_1=n_cubicles_1,
                        n_cubicles_2=n_cubicles_2,
                        non_trauma_treat_p=non_trauma_treat_p,
                        prob_trauma=prob_trauma)


        model = TreatmentCentreModel(args)

        st.subheader("Single Run")

        results_df = single_run(args)

        st.dataframe(results_df)

        st.subheader("Multiple Runs")

        df_results_summary, detailed_results = multiple_replications(
                        args,
                        n_reps=n_reps,
                        rc_period=run_time_days*24*60,
                        return_detailed_logs=True
                    )

        st.dataframe(df_results_summary)

        event_position_df = pd.DataFrame([
                # {'event': 'arrival', 'x':  10, 'y': 250, 'label': "Arrival" },

                # Triage - minor and trauma
                {'event': 'triage_wait_begins',
                 'x':  155, 'y': 400, 'label': "Waiting for<br>Triage"  },
                {'event': 'triage_begins',
                 'x':  155, 'y': 315, 'resource':'n_triage', 'label': "Being Triaged" },

                # Minors (non-trauma) pathway
                {'event': 'MINORS_registration_wait_begins',
                 'x':  295, 'y': 145, 'label': "Waiting for<br>Registration"  },
                {'event': 'MINORS_registration_begins',
                 'x':  295, 'y': 85, 'resource':'n_reg', 'label':'Being<br>Registered'  },

                {'event': 'MINORS_examination_wait_begins',
                 'x':  460, 'y': 145, 'label': "Waiting for<br>Examination"  },
                {'event': 'MINORS_examination_begins',
                 'x':  460, 'y': 85, 'resource':'n_exam', 'label': "Being<br>Examined" },

                {'event': 'MINORS_treatment_wait_begins',
                 'x':  625, 'y': 145, 'label': "Waiting for<br>Treatment"  },
                {'event': 'MINORS_treatment_begins',
                 'x':  625, 'y': 85, 'resource':'n_cubicles_1', 'label': "Being<br>Treated" },

                # Trauma pathway
                {'event': 'TRAUMA_stabilisation_wait_begins',
                 'x': 295, 'y': 540, 'label': "Waiting for<br>Stabilisation" },
                {'event': 'TRAUMA_stabilisation_begins',
                 'x': 295, 'y': 480, 'resource':'n_trauma', 'label': "Being<br>Stabilised" },

                {'event': 'TRAUMA_treatment_wait_begins',
                 'x': 625, 'y': 540, 'label': "Waiting for<br>Treatment" },
                {'event': 'TRAUMA_treatment_begins',
                 'x': 625, 'y': 480, 'resource':'n_cubicles_2', 'label': "Being<br>Treated" },

                 {'event': 'exit',
                 'x':  670, 'y': 330, 'label': "Exit"}
            ])


        st.plotly_chart(
            animate_activity_log(
                event_log=detailed_results[detailed_results['rep']==1],
                event_position_df= event_position_df,
                scenario=args,
                debug_mode=True,
                limit_duration=run_time_days*24*60,
                every_x_time_units=5,
                include_play_button=True,
                gap_between_entities=10,
                gap_between_rows=25,
                plotly_height=900,
                plotly_width=1600,
                override_x_max=700,
                override_y_max=675,
                icon_and_text_size=22,
                wrap_queues_at=10,
                step_snapshot_max=30,
                time_display_units="dhm",
                display_stage_labels=False,
                add_background_image="https://raw.githubusercontent.com/hsma-programme/Teaching_DES_Concepts_Streamlit/main/resources/Full%20Model%20Background%20Image%20-%20Horizontal%20Layout.drawio.png",
            ), use_container_width=False,
                config = {'displayModeBar': False}
        )
