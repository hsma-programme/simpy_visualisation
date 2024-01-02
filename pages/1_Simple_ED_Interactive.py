import streamlit as st
import pandas as pd
import plotly.express as px
from examples.ex_1_simplest_case.simulation_execution_functions import single_run, multiple_replications
from examples.ex_1_simplest_case.model_classes import Scenario, TreatmentCentreModelSimpleNurseStepOnly
from examples.distribution_classes import Normal
from output_animation_functions import animate_activity_log
import gc

st.set_page_config(layout="wide", 
                   initial_sidebar_state="expanded",
                   page_title="Forced Overcrowding - Simple ED")

st.title("Simple Interactive Treatment Step")

st.markdown(
    """
This interactive simulation shows the simplest use of the animated event log.
    """
)

gc.collect()

col1, col2 = st.columns(2)

with col1:

    nurses = st.slider("👨‍⚕️👩‍⚕️ How Many Rooms/Nurses Are Available?", 1, 15, step=1, value=4)

    seed = st.slider("🎲 Set a random number for the computer to start from",
                        1, 1000,
                        step=1, value=42)

    with st.expander("Previous Parameters"):

        st.markdown("If you like, you can edit these parameters too!")
        
        n_reps = st.slider("🔁 How many times should the simulation run?",
                        1, 30,
                        step=1, value=6)
        
        run_time_days = st.slider("🗓️ How many days should we run the simulation for each time?",
                                1, 40,
                                step=1, value=10)

    
        mean_arrivals_per_day = st.slider("🧍 How many patients should arrive per day on average?",
                                        10, 300,
                                        step=5, value=120)

with col2:

    consult_time = st.slider("⏱️ How long (in minutes) does a consultation take on average?",
                                5, 150, step=5, value=50)

    consult_time_sd = st.slider("🕔 🕣 How much (in minutes) does the time for a consultation usually vary by?",
                                5, 30, step=5, value=10)

    norm_dist = Normal(consult_time, consult_time_sd, random_seed=seed)
    norm_fig = px.histogram(norm_dist.sample(size=2500), height=150)
    
    norm_fig.update_layout(yaxis_title="", xaxis_title="Consultation Time<br>(Minutes)")

    norm_fig.update_xaxes(tick0=0, dtick=10, range=[0, 
                                                    # max(norm_dist.sample(size=2500))
                                                    240
                                                    ])

    

    norm_fig.layout.update(showlegend=False, 
                            margin=dict(l=0, r=0, t=0, b=0))
    
    st.markdown("#### Consultation Time Distribution")
    st.plotly_chart(norm_fig,
                    use_container_width=True,
                    config = {'displayModeBar': False})
    
    
        
# A user must press a streamlit button to run the model
button_run_pressed = st.button("Run simulation")


if button_run_pressed:

    # add a spinner and then display success box
    with st.spinner('Simulating the minor injuries unit...'):

        args = Scenario(manual_arrival_rate=60/(mean_arrivals_per_day/24),
                        n_cubicles_1=nurses,
                        random_number_set=seed,
                        trauma_treat_mean=consult_time,
                        trauma_treat_var=consult_time_sd)

        model = TreatmentCentreModelSimpleNurseStepOnly(args)

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
        # st.dataframe(detailed_results)

        # animation_df = reshape_for_animations(
        #     event_log=detailed_results[detailed_results['rep']==1], 
        #     every_x_time_units=10,
        #     limit_duration=10*60*24,
        #     step_snapshot_max=50
        #     )

        # st.dataframe(
        #     animation_df
        # )


        event_position_df = pd.DataFrame([
                    {'event': 'arrival', 'x':  50, 'y': 300, 'label': "Arrival" },
                    
                    # Triage - minor and trauma                
                    {'event': 'treatment_wait_begins', 'x':  205, 'y': 170, 'label': "Waiting for Treatment"  },
                    {'event': 'treatment_begins', 'x':  205, 'y': 110, 'resource':'n_cubicles_1', 'label': "Being Treated" },

                    {'event': 'exit', 'x':  270, 'y': 70, 'label': "Exit"}
                
                ])


        st.plotly_chart(
            animate_activity_log(
                event_log=detailed_results[detailed_results['rep']==1],
                event_position_df= event_position_df,
                scenario=args,
                debug_mode=True,
                every_x_time_units=5,
                include_play_button=True,
                return_df_only=False,
                icon_and_text_size=20,
                gap_between_entities=6,
                gap_between_rows=15,
                plotly_height=700,
                plotly_width=1200,
                override_x_max=300,
                override_y_max=500,
                wrap_queues_at=25,
                step_snapshot_max=125,
                time_display_units="dhm",
                display_stage_labels=False,
                add_background_image="https://raw.githubusercontent.com/hsma-programme/Teaching_DES_Concepts_Streamlit/main/resources/Simplest%20Model%20Background%20Image%20-%20Horizontal%20Layout.drawio.png",
            ), use_container_width=False,
                config = {'displayModeBar': False}
        )