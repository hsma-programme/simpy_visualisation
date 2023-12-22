import streamlit as st
import pandas as pd
from examples.ex_1_simplest_case.simulation_execution_functions import single_run, multiple_replications
from examples.ex_1_simplest_case.model_classes import Scenario, TreatmentCentreModelSimpleNurseStepOnly
from output_animation_functions import animate_activity_log, reshape_for_animations

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

args = Scenario(manual_arrival_rate=2,
                n_cubicles_1=5)

model = TreatmentCentreModelSimpleNurseStepOnly(args)

st.subheader("Single Run")

results_df = single_run(args)

st.dataframe(results_df)

st.subheader("Multiple Runs")

df_results_summary, detailed_results = multiple_replications(
                args,
                n_reps=5,
                rc_period=10*60*24,
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
            {'event': 'treatment_wait_begins', 'x':  200, 'y': 170, 'label': "Waiting for Treatment"  },
            {'event': 'treatment_begins', 'x':  200, 'y': 110, 'resource':'n_cubicles_1', 'label': "Being Treated" },

            {'event': 'exit', 'x':  270, 'y': 70, 'label': "Exit"}
        
        ])


st.plotly_chart(
    animate_activity_log(
        event_log=detailed_results[detailed_results['rep']==1],
        event_position_df= event_position_df,
        scenario=args,
        every_x_time_units=5,
        include_play_button=True,
        return_df_only=False,
        icon_and_text_size=20,
        gap_between_entities=5,
        gap_between_rows=15,
        plotly_height=700,
        plotly_width=1200,
        override_x_max=300,
        override_y_max=500,
        wrap_queues_at=20,
        step_snapshot_max=100,
        time_display_units="dhm",
        display_stage_labels=False,
        add_background_image="https://raw.githubusercontent.com/hsma-programme/Teaching_DES_Concepts_Streamlit/main/resources/Simplest%20Model%20Background%20Image%20-%20Horizontal%20Layout.drawio.png",
    ), use_container_width=False,
        config = {'displayModeBar': False}
)