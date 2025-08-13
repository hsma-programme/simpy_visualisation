import gc
import time
import datetime as dt
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from examples.ex_3_theatres_beds.simulation_execution_functions import multiple_replications
from examples.ex_3_theatres_beds.model_classes import Scenario, Schedule
from vidigi.prep import reshape_for_animations, generate_animation_df
from vidigi.animation import generate_animation
from plotly.subplots import make_subplots

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

st.header("Animation parameters")

anim_param_col_1, anim_param_col_2 = st.columns(2)

show_los = anim_param_col_1.toggle("Show Individual Length of Stay")
show_delayed_discharges = anim_param_col_1.toggle("Show Delayed Discharges")
show_operation_type = anim_param_col_2.selectbox(
    "Choose Surgery Detail to Display",
    ["Show knee vs hip", "Show revision vs primary", "Show both", "Show standard patient icons"],
    index=0)

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

    @st.fragment
    def generate_animation_fig():
        results = multiple_replications(
                        return_detailed_logs=True,
                        scenario=args,
                        n_reps=replications,
                        results_collection=runtime
                    )
        with st.expander("Click to see a summary of the results"):
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
                    'x':  200, 'y': 650, 'label': "Waiting for<br>Availability of<br>Bed to be Confirmed<br>Before Surgery" },

                    {'event': 'no_bed_available',
                    'x':  600, 'y': 650, 'label': "No Bed<br>Available:<br>Surgery Cancelled" },

                    {'event': 'post_surgery_stay_begins',
                    'x':  650, 'y': 220, 'resource':'n_beds', 'label': "In Bed:<br>Recovering from<br>Surgery" },

                    {'event': 'discharged_after_stay',
                    'x':  670, 'y': 50, 'label': "Discharged from Hospital<br>After Recovery"}
                    # {'event': 'exit',
                    #  'x':  670, 'y': 100, 'label': "Exit"}

                    ])

        full_patient_df = reshape_for_animations(full_log_with_patient_details,
                                                entity_col_name="patient",
                                                every_x_time_units=1,
                                                limit_duration=runtime,
                                                step_snapshot_max=50,
                                                debug_mode=debug_mode
                                                )

        if debug_mode:
            print(f'Reshaped animation dataframe finished construction at {time.strftime("%H:%M:%S", time.localtime())}')

        full_patient_df_plus_pos = generate_animation_df(
                                    full_entity_df=full_patient_df,
                                    event_position_df=event_position_df,
                                    entity_col_name="patient",
                                    wrap_queues_at=20,
                                    wrap_resources_at=40,
                                    step_snapshot_max=50,
                                    gap_between_entities=15,
                                    gap_between_resources=15,
                                    gap_between_queue_rows=175,
                                    debug_mode=debug_mode
                            )

        def set_icon_standard(row):
            return f"{row['icon']}<br>"

        def set_icon_surgery_target(row):
            if "knee" in row["surgery type"]:
                return "🦵<br> "
            elif "hip" in row["surgery type"]:
                return "🕺<br> "
            else:
                return f"CHECK<br>{row['icon']}"

        def set_icon_surgery_type(row):
            if "p_" in row["surgery type"]:
                return "1️⃣<br> "
            elif "r_" in row["surgery type"]:
                return "♻️<br> "
            elif "uni_" in row["surgery type"] == "p_hip":
                return "✳️<br> "
            else:
                return f"CHECK<br>{row['icon']}"

        def set_icon_full(row):
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


        if show_operation_type == "Show both":
            full_patient_df_plus_pos = full_patient_df_plus_pos.assign(icon=full_patient_df_plus_pos.apply(set_icon_full, axis=1))
        elif show_operation_type == "Show knee vs hip":
            full_patient_df_plus_pos = full_patient_df_plus_pos.assign(icon=full_patient_df_plus_pos.apply(set_icon_surgery_target, axis=1))
        elif show_operation_type == "Show revision vs primary":
            full_patient_df_plus_pos = full_patient_df_plus_pos.assign(icon=full_patient_df_plus_pos.apply(set_icon_surgery_type, axis=1))
        else:
            full_patient_df_plus_pos = full_patient_df_plus_pos.assign(icon=full_patient_df_plus_pos.apply(set_icon_standard, axis=1))

        # TODO: Check why this doesn't seem to be working quite right for the 'discharged after stay'
        # step. e.g. 194Primary is discharged on 28th July showing a LOS of 1 but prior to this shows a LOS of 9.
        def add_los_to_icon(row):
            if row["event"] == "post_surgery_stay_begins":
                return f'{row["icon"]}<br>{row["snapshot_time"]-row["time"]:.0f}'
            elif row["event"] == "discharged_after_stay":
                return f'{row["icon"]}<br>{row["los"]:.0f}'
            else:
                return row["icon"]

        if show_los:
            full_patient_df_plus_pos = full_patient_df_plus_pos.assign(icon=full_patient_df_plus_pos.apply(add_los_to_icon, axis=1))


        def indicate_delay_via_icon(row):
            if row["delayed discharge"] is True:
                return f'{row["icon"]}<br>*'
            else:
                return f'{row["icon"]}<br> '

        if show_delayed_discharges:
            full_patient_df_plus_pos = full_patient_df_plus_pos.assign(icon=full_patient_df_plus_pos.apply(indicate_delay_via_icon, axis=1))


        with st.expander("Click here to view detailed event dataframes"):
            st.subheader("Event Log")
            st.subheader("Data - After merging full log with patient details")
            st.dataframe(full_log_with_patient_details)
            st.subheader("Dataframe - Reshaped for animation (step 1)")
            st.dataframe(full_patient_df)
            st.subheader("Dataframe - Reshaped for animation (step 2)")
            st.dataframe(full_patient_df_plus_pos)

        cancelled_due_to_no_bed_available = (
            len(full_log_with_patient_details[full_log_with_patient_details['event'] == "no_bed_available"]
                ["patient"].unique()
            )
        )
        total_patients = len(full_log_with_patient_details["patient"].unique())

        cancelled_perc = cancelled_due_to_no_bed_available/total_patients

        st.markdown(f"Surgeries cancelled due to no bed being available in time: {cancelled_perc:.2%} ({cancelled_due_to_no_bed_available} of {total_patients})")

        fig = generate_animation(
                full_entity_df_plus_pos=full_patient_df_plus_pos,
                event_position_df=event_position_df,
                scenario=args,
                entity_col_name="patient",
                plotly_height=950,
                plotly_width=1000,
                override_x_max=800,
                override_y_max=1000,
                text_size=14,
                resource_icon_size=16,
                entity_icon_size=14,
                wrap_resources_at=40,
                gap_between_resources=15,
                include_play_button=True,
                add_background_image=None,
                # we want the stage labels, but due to a bug
                # when we add in additional animated traces later,
                # they will disappear - so better to leave them out here
                # and then re-add them manually
                display_stage_labels=False,
                custom_resource_icon="🛏️",
                time_display_units="d",
                simulation_time_unit="day",
                start_date="2022-06-27",
                setup_mode=False,
                frame_duration=1500, #milliseconds
                frame_transition_duration=1000, #milliseconds
                debug_mode=False
            )

        # Create an additional dataframe calculating the number of times cancellations occur due
        # to no bed being available
        counts_not_avail = (
            full_patient_df_plus_pos[full_patient_df_plus_pos['event']=='no_bed_available']
            .sort_values('snapshot_time')
            [['snapshot_time','patient']]
            .groupby('snapshot_time')
            .agg('count')
            )

        # Ensure we have a value for every snapshot time in the animation by using this as the
        # index - this avoids the risk of the number of frames represented in this dataframe not
        # matching the total number of animation frames in the actual output figure
        counts_not_avail = (
            counts_not_avail.reset_index()
            .merge(full_patient_df_plus_pos[['snapshot_time']].drop_duplicates(), how='right')
            .sort_values('snapshot_time')).reset_index(drop=True)

        counts_not_avail['patient'] = counts_not_avail['patient'].fillna(0)

        # Calculate a running total of this value, which will be used to add the correct value
        # to each individual frame
        counts_not_avail['running_total'] = counts_not_avail.sort_values('snapshot_time')['patient'].cumsum()

        # Create an additional dataframe calculating the number of operations completed per day
        counts_ops_completed = (
            full_patient_df_plus_pos[full_patient_df_plus_pos['event']=='post_surgery_stay_begins']
            [['snapshot_time','patient']]
            .drop_duplicates('patient')
            .groupby('snapshot_time')
            .agg('count')
            )
        # Ensure we have a value for every snapshot time in the animation by using this as the
        # index - this avoids the risk of the number of frames represented in this dataframe not
        # matching the total number of animation frames in the actual output figure
        counts_ops_completed = (
            counts_ops_completed.reset_index()
            .merge(full_patient_df_plus_pos[['snapshot_time']].drop_duplicates(), how='right')
            .sort_values('snapshot_time')
            ).reset_index(drop=True)

        # For any days with no value, ensure this is changed to a 0
        counts_ops_completed['patient'] = counts_ops_completed['patient'].fillna(0)

        # Calculate a running total of this value, which will be used to add the correct value
        # to each individual frame
        counts_ops_completed['running_total'] = counts_ops_completed.sort_values('snapshot_time')['patient'].cumsum()
        counts_not_avail = (
            counts_not_avail
            .merge(counts_ops_completed.rename(columns={'running_total':'completed'}),
                   how="left", on="snapshot_time")
                   )
        counts_not_avail['perc_slots_lost'] = (
            counts_not_avail['running_total'] /
            (counts_not_avail['running_total'] + counts_not_avail['completed'])
            )

        #####################################################
        # Adding additional animation traces
        #####################################################

        ## First, add each trace so it will show up initially

        # Due to issues detailed in the following SO threads, it's essential to initialize the traces
        # outside of the frames argument else they will not show up at all (or show up intermittently)
        # https://stackoverflow.com/questions/69867334/multiple-traces-per-animation-frame-in-plotly
        # https://stackoverflow.com/questions/69367344/plotly-animating-a-variable-number-of-traces-in-each-frame-in-r
        # TODO: More explanation and investigation needed of why sometimes traces do and don't show up after being added in
        # via this method. Behaviour seems very inconsistent and not always logical (e.g. order you put traces in to the later
        # loop sometimes seems to make a difference but sometimes doesn't; making initial trace transparent sometimes seems to
        # stop it showing up when added in the frames but not always; sometimes the initial trace doesn't disappear).

        # Add bed trace in manually to ensure it can be referenced later
        fig.add_trace(go.Scatter(x=[100], y=[100]))

        fig.add_trace(fig.data[1])

        # Add animated text trace that gives running total of operations completed
        fig.add_trace(go.Scatter(
                        x=[100],
                        y=[30],
                        text=f"Operations Completed: {int(counts_ops_completed.sort_values('snapshot_time')['running_total'][0])}",
                        mode='text',
                        textfont=dict(size=20),
                        opacity=0,
                        showlegend=False,
                ))

        # Add animated trace giving running total of slots lost and percentage of total slots this represents
        fig.add_trace(go.Scatter(
            x=[600],
            y=[850],
            text="",
            # text=f"Total slots lost: {int(counts_not_avail['running_total'][0])}<br>({counts_not_avail['perc_slots_lost'][0]:.1%})",
            mode='text',
            textfont=dict(size=20),
            # opacity=0,
            showlegend=False,
        ))

        # Add trace for the event labels (as these get lost from the animation once we start trying to add other things in,
        # so need manually re-adding)
        fig.add_trace(go.Scatter(
                x=[pos+10 for pos in event_position_df['x'].to_list()],
                y=event_position_df['y'].to_list(),
                mode="text",
                name="",
                text=event_position_df['label'].to_list(),
                textposition="middle right",
                hoverinfo='none'
            ))

        # Ensure these all have the right text size
        fig.update_traces(textfont_size=14)

        # Now set up the desired subplot layout
        sp = make_subplots(rows=2, cols=1, row_heights=[0.85, 0.15], subplot_titles=("", "Daily lost slots"))

        # Overwrite the domain of our original x and y axis with domain from the new axis
        fig.layout['xaxis']['domain'] = sp.layout['xaxis']['domain']
        fig.layout['yaxis']['domain'] = sp.layout['yaxis']['domain']

        # Add in the attributes for the secondary axis from our subplot
        fig.layout['xaxis2'] = sp.layout['xaxis2']
        fig.layout['yaxis2'] = sp.layout['yaxis2']

        # Final key step - copy over the _grid_ref attribute
        # This isn't meant to be something we modify but it's an essential
        # part of the subplot code because otherwise plotly doesn't truly know
        # how the different subplots are arranged and referenced
        fig._grid_ref = sp._grid_ref

        # Add an initial trace to our secondary line chart
        fig.add_trace(go.Scatter(
            x=counts_not_avail['snapshot_time'],
            y=counts_not_avail['patient_x'],
            mode='lines',
            showlegend=False,
            # name='line',
            opacity=0.2,
            xaxis="x2",
            yaxis="y2"
            # We place it in our new subplot using the following line
        ), row=2, col=1)

        ##########################################################
        # Now we need to add our traces to each individual frame
        ##########################################################
        # To work correctly, these need to be provided in the same order as the traces above
        for i, frame in enumerate(fig.frames):
            frame.data =  (frame.data +
            # bed icons
            (fig.data[1],) +
            # Slots used/operations occurred
            (
                go.Scatter(
                    x=[100],
                    y=[30],
                    text=f"Operations Completed: {int(counts_ops_completed.sort_values('snapshot_time')['running_total'][i])}",
                    mode='text',
                    textfont=dict(size=20),
                    showlegend=False,
                ),)
                +
            # Slots lost
            (go.Scatter(
                    x=[600],
                    y=[800],
                    text=f"Total slots lost: {int(counts_not_avail['running_total'][i])}<br>({counts_not_avail['perc_slots_lost'][i]:.1%})",
                    mode='text',
                    textfont=dict(size=20),
                    showlegend=False,
                ),) +
            # Position labels
            (go.Scatter(
                x=[pos+10 for pos in event_position_df['x'].to_list()],
                y=event_position_df['y'].to_list(),
                mode="text",
                name="",
                text=event_position_df['label'].to_list(),
                textposition="middle right",
                hoverinfo='none'
            ),) +
            # Line subplot
            (go.Scatter(
                x=counts_not_avail['snapshot_time'][0: i+1].values,
                y=counts_not_avail['patient_x'][0: i+1].values,
                mode="lines",
                # name="line",
                # hoverinfo='none',
                showlegend=False,
                name="line_subplot",
                # line=dict(color="#f71707"),
                xaxis='x2',
                yaxis='y2'
            ),)
            #  +
            #  (
            #     go.Scatter(
            #         x=counts_ops_completed,
            #         y=[50],
            #         text=f"Operations Completed: {int(counts_ops_completed['running_total'][i])}",
            #         mode='text',
            #         textfont=dict(size=20),
            #         showlegend=False,
            #     ),
            #     )
            )
            #+ ((fig.data[-1]), ) + ((fig.data[-2]), )

        # Ensure we tell it which traces we are animating
        # (as per https://chart-studio.plotly.com/~empet/15243/animating-traces-in-subplotsbr/#/)
        for i, frame in enumerate(fig.frames):
            # This will ensure it matches the number of traces we have
            frame['traces'] = [i for i in range(len(fig.data)+1)]

        # for frame in fig.frames:
        #     fig._set_trace_grid_position(frame.data[-1], 2,1)

        # Finally, match these new traces with the text size used elsewhere
        fig.update_traces(textfont_size=14)

        # sp_test = make_subplots(rows=2, cols=1, row_heights=[0.85, 0.15])

        # test = sp_test.add_trace(go.Scatter(
        #         x=counts_not_avail['snapshot_time'][0: i+1].values,
        #         y=counts_not_avail['running_total'][0: i+1].values,
        #         mode="lines",
        #         name="line",
        #         # hoverinfo='none',
        #         showlegend=False,
        #         # line=dict(color="#f71707"),
        #         xaxis='x2',
        #         yaxis='y2'
        #     ),row=2, col=1)

        # fig



        # fig.frames[0]

        return fig


    with st.spinner():
        fig = generate_animation_fig()

    @st.fragment()
    def display_animation_fig(fig):

        frame_duration_col, frame_transition_duration_col = st.columns(2)

        frame_duration = frame_duration_col.number_input("Choose the frame duration in milliseconds (default is 1500ms)", 50, 5000, 1500)
        frame_transition_duration = frame_transition_duration_col.number_input("Choose the length of the transition between frames in milliseconds (default is 1000ms)", 0, 5000, 1000)

        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = frame_duration
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = frame_transition_duration

        st.plotly_chart(
            fig
        )

    display_animation_fig(fig=fig)




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

        Note that the "No Bed Available: Surgery Cancelled" and "Discharged from Hospital after Recovery" stages in the animation are lagged by one day.
        For example, on the 2nd of July, this will show the patients who had their surgery cancelled on 1st July or were discharged on 1st July.
        These steps are included to make it easier to understand the destinations of different clients, but due to the size of the simulation step shown (1 day) it is difficult to demonstrate this differently.
        """
    )



    # sp.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
    #             row=1, col=1)

    # sp.add_trace(go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
                # row=2, col=1)

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



# fig.b
