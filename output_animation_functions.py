import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime as dt

def reshape_for_animations(full_event_log, every_x_minutes=10):
    minute_dfs = list()
    patient_dfs = list()

    for rep in range(1, max(full_event_log['rep'])+1):
        # print("Rep {}".format(rep))
        # Start by getting data for a single rep
        filtered_log_rep = full_event_log[full_event_log['rep'] == rep].drop('rep', axis=1)
        pivoted_log = filtered_log_rep.pivot_table(values="time", 
                                            index=["patient","event_type","pathway"], 
                                            columns="event").reset_index()

        for minute in range(10*60*24):
            # print(minute)
            # Get patients who arrived before the current minute and who left the system after the current minute
            # (or arrived but didn't reach the point of being seen before the model run ended)
            # When turning this into a function, think we will want user to pass
            # 'first step' and 'last step' or something similar
            # and will want to reshape the event log for this so that it has a clear start/end regardless
            # of pathway (move all the pathway stuff into a separate column?)

            # Think we maybe need a pathway order and pathway precedence column
            # But what about shared elements of each pathway?
            if minute % every_x_minutes == 0:

                try:
                    current_patients_in_moment = pivoted_log[(pivoted_log['arrival'] <= minute) & 
                                (
                                    (pivoted_log['depart'] >= minute) |
                                    (pivoted_log['depart'].isnull() )
                                )]['patient'].values
                except KeyError:
                    current_patients_in_moment = None
                
                if current_patients_in_moment is not None:
                    patient_minute_df = filtered_log_rep[filtered_log_rep['patient'].isin(current_patients_in_moment)]
                    # print(len(patient_minute_df))
                    # Grab just those clients from the filtered log (the unpivoted version)
                    # Each person can only be in a single place at once, so filter out any events
                    # that have taken place after the minute
                    # then just take the latest event that has taken place for each client
                    # most_recent_events_minute = patient_minute_df[patient_minute_df['time'] <= minute] \
                    #     .sort_values('time', ascending=True) \
                    #     .groupby(['patient',"event_type","pathway"]) \
                    #     .tail(1)  

                    most_recent_events_minute_ungrouped = patient_minute_df[patient_minute_df['time'] <= minute].reset_index() \
                        .sort_values(['time', 'index'], ascending=True) \
                        .groupby(['patient']) \
                        .tail(1) 

                    patient_dfs.append(most_recent_events_minute_ungrouped.assign(minute=minute, rep=rep))

                    # Now count how many people are in each state
                    # CHECK - I THINK THIS IS PROBABLY DOUBLE COUNTING PEOPLE BECAUSE OF THE PATHWAY AND EVENT TYPE. JUST JOIN PATHWAY/EVENT TYPE BACK IN INSTEAD?
                    state_counts_minute = most_recent_events_minute_ungrouped[['event']].value_counts().rename("count").reset_index().assign(minute=minute, rep=rep)
                    
                    minute_dfs.append(state_counts_minute)


    minute_counts_df = pd.concat(minute_dfs).merge(filtered_log_rep[['event','event_type', 'pathway']].drop_duplicates().reset_index(drop=True), on="event")
    full_patient_df = pd.concat(patient_dfs).sort_values(["rep", "minute", "event"])

    # Add a final exit step for each client
    final_step = full_patient_df.sort_values(["rep", "patient", "minute"], ascending=True).groupby(["rep", "patient"]).tail(1)
    final_step['minute'] = final_step['minute'] + every_x_minutes
    final_step['event'] = "exit"
    # final_step['event_type'] = "arrival_departure"

    full_patient_df = full_patient_df.append(final_step)

    minute_counts_df_pivoted = minute_counts_df.pivot_table(values="count", 
                                            index=["minute", "rep", "event_type", "pathway"], 
                                            columns="event").reset_index().fillna(0)

    minute_counts_df_complete = minute_counts_df_pivoted.melt(id_vars=["minute", "rep","event_type","pathway"])

    return {
        "minute_counts_df": minute_counts_df,
        "minute_counts_df_complete": minute_counts_df_complete,
        "full_patient_df": full_patient_df.sort_values(["rep", "minute", "event"])
        
    }


# ['TRAUMA_triage_wait_begins', 'TRAUMA_triage_begins', 'TRAUMA_triage_complete', 
#                                     'TRAUMA_stabilisation_wait_begins', 'TRAUMA_stabilisation_begins', 'TRAUMA_stabilisation_complete', 
#                                     'TRAUMA_treatment_wait_begins', 'TRAUMA_treatment_begins', 'TRAUMA_treatment_wait_begins'
#                                     ]


def animate_activity_log(
        full_patient_df,
        event_position_df,
        scenario,
        rep=1,
        plotly_height=900,
        plotly_width=None,
        wrap_queues_at=None,
        include_play_button=True,
        return_df_only=False,
        add_background_image=None,
        display_stage_labels=True,
        icon_and_text_size=24,
        override_x_max=None,
        override_y_max=None,
        time_display_units=None,
        setup_mode=False,
        frame_duration=400, #milliseconds
        frame_transition_duration=600 #milliseconds
        ):
    """_summary_

    Args:
        full_patient_df (pd.Dataframe): 
        
        event_position_dicts (pd.Dataframe): 
            dataframe with three cols - event, x and y
            Can be more easily created by passing a list of dicts to pd.DataFrame
            list of dictionaries with one dicitionary per event type
            containing keys 'event', 'x' and 'y'
            This will determine the intial position of any entries in the animated log
            (think of it as the bottom right hand corner of any group of entities at each stage)

        scenario:
            Pass in an object that specifies the number of resources at different steps

        rep (int, optional): Defaults to 1.
            The replication of any model to include. Can only display one rep at a time, so will take
            the first rep if not otherwise specified. 
        
        plotly_height (int, optional): Defaults to 900.

    Returns:
       Plotly fig object
    """        

    # Filter to only a single replication

    # TODO: Remove this from this function, and instead write a test
    # to ensure that no patient ID appears in multiple places at a single minute
    # and return an error if it does so
    # Move the step of ensuring there's only a single model run involved to outside
    # of this function as it's not really its job. 

    full_patient_df = full_patient_df[full_patient_df['rep'] == rep].sort_values([
        'event','minute','time'
        ])

    # full_patient_df['count'] = full_patient_df.groupby(['event','minute','rep'])['minute'] \
    #                            .transform('count')
    
    # Order patients within event/minute/rep to determine their eventual position in the line
    full_patient_df['rank'] = full_patient_df.groupby(['event','minute','rep'])['minute'] \
                              .rank(method='first')

    full_patient_df_plus_pos = full_patient_df.merge(event_position_df, on="event", how='left') \
                             .sort_values(["rep", "event", "minute", "time"])

    # Determine the position for any resource use steps
    resource_use = full_patient_df_plus_pos[full_patient_df_plus_pos['event_type'] == "resource_use"].copy()
    resource_use['y_final'] =  resource_use['y']
    resource_use['x_final'] = resource_use['x'] - resource_use['resource_id']*10

    # Determine the position for any queuing steps
    queues = full_patient_df_plus_pos[full_patient_df_plus_pos['event_type']=='queue']
    queues['y_final'] =  queues['y']
    queues['x_final'] = queues['x'] - queues['rank']*10

    # If we want people to wrap at a certain queue length, do this here
    # They'll wrap at the defined point and then the queue will start expanding upwards
    # from the starting row
    if wrap_queues_at is not None:
        queues['row'] = np.floor((queues['rank']) / (wrap_queues_at+1))
        queues['x_final'] = queues['x_final'] + (wrap_queues_at*queues['row']*10)
        queues['y_final'] = queues['y_final'] + (queues['row'] * 30)

    full_patient_df_plus_pos = pd.concat([queues, resource_use])

    # full_patient_df_plus_pos['icon'] = '🙍'

    individual_patients = full_patient_df['patient'].drop_duplicates().sort_values()
    
    # Recommend https://emojipedia.org/ for finding emojis to add to list
    # note that best compatibility across systems can be achieved by using 
    # emojis from v12.0 and below - Windows 10 got no more updates after that point
    icon_list = [
        '🧔🏼', '👨🏿‍🦯', '👨🏻‍🦰', '🧑🏻', '👩🏿‍🦱', 
        '🤰', '👳🏽', '👩🏼‍🦳', '👨🏿‍🦳', '👩🏼‍🦱', 
        '🧍🏽‍♀️', '👨🏼‍🔬', '👩🏻‍🦰', '🧕🏿', '👨🏼‍🦽', 
        '👴🏾', '👨🏼‍🦱', '👷🏾', '👧🏿', '🙎🏼‍♂️',
        '👩🏻‍🦲', '🧔🏾', '🧕🏻', '👨🏾‍🎓', '👨🏾‍🦲',
        '👨🏿‍🦰', '🙍🏼‍♂️', '🙋🏾‍♀️', '👩🏻‍🔧', '👨🏿‍🦽', 
        '👩🏼‍🦳', '👩🏼‍🦼', '🙋🏽‍♂️', '👩🏿‍🎓', '👴🏻', 
        '🤷🏻‍♀️', '👶🏾', '👨🏻‍✈️', '🙎🏿‍♀️', '👶🏻', 
        '👴🏿', '👨🏻‍🦳', '👩🏽', '👩🏽‍🦳', '🧍🏼‍♂️', 
        '👩🏽‍🎓', '👱🏻‍♀️', '👲🏼', '🧕🏾', '👨🏻‍🦯', 
        '🧔🏿', '👳🏿', '🤦🏻‍♂️', '👩🏽‍🦰', '👨🏼‍✈️', 
        '👨🏾‍🦲', '🧍🏾‍♂️', '👧🏼', '🤷🏿‍♂️', '👨🏿‍🔧', 
        '👱🏾‍♂️', '👨🏼‍🎓', '👵🏼', '🤵🏿', '🤦🏾‍♀️',
        '👳🏻', '🙋🏼‍♂️', '👩🏻‍🎓', '👩🏼‍🌾', '👩🏾‍🔬',
        '👩🏿‍✈️', '🎅🏼', '👵🏿', '🤵🏻', '🤰'
    ]

    full_icon_list = icon_list * int(np.ceil(len(individual_patients)/len(icon_list)))

    full_icon_list = full_icon_list[0:len(individual_patients)]

    full_patient_df_plus_pos = full_patient_df_plus_pos.merge(
        pd.DataFrame({'patient':list(individual_patients),
                      'icon':full_icon_list}),
        on="patient")

    if return_df_only:
        return full_patient_df_plus_pos

    if override_x_max is not None:
        x_max = override_x_max
    else:
        x_max = event_position_df['x'].max()*1.25

    if override_y_max is not None:
        y_max = override_x_max
    else:
        y_max = event_position_df['y'].max()*1.1

    # If we're displaying time as a clock instead of as units of whatever time our model
    # is working in, create a minute_display column that will display as a psuedo datetime
    
    # For now, it starts a few months after the current date, just to give the
    # idea of simulating some hypothetical future time. It might be nice to allow
    # the start point to be changed, particular if we're simulating something on
    # a larger timescale that includes a level of weekly or monthly seasonality.

    # We need to keep the original minute column in existance because it's important for sorting
    if time_display_units == "dhm":
        full_patient_df_plus_pos['minute'] = dt.date.today() + pd.DateOffset(days=165) +  pd.TimedeltaIndex(full_patient_df_plus_pos['minute'], unit='m')
        # https://strftime.org/
        full_patient_df_plus_pos['minute_display'] = full_patient_df_plus_pos['minute'].apply(
            lambda x: dt.datetime.strftime(x, '%d %B %Y\n%H:%M')
            )
        full_patient_df_plus_pos['minute'] = full_patient_df_plus_pos['minute'].apply(
            lambda x: dt.datetime.strftime(x, '%Y-%m-%d %H:%M')
            )
    else:
        full_patient_df_plus_pos['minute_display'] = full_patient_df_plus_pos['minute']

    # full_patient_df_plus_pos['size'] = 24

    # We are effectively making use of an animated plotly express scatterploy
    # to do all of the heavy lifting
    # Because of the way plots animate in this, it deals with all of the difficulty
    # of paths between individual positions - so we just have to tell it where to put
    # people at each defined step of the process, and the scattergraph will move them

    fig = px.scatter(
            full_patient_df_plus_pos.sort_values('minute'),
            x="x_final",
            y="y_final",
            # Each frame is one step of time, with the gap being determined
            # in the reshape_for_animation function
            animation_frame="minute_display",
            # Important to group by patient here
            animation_group="patient",
            text="icon",
            # Can't have colours because it causes bugs with
            # lots of points failing to appear
            #color="event",
            hover_name="event",
            hover_data=["patient", "pathway", "time", "minute", "resource_id"],
            # The approach of putting in the people as symbols didn't work
            # Went with making emoji text labels instead - this works better!
            # But leaving in as a reminder that the symbol approach doens't work.
            #symbol="rep",
            #symbol_sequence=["⚽"],
            #symbol_map=dict(rep_choice = "⚽"),
            range_x=[0, x_max],
            range_y=[0, y_max],
            height=plotly_height,
            width=plotly_width,
            # This sets the opacity of the points that sit behind
            opacity=0
            #    size="size"
            )

    # Now add labels identifying each stage (optional - can either be used
    # in conjunction with a background image or as a way to see stage names
    # without the need to create a background image)
    if display_stage_labels:
        fig.add_trace(go.Scatter(
            x=[pos+10 for pos in event_position_df['x'].to_list()],
            y=event_position_df['y'].to_list(),
            mode="text",
            name="",
            text=event_position_df['label'].to_list(),
            textposition="middle right",
            hoverinfo='none'
        ))

    # Update the size of the icons and labels
    # This is what determines the size of the individual emojis that 
    # represent our people!
    fig.update_traces(textfont_size=icon_and_text_size)

    # Finally add in icons to indicate the available resources
    # Make an additional dataframe that has one row per resource type
    # Then, starting from the initial position, make that many large circles
    # make them semi-transparent or you won't see the people using them! 
    events_with_resources = event_position_df[event_position_df['resource'].notnull()].copy()
    events_with_resources['resource_count'] = events_with_resources['resource'].apply(lambda x: getattr(scenario, x))

    events_with_resources = events_with_resources.join(events_with_resources.apply(
        lambda r: pd.Series({'x_final': [r['x']-(10*(i+1)) for i in range(r['resource_count'])]}), axis=1).explode('x_final'),
        how='right')

    # This just adds an additional scatter trace that creates large dots
    # that represent the individual resources
    fig.add_trace(go.Scatter(
        x=events_with_resources['x_final'].to_list(),
        # Place these slightly below the y position for each entity
        # that will be using the resource
        y=[i-10 for i in events_with_resources['y'].to_list()],
        mode="markers",
        # Define what the marker will look like
        marker=dict(
            color='LightSkyBlue',
            size=15),
        opacity=0.8,
        hoverinfo='none'
    ))

    # Optional step to add a background image
    # This can help to better visualise the layout/structure of a pathway
    # Simple FOSS tool for creating these background images is draw.io
    # Ideally your queueing steps should always be ABOVE your resource use steps
    # as this then results in people nicely flowing from the front of the queue 
    # to the next stage
    if add_background_image is not None:
        fig.add_layout_image(
            dict(
                source=add_background_image,
                xref="x domain",
                yref="y domain",
                x=1,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="right",
                yanchor="top",
                sizing="stretch",
                opacity=0.5,
                layer="below")
    )

    # We don't need any gridlines or tickmarks for the final output, so remove
    # However, can be useful for the initial setup phase of the outputs, so give the 
    # option to inlcude
    if not setup_mode:
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, 
                         # Prevent zoom
                         fixedrange=True)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, 
                         # Prevent zoom
                         fixedrange=True)

    fig.update_layout(yaxis_title=None, xaxis_title=None, showlegend=False,
                      # Increase the size of the play button and animation timeline
                      sliders=[dict(currentvalue=dict(font=dict(size=35) ,
                                    prefix=""))]
                                )

    # You can get rid of the play button if desired
    # Was more useful in older versions of the function
    if not include_play_button:
        fig["layout"].pop("updatemenus")

    # Adjust speed of animation
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = frame_duration
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = frame_transition_duration

    return fig


def animate_queue_activity_bar_chart(minute_counts_df_complete,
                                     event_order,
                                     rep=1):
    # Downsample to only include a snapshot every 10 minutes (else it falls over completely)
    # For runs of more days will have to downsample more aggressively - every 10 minutes works for 15 days
    fig = px.bar(minute_counts_df_complete[minute_counts_df_complete["rep"] == int(rep)].sort_values('minute'),
                x="event",
                y="value",
                animation_frame="minute",
                range_y=[0,minute_counts_df_complete['value'].max()*1.1])

    fig.update_xaxes(categoryorder='array',
                    categoryarray= event_order)

    fig["layout"].pop("updatemenus")

    return fig