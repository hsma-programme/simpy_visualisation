# ## Executing a model
import pandas as pd
import numpy as np
from simulation_summary_functions import SimulationSummary

def single_run(scenario, 
               model,
               rc_period=600,
               random_no_set=1,
               return_detailed_logs=False,
               ):
    '''
    Perform a single run of the model and return the results

    Parameters:
    -----------

    scenario: Scenario object
        The scenario/paramaters to run

    rc_period: int
        The length of the simulation run that collects results

    random_no_set: int or None, optional (default=DEFAULT_RNG_SET)
        Controls the set of random seeds used by the stochastic parts of the 
        model.  Set to different ints to get different results.  Set to None
        for a random set of seeds.

    Returns:
    --------
        pandas.DataFrame:
        results from single run.
    '''
    # set random number set - this controls sampling for the run.
    scenario.set_random_no_set(random_no_set)

    # create an instance of the model
    model = model(scenario)

    # run the model
    model.run(results_collection_period=rc_period)

    # run results
    summary = SimulationSummary(model)

    if return_detailed_logs:
        return {
            'full_event_log': pd.DataFrame(model.full_event_log),
            'patient_log':  pd.DataFrame(summary.patient_log),
            'utilisation_audit': pd.DataFrame(model.utilisation_audit),
            'summary_df': summary.summary_frame()
        }

    summary_df = summary.summary_frame()

    return summary_df


def multiple_replications(scenario,
                          model,
                          rc_period=600,
                          n_reps=10,
                          return_detailed_logs=False):
    '''
    Perform multiple replications of the model.

    Params:
    ------
    scenario: Scenario
        Parameters/arguments to configurethe model

    rc_period: float, optional (default=DEFAULT_RESULTS_COLLECTION_PERIOD)
        results collection period.  
        the number of minutes to run the model to collect results

    n_reps: int, optional (default=DEFAULT_N_REPS)
        Number of independent replications to run.

    Returns:
    --------
    pandas.DataFrame
    '''

    if return_detailed_logs:
        results = [{'rep': rep+1,
                    'results': single_run(scenario,
                                          model,
                                          rc_period,
                                          random_no_set=(scenario.random_number_set)+rep,
                                          return_detailed_logs=True)}
                   for rep in range(n_reps)]

        # format and return results in a dataframe

        return results


    # If not returning detailed logs, do some additional steps before returning the summary df
    results = [single_run(scenario,
                          model,
                          rc_period,
                          random_no_set=(scenario.random_number_set)+rep)
               for rep in range(n_reps)]

    # format and return results in a dataframe
    df_results = pd.concat(results)
    df_results.index = np.arange(1, len(df_results)+1)
    df_results.index.name = 'rep'
    return df_results