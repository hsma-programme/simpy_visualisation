import numpy as np
import pandas as pd

class SimulationSummary:
    '''
    End of run result processing logic of the simulation model
    '''

    def __init__(self, model):
        '''
        Constructor

        Params:
        ------
        model: TraumaCentreModel
            The model.
        '''
        self.model = model
        self.args = model.args
        self.results = None
        self.patient_log = None
        self.event_log = model.event_log
        self.utilisation_audit = model.utilisation_audit

    def get_mean_metric(self, metric, patients):
        '''
        Calculate mean of the performance measure for the
        select cohort of patients,

        Only calculates metrics for patients where it has been 
        measured.

        Params:
        -------
        metric: str
            The name of the metric e.g. 'wait_treat'

        patients: list
            A list of patients
        '''
        mean = np.array([getattr(p, metric) for p in patients
                         if getattr(p, metric) > -np.inf]).mean()
        return mean
    
    def get_perc_wait_target_met(self, metric, patients, target):
        '''
        Calculate the percentage of patients where a target was met for 
        the select cohort of patients,

        Only calculates metrics for patients where it has been 
        measured.

        Params:
        -------
        metric: str
            The name of the metric e.g. 'wait_treat'

        patients: list
            A list of patients
        '''
        met = len(np.array([getattr(p, metric) for p in patients
                         if getattr(p, metric) < target]))
        total = len(np.array([getattr(p, metric) for p in patients
                         if getattr(p, metric) > -np.inf]))
        return met/total

    def get_resource_util(self, metric, n_resources, patients):
        '''
        Calculate proportion of the results collection period
        where a resource was in use.

        Done by tracking the duration by patient.

        Only calculates metrics for patients where it has been 
        measured.

        Params:
        -------
        metric: str
            The name of the metric e.g. 'treatment_duration'

        patients: list
            A list of patients
        '''
        total = np.array([getattr(p, metric) for p in patients
                         if getattr(p, metric) > -np.inf]).sum()

        return total / (self.model.rc_period * n_resources)

    def get_throughput(self, patients):
        '''
        Returns the total number of patients that have successfully
        been processed and discharged in the treatment centre
        (they have a total time record)

        Params:
        -------
        patients: list
            list of all patient objects simulated.

        Returns:
        ------
        float
        '''
        return len([p for p in patients if p.total_time > -np.inf])


    def process_run_results(self, 
                            wait_target_per_step=120):
        '''
        Calculates statistics at end of run.
        '''
        self.results = {}

        self.patient_log = self.model.patients

        self.results = {'00_arrivals': len(self.model.patients),
                        '01a_treatment_wait': self.get_mean_metric('wait_treat', self.model.patients),
                        '01b_treatment_util': self.get_resource_util('treat_duration', self.args.n_cubicles_1,self.model.patients),
                        '01c_treatment_wait_target_met': self.get_perc_wait_target_met('wait_treat', self.model.patients,target=wait_target_per_step),
                        '02_total_time': self.get_mean_metric('total_time', self.model.patients),
                        '03_throughput': self.get_throughput(self.model.patients)
                        }
            
      
    
    def summary_frame(self):
        '''
        Returns run results as a pandas.DataFrame

        Returns:
        -------
        pd.DataFrame
        '''
        # append to results df
        if self.results is None:
            self.process_run_results()

        df = pd.DataFrame({'1': self.results})
        df = df.T
        df.index.name = 'rep'
        return df

    def detailed_logs(self):
        '''
        Returns run results as a pandas.DataFrame

        Returns:
        -------
        pd.DataFrame
        '''
        # append to results df
        if self.event_log is None:
            self.process_run_results()

        return {
            'patient': self.patient_log,
            'event_log': self.event_log,
            'results_summary': self.results
        }
