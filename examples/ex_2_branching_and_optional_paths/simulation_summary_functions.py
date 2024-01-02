import numpy as np
import pandas as pd

# list of metrics useful for external apps
# RESULT_FIELDS = ['00_arrivals',
#                  '01a_triage_wait',
#                  '01b_triage_util',
#                  '02a_registration_wait',
#                  '02b_registration_util',
#                  '03a_examination_wait',
#                  '03b_examination_util',
#                  '04a_treatment_wait(non_trauma)',
#                  '04b_treatment_util(non_trauma)',
#                  '05_total_time(non-trauma)',
#                  '06a_trauma_wait',
#                  '06b_trauma_util',
#                  '07a_treatment_wait(trauma)',
#                  '07b_treatment_util(trauma)',
#                  '08_total_time(trauma)',
#                  '09_throughput']

# # list of metrics useful for external apps
# RESULT_LABELS = {'00_arrivals': 'Arrivals',
#                  '01a_triage_wait': 'Triage Wait (mins)',
#                  '01b_triage_util': 'Triage Utilisation',
#                  '02a_registration_wait': 'Registration Waiting Time (mins)',
#                  '02b_registration_util': 'Registration Utilisation',
#                  '03a_examination_wait': 'Examination Waiting Time (mins)',
#                  '03b_examination_util': 'Examination Utilisation',
#                  '04a_treatment_wait(non_trauma)': 'Non-trauma cubicle waiting time (mins)',
#                  '04b_treatment_util(non_trauma)': 'Non-trauma cubicle utilisation',
#                  '05_total_time(non-trauma)': 'Total time (non-trauma)',
#                  '06a_trauma_wait': 'Trauma stabilisation waiting time (mins)',
#                  '06b_trauma_util': 'Trauma stabilisation utilisation',
#                  '07a_treatment_wait(trauma)': 'Trauma cubicle waiting time (mins)',
#                  '07b_treatment_util(trauma)': 'Trauma cubicle utilisation',
#                  '08_total_time(trauma)': 'Total time (trauma)',
#                  '09_throughput': 'throughput'}


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


    def process_run_results(self):
        '''
        Calculates statistics at end of run.
        '''
        self.results = {}

        patients = self.model.non_trauma_patients + self.model.trauma_patients
        
        # self.patient_log = self.model.patients

        self.results = {
            '00_arrivals': len(patients),
            
            '01a_triage_wait': self.get_mean_metric('wait_triage', 
                                                    patients),
            '01b_triage_util': self.get_resource_util('triage_duration', 
                                                      self.args.n_triage,
                                                      patients),
            # '01c_triage_wait_target_met': self.get_perc_wait_target_met('wait_triage', 
            #                                                             self.model.patients,
            #                                                             target=10),

            '02a_reg_wait': self.get_mean_metric('wait_reg', 
                                                    self.model.non_trauma_patients),
            '02b_reg_util': self.get_resource_util('reg_duration', 
                                                      self.args.n_reg,
                                                      self.model.non_trauma_patients),
            # '02c_reg_wait_target_met': self.get_perc_wait_target_met('wait_reg', 
            #                                                             self.model.non_trauma_patients,
            #                                                             target=60),

            '03a_exam_wait': self.get_mean_metric('wait_exam', 
                                                    self.model.non_trauma_patients),
            '03b_exam_util': self.get_resource_util('exam_duration', 
                                                      self.args.n_exam,
                                                      self.model.non_trauma_patients),
            # '03c_exam_wait_target_met': self.get_perc_wait_target_met('wait_reg', 
            #                                                             self.model.non_trauma_patients,
            #                                                             target=60),

            '04a_non_trauma_treat_wait': self.get_mean_metric('wait_treat', 
                                                    self.model.non_trauma_patients),
            '04b_non_trauma_treat_util': self.get_resource_util('treat_duration', 
                                                      self.args.n_cubicles_1,
                                                      self.model.non_trauma_patients),
            # '04c_non_trauma_treat_wait_target_met': self.get_perc_wait_target_met('wait_treat', 
            #                                                             self.model.non_trauma_patients,
            #                                                             target=60),

            '05a_trauma_stabilisation_wait': self.get_mean_metric('wait_treat', 
                                                    self.model.trauma_patients),
            '05b_trauma_stabilisation_util': self.get_resource_util('treat_duration', 
                                                      self.args.n_trauma,
                                                      self.model.trauma_patients),
            # '05c_trauma_stabilisation_wait_target_met': self.get_perc_wait_target_met('wait_treat', 
            #                                                             self.model.trauma_patients,
            #                                                             target=60),

            '06a_trauma_treat_wait': self.get_mean_metric('wait_treat', 
                                                    self.model.trauma_patients),
            '06b_trauma_treat_util': self.get_resource_util('treat_duration', 
                                                      self.args.n_cubicles_2,
                                                      self.model.trauma_patients),
            # '06c_trauma_treat_wait_target_met': self.get_perc_wait_target_met('wait_treat', 
            #                                                             self.model.trauma_patients,
            #                                                             target=60),
            
            '07_total_time': self.get_mean_metric('total_time', self.model.patients),
            '08_throughput': self.get_throughput(self.model.patients)
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
