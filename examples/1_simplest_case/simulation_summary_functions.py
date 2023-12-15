import numpy as np


# Pass in 

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
        self.full_event_log = model.full_event_log
        self.utilisation_audit = model.utilisation_audit

    def process_run_results(self, 
                            wait_target_per_step=120):
        '''
        Calculates statistics at end of run.
        '''
        self.results = {}

        if self.args.model == "simplest":


            self.patient_log = self.model.patients

            mean_treat_wait = self.get_mean_metric('wait_treat', self.model.patients)

            perc_treat_wait_target_met = self.get_perc_wait_target_met('wait_treat',
                                                                       self.model.patients,
                                                                       target=wait_target_per_step)

            # triage utilisation (both types of patient)
            treat_util = self.get_resource_util('treat_duration',
                                                self.args.n_cubicles_1,
                                                self.model.patients)

            mean_total = self.get_mean_metric('total_time', self.model.patients)

            self.results = {'00_arrivals': len(self.model.patients),
                            '01a_treatment_wait': mean_treat_wait,
                            '01b_treatment_util': treat_util,
                            '01c_treatment_wait_target_met': perc_treat_wait_target_met,
                            '08_total_time': mean_total,
                            '09_throughput': self.get_throughput(self.model.patients)
                            }
            
        elif self.args.model == "simple_with_branch":

            self.patient_log = self.model.patients

            # mean waiting time for examination (non_trauma)
            mean_wait_exam = self.get_mean_metric('wait_exam',
                                                self.model.patients)

            # examination utilisation (non-trauma)
            exam_util = self.get_resource_util('exam_duration',
                                            self.args.n_exam,
                                            self.model.patients)

            mean_treat_wait = self.get_mean_metric('wait_treat', self.model.patients)

            perc_wait_exam_target_met = self.get_perc_wait_target_met('wait_exam',
                                                            self.model.patients,
                                                            target=120)

            # triage utilisation (both types of patient)
            treat_util = self.get_resource_util('treat_duration',
                                                self.args.n_cubicles_1,
                                                self.model.patients)

            mean_total = self.get_mean_metric('total_time', self.model.patients)

            self.results = {'00_arrivals': len(self.model.patients),
                            '01a_examination_wait': mean_wait_exam,
                            '01b_examination_util': exam_util,
                            '01c_examination_wait_target_met': perc_wait_exam_target_met,
                            '02a_treatment_wait': mean_treat_wait,
                            '02b_treatment_util': treat_util,
                            '08_total_time': mean_total,
                            '09_throughput': self.get_throughput(self.model.patients)
                            }
                            

        else:
        # list of all patients
            patients = self.model.non_trauma_patients + self.model.trauma_patients

            # mean triage times (both types of patient)
            mean_triage_wait = self.get_mean_metric('wait_triage', patients)

            # triage utilisation (both types of patient)
            triage_util = self.get_resource_util('triage_duration',
                                                self.args.n_triage,
                                                patients)

            # mean waiting time for registration (non_trauma)
            mean_reg_wait = self.get_mean_metric('wait_reg',
                                                self.model.non_trauma_patients)

            # registration utilisation (trauma)
            reg_util = self.get_resource_util('reg_duration',
                                            self.args.n_reg,
                                            self.model.non_trauma_patients)

            # mean waiting time for examination (non_trauma)
            mean_wait_exam = self.get_mean_metric('wait_exam',
                                                self.model.non_trauma_patients)

            # examination utilisation (non-trauma)
            exam_util = self.get_resource_util('exam_duration',
                                            self.args.n_exam,
                                            self.model.non_trauma_patients)

            # mean waiting time for treatment (non-trauma)
            mean_treat_wait = self.get_mean_metric('wait_treat',
                                                self.model.non_trauma_patients)

            # treatment utilisation (non_trauma)
            treat_util1 = self.get_resource_util('treat_duration',
                                                self.args.n_cubicles_1,
                                                self.model.non_trauma_patients)

            # mean total time (non_trauma)
            mean_total = self.get_mean_metric('total_time',
                                            self.model.non_trauma_patients)

            # mean waiting time for trauma
            mean_trauma_wait = self.get_mean_metric('wait_trauma',
                                                    self.model.trauma_patients)

            # trauma utilisation (trauma)
            trauma_util = self.get_resource_util('trauma_duration',
                                                self.args.n_trauma,
                                                self.model.trauma_patients)

            # mean waiting time for treatment (rauma)
            mean_treat_wait2 = self.get_mean_metric('wait_treat',
                                                    self.model.trauma_patients)

            # treatment utilisation (trauma)
            treat_util2 = self.get_resource_util('treat_duration',
                                                self.args.n_cubicles_2,
                                                self.model.trauma_patients)

            # mean total time (trauma)
            mean_total2 = self.get_mean_metric('total_time',
                                            self.model.trauma_patients)

            self.patient_log = patients

            self.results = {'00_arrivals': len(patients),
                            '01a_triage_wait': mean_triage_wait,
                            '01b_triage_util': triage_util,
                            '02a_registration_wait': mean_reg_wait,
                            '02b_registration_util': reg_util,
                            '03a_examination_wait': mean_wait_exam,
                            '03b_examination_util': exam_util,
                            '04a_treatment_wait(non_trauma)': mean_treat_wait,
                            '04b_treatment_util(non_trauma)': treat_util1,
                            '05_total_time(non-trauma)': mean_total,
                            '06a_trauma_wait': mean_trauma_wait,
                            '06b_trauma_util': trauma_util,
                            '07a_treatment_wait(trauma)': mean_treat_wait2,
                            '07b_treatment_util(trauma)': treat_util2,
                            '08_total_time(trauma)': mean_total2,
                            '09_throughput': self.get_throughput(patients)
                            }

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
        if self.full_event_log is None:
            self.process_run_results()

        if self.utilisation_audit is None:
            self.process_run_results()

        return {
            'patient': self.patient_log,
            'event_log': self.full_event_log,
            'utilisation_audit': self.utilisation_audit,
            'results_summary': self.results
        }

