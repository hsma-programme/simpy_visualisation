#!/usr/bin/env python
# coding: utf-8

'''
FirstTreatment: A health clinic based in the US.

A patient arrives at the treatment centre, is seen, and then leaves
'''

import itertools
import numpy as np
import pandas as pd
import simpy


from distribution_classes import (
    Exponential, Normal, Uniform, Bernoulli, Lognormal)

from simulation_utility_functions import trace, CustomResource

# Constants and defaults for modelling **as-is**

# Distribution parameters

# sign-in/triage parameters
DEFAULT_TRIAGE_MEAN = 6.0

# registration parameters
DEFAULT_REG_MEAN = 8.0
DEFAULT_REG_VAR = 2.0

# examination parameters
DEFAULT_EXAM_MEAN = 16.0
DEFAULT_EXAM_VAR = 3.0

# trauma/stabilisation
DEFAULT_TRAUMA_MEAN = 90.0

# Trauma treatment
DEFAULT_TRAUMA_TREAT_MEAN = 30.0
DEFAULT_TRAUMA_TREAT_VAR = 4.0

# Non trauma treatment
DEFAULT_NON_TRAUMA_TREAT_MEAN = 13.3
DEFAULT_NON_TRAUMA_TREAT_VAR = 2.0

# prob patient requires treatment given trauma
DEFAULT_NON_TRAUMA_TREAT_P = 0.60

# proportion of patients triaged as trauma
DEFAULT_PROB_TRAUMA = 0.12


# Time dependent arrival rates data
# The data for arrival rates varies between clinic opening at 6am and closure at
# 12am.

NSPP_PATH = ''

OVERRIDE_ARRIVAL_RATE = False
MANUAL_ARRIVAL_RATE_VALUE = 1

# Resource counts

DEFAULT_N_TRIAGE = 1
DEFAULT_N_REG = 1
DEFAULT_N_EXAM = 3
DEFAULT_N_TRAUMA = 2

# Non-trauma cubicles
DEFAULT_N_CUBICLES_1 = 1

# trauma pathway cubicles
DEFAULT_N_CUBICLES_2 = 1

# Simulation model run settings

# default random number SET
N_STREAMS = 20

# default results collection period
DEFAULT_RESULTS_COLLECTION_PERIOD = 60 * 19

# number of replications.
DEFAULT_N_REPS = 5

# Show the a trace of simulated events
# not recommended when running multiple replications
TRACE = False

# list of metrics useful for external apps
RESULT_FIELDS = ['00_arrivals',
                 '01a_triage_wait',
                 '01b_triage_util',
                 '02a_registration_wait',
                 '02b_registration_util',
                 '03a_examination_wait',
                 '03b_examination_util',
                 '04a_treatment_wait(non_trauma)',
                 '04b_treatment_util(non_trauma)',
                 '05_total_time(non-trauma)',
                 '06a_trauma_wait',
                 '06b_trauma_util',
                 '07a_treatment_wait(trauma)',
                 '07b_treatment_util(trauma)',
                 '08_total_time(trauma)',
                 '09_throughput']

# list of metrics useful for external apps
RESULT_LABELS = {'00_arrivals': 'Arrivals',
                 '01a_triage_wait': 'Triage Wait (mins)',
                 '01b_triage_util': 'Triage Utilisation',
                 '02a_registration_wait': 'Registration Waiting Time (mins)',
                 '02b_registration_util': 'Registration Utilisation',
                 '03a_examination_wait': 'Examination Waiting Time (mins)',
                 '03b_examination_util': 'Examination Utilisation',
                 '04a_treatment_wait(non_trauma)': 'Non-trauma cubicle waiting time (mins)',
                 '04b_treatment_util(non_trauma)': 'Non-trauma cubicle utilisation',
                 '05_total_time(non-trauma)': 'Total time (non-trauma)',
                 '06a_trauma_wait': 'Trauma stabilisation waiting time (mins)',
                 '06b_trauma_util': 'Trauma stabilisation utilisation',
                 '07a_treatment_wait(trauma)': 'Trauma cubicle waiting time (mins)',
                 '07b_treatment_util(trauma)': 'Trauma cubicle utilisation',
                 '08_total_time(trauma)': 'Total time (trauma)',
                 '09_throughput': 'throughput'}


# ## Model parameterisation

class Scenario:
    '''
    Container class for scenario parameters/arguments

    Passed to a model and its process classes
    '''

    def __init__(self,
                 random_number_set=1,
                 n_triage=DEFAULT_N_TRIAGE,
                 n_reg=DEFAULT_N_REG,
                 n_exam=DEFAULT_N_EXAM,
                 n_trauma=DEFAULT_N_TRAUMA,
                 n_cubicles_1=DEFAULT_N_CUBICLES_1,
                 n_cubicles_2=DEFAULT_N_CUBICLES_2,
                 triage_mean=DEFAULT_TRIAGE_MEAN,
                 reg_mean=DEFAULT_REG_MEAN,
                 reg_var=DEFAULT_REG_VAR,
                 exam_mean=DEFAULT_EXAM_MEAN,
                 exam_var=DEFAULT_EXAM_VAR,
                 trauma_mean=DEFAULT_TRAUMA_MEAN,
                 trauma_treat_mean=DEFAULT_TRAUMA_TREAT_MEAN,
                 trauma_treat_var=DEFAULT_TRAUMA_TREAT_VAR,
                 non_trauma_treat_mean=DEFAULT_NON_TRAUMA_TREAT_MEAN,
                 non_trauma_treat_var=DEFAULT_NON_TRAUMA_TREAT_VAR,
                 non_trauma_treat_p=DEFAULT_NON_TRAUMA_TREAT_P,
                 prob_trauma=DEFAULT_PROB_TRAUMA,
                 arrival_df=NSPP_PATH,
                 override_arrival_rate=OVERRIDE_ARRIVAL_RATE,
                 manual_arrival_rate=MANUAL_ARRIVAL_RATE_VALUE,
                 model="full"
                 ):
        '''
        Create a scenario to parameterise the simulation model

        Parameters:
        -----------
        random_number_set: int, optional (default=DEFAULT_RNG_SET)
            Set to control the initial seeds of each stream of pseudo
            random numbers used in the model.

        n_triage: int
            The number of triage cubicles

        n_reg: int
            The number of registration clerks

        n_exam: int
            The number of examination rooms

        n_trauma: int
            The number of trauma bays for stablisation

        n_cubicles_1: int
            The number of non-trauma treatment cubicles

        n_cubicles_2: int
            The number of trauma treatment cubicles

        triage_mean: float
            Mean duration of the triage distribution (Exponential)

        reg_mean: float
            Mean duration of the registration distribution (Lognormal)

        reg_var: float
            Variance of the registration distribution (Lognormal)

        exam_mean: float
            Mean of the examination distribution (Normal)

        exam_var: float
            Variance of the examination distribution (Normal)

        trauma_mean: float
            Mean of the trauma stabilisation distribution (Exponential)

        trauma_treat_mean: float
            Mean of the trauma cubicle treatment distribution (Lognormal)

        trauma_treat_var: float
            Variance of the trauma cubicle treatment distribution (Lognormal)

        non_trauma_treat_mean: float
            Mean of the non trauma treatment distribution

        non_trauma_treat_var: float
            Variance of the non trauma treatment distribution

        non_trauma_treat_p: float
            Probability non trauma patient requires treatment

        prob_trauma: float
            probability that a new arrival is a trauma patient.

        model: string
            What model to run. Default is full. 
            Options are "full", "simplest", "simple_with_branch"
        '''
        # sampling
        self.random_number_set = random_number_set

        # store parameters for sampling
        self.triage_mean = triage_mean
        self.reg_mean = reg_mean
        self.reg_var = reg_var
        self.exam_mean = exam_mean
        self.exam_var = exam_var
        self.trauma_mean = trauma_mean
        self.trauma_treat_mean = trauma_treat_mean
        self.trauma_treat_var = trauma_treat_var
        self.non_trauma_treat_mean = non_trauma_treat_mean
        self.non_trauma_treat_var = non_trauma_treat_var
        self.non_trauma_treat_p = non_trauma_treat_p
        self.prob_trauma = prob_trauma
        self.manual_arrival_rate = manual_arrival_rate
        self.arrival_df = arrival_df
        self.override_arrival_rate = override_arrival_rate
        self.model = model

        self.init_sampling()

        # count of each type of resource
        self.init_resource_counts(n_triage, n_reg, n_exam, n_trauma,
                                  n_cubicles_1, n_cubicles_2)

    def set_random_no_set(self, random_number_set):
        '''
        Controls the random sampling 
        Parameters:
        ----------
        random_number_set: int
            Used to control the set of psuedo random numbers
            used by the distributions in the simulation.
        '''
        self.random_number_set = random_number_set
        self.init_sampling()

    def init_resource_counts(self, n_triage, n_reg, n_exam, n_trauma,
                             n_cubicles_1, n_cubicles_2):
        '''
        Init the counts of resources to default values...
        '''
        self.n_triage = n_triage
        self.n_reg = n_reg
        self.n_exam = n_exam
        self.n_trauma = n_trauma

        # non-trauma (1), trauma (2) treatment cubicles
        self.n_cubicles_1 = n_cubicles_1
        self.n_cubicles_2 = n_cubicles_2

    def init_sampling(self):
        '''
        Create the distributions used by the model and initialise 
        the random seeds of each.
        '''
        # create random number streams
        rng_streams = np.random.default_rng(self.random_number_set)
        self.seeds = rng_streams.integers(0, 999999999, size=N_STREAMS)

        # create distributions

        # Triage duration
        self.triage_dist = Exponential(self.triage_mean,
                                       random_seed=self.seeds[0])

        # Registration duration (non-trauma only)
        self.reg_dist = Lognormal(self.reg_mean,
                                  np.sqrt(self.reg_var),
                                  random_seed=self.seeds[1])

        # Evaluation (non-trauma only)
        self.exam_dist = Normal(self.exam_mean,
                                np.sqrt(self.exam_var),
                                random_seed=self.seeds[2])

        # Trauma/stablisation duration (trauma only)
        self.trauma_dist = Exponential(self.trauma_mean,
                                       random_seed=self.seeds[3])

        # Non-trauma treatment
        self.nt_treat_dist = Lognormal(self.non_trauma_treat_mean,
                                       np.sqrt(self.non_trauma_treat_var),
                                       random_seed=self.seeds[4])

        # treatment of trauma patients
        self.treat_dist = Lognormal(self.trauma_treat_mean,
                                    np.sqrt(self.non_trauma_treat_var),
                                    random_seed=self.seeds[5])

        # probability of non-trauma patient requiring treatment
        self.nt_p_treat_dist = Bernoulli(self.non_trauma_treat_p,
                                         random_seed=self.seeds[6])

        # probability of non-trauma versus trauma patient
        self.p_trauma_dist = Bernoulli(self.prob_trauma,
                                       random_seed=self.seeds[7])

        # init sampling for non-stationary poisson process
        self.init_nspp()

    def init_nspp(self):

        # read arrival profile
        self.arrivals = pd.read_csv(NSPP_PATH)  # pylint: disable=attribute-defined-outside-init
        self.arrivals['mean_iat'] = 60 / self.arrivals['arrival_rate']

        # maximum arrival rate (smallest time between arrivals)
        self.lambda_max = self.arrivals['arrival_rate'].max()  # pylint: disable=attribute-defined-outside-init

        # thinning exponential
        if self.override_arrival_rate is True:

            self.arrival_dist = Exponential(self.manual_arrival_rate,  # pylint: disable=attribute-defined-outside-init
                                            random_seed=self.seeds[8])
        else:
            self.arrival_dist = Exponential(60.0 / self.lambda_max,  # pylint: disable=attribute-defined-outside-init
                                            random_seed=self.seeds[8])

            # thinning uniform rng
            self.thinning_rng = Uniform(low=0.0, high=1.0,  # pylint: disable=attribute-defined-outside-init
                                        random_seed=self.seeds[9])

# ## Patient Pathways Process Logic

class SimplePathway(object):
    '''
    Encapsulates the process for a patient with minor injuries and illness.

    These patients are arrived, then seen and treated by a nurse as soon as one is available.
    No place-based resources are considered in this pathway.

    Following treatment they are discharged.
    '''

    def __init__(self, identifier, env, args, full_event_log):
        '''
        Constructor method

        Params:
        -----
        identifier: int
            a numeric identifier for the patient.

        env: simpy.Environment
            the simulation environment

        args: Scenario
            Container class for the simulation parameters

        '''
        self.identifier = identifier
        self.env = env
        self.args = args
        self.full_event_log = full_event_log

        # metrics
        self.arrival = -np.inf
        self.wait_treat = -np.inf
        self.total_time = -np.inf

        self.treat_duration = -np.inf

    def execute(self):
        '''
        simulates the simplest minor treatment process for a patient

        1. Arrive
        2. Examined/treated by nurse when one available
        3. Discharged
        '''
        # record the time of arrival and entered the triage queue
        self.arrival = self.env.now
        self.full_event_log.append(
            {'patient': self.identifier,
             'pathway': 'Simplest',
             'event_type': 'arrival_departure',
             'event': 'arrival',
             'time': self.env.now}
        )

        # request examination resource
        start_wait = self.env.now
        self.full_event_log.append(
            {'patient': self.identifier,
             'pathway': 'Simplest',
             'event': 'treatment_wait_begins',
             'event_type': 'queue',
             'time': self.env.now}
        )

        # Seize a treatment resource when available
        treatment_resource = yield self.args.treatment.get()
            
        # record the waiting time for registration
        self.wait_treat = self.env.now - start_wait
        self.full_event_log.append(
            {'patient': self.identifier,
                'pathway': 'Simplest',
                'event': 'treatment_begins',
                'event_type': 'resource_use',
                'time': self.env.now,
                'resource_id': treatment_resource.id_attribute
                }
        )

        # sample examination duration.
        self.treat_duration = self.args.treat_dist.sample()
        yield self.env.timeout(self.treat_duration)
        
        self.full_event_log.append(
            {'patient': self.identifier,
                'pathway': 'Simplest',
                'event': 'treatment_complete',
                'event_type': 'resource_use_end',
                'time': self.env.now,
                'resource_id': treatment_resource.id_attribute}
        )
    
        # Resource is no longer in use, so put it back in
        self.args.treatment.put(treatment_resource) 

        # total time in system
        self.total_time = self.env.now - self.arrival
        self.full_event_log.append(
            {'patient': self.identifier,
            'pathway': 'Simplest',
            'event': 'depart',
            'event_type': 'arrival_departure',
            'time': self.env.now}
        )



class TreatmentCentreModelSimpleNurseStepOnly:
    '''
    The treatment centre model

    Patients arrive at random to a treatment centre, see a nurse, then leave.

    The main class that a user interacts with to run the model is
    `TreatmentCentreModel`.  This implements a `.run()` method, contains a simple
    algorithm for the non-stationary poission process for patients arrivals and
    inits instances of the nurse pathway.

    '''

    def __init__(self, args):
        self.env = simpy.Environment()
        self.args = args
        self.init_resources()

        self.patients = []

        self.rc_period = None
        self.results = None

        self.full_event_log = []
        self.utilisation_audit = []

    def init_resources(self):
        '''
        Init the number of resources
        and store in the arguments container object

        Resource list:
            1. Nurses/treatment bays (same thing in this model)

        '''
        # examination
        # self.args.treatment = CustomResource(self.env,
        #                                 capacity=self.args.n_cubicles_1)
        
        self.args.treatment = simpy.Store(self.env)

        for i in range(self.args.n_cubicles_1):
            self.args.treatment.put(
                CustomResource(
                    self.env,
                    capacity=1,
                    id_attribute = i+1)
                )



    def run(self, results_collection_period=DEFAULT_RESULTS_COLLECTION_PERIOD):
        '''
        Conduct a single run of the model in its current
        configuration


        Parameters:
        ----------
        results_collection_period, float, optional
            default = DEFAULT_RESULTS_COLLECTION_PERIOD

        warm_up, float, optional (default=0)

            length of initial transient period to truncate
            from results.

        Returns:
        --------
            None
        '''
        # setup the arrival generator process
        self.env.process(self.arrivals_generator())

        # resources_list = [
        #     {'resource_name': 'treatment_cubicle_or_nurse',
        #         'resource_object': self.args.n_cubicles_1}
        # ]

        # self.env.process(
        #     self.interval_audit_utilisation(
        #         resources=resources_list,
        #         interval=5
        #     )
        # )

        # store rc perio
        self.rc_period = results_collection_period

        # run
        self.env.run(until=results_collection_period)

    def interval_audit_utilisation(self, resources, interval=1):
        '''
        Record utilisation at defined intervals. 

        Needs to be passed to env.process when running model

        Parameters:
        ------
        resource: SimPy resource object
            The resource to monitor
            OR 
            a list of dictionaries containing simpy resource objects in the format
            [{'resource_name':'my_resource', 'resource_object': resource}]

        interval: int:
            Time between audits. 
            1 unit of time is 1 day in this model.  
        '''

        while True:
            # Record time
            if isinstance(resources, list):
                for i in range(len(resources)):
                    self.utilisation_audit.append({
                        'resource_name': resources[i]['resource_name'],
                        'simulation_time': self.env.now,  # The current simulation time
                        # The number of users
                        'number_utilised': resources[i]['resource_object'].count,
                        'number_available': resources[i]['resource_object'].capacity,
                        # The number of queued processes
                        'number_queued': len(resources[i]['resource_object'].queue),
                    })

            else:
                self.utilisation_audit.append({
                    # 'simulation_time': resource._env.now,
                    'simulation_time': self.env.now,  # The current simulation time
                    'number_utilised': resources.count,  # The number of users
                    'number_available': resources.capacity,
                    # The number of queued processes
                    'number_queued': len(resources.queue),
                })

            # Trigger next audit after interval
            yield self.env.timeout(interval)

    def arrivals_generator(self):
        '''
        Simulate the arrival of patients to the model

        Patients follow the SimplePathway process.

        Non stationary arrivals implemented via Thinning acceptance-rejection
        algorithm.
        '''
        for patient_count in itertools.count():

            # this give us the index of dataframe to use
            t = int(self.env.now // 60) % self.args.arrivals.shape[0]
            lambda_t = self.args.arrivals['arrival_rate'].iloc[t]

            # set to a large number so that at least 1 sample taken!
            u = np.Inf

            interarrival_time = 0.0

            if self.args.override_arrival_rate:
                interarrival_time += self.args.arrival_dist.sample()
            else:
            # reject samples if u >= lambda_t / lambda_max
                while u >= (lambda_t / self.args.lambda_max):
                    interarrival_time += self.args.arrival_dist.sample()
                    u = self.args.thinning_rng.sample()

            # iat
            yield self.env.timeout(interarrival_time)

            trace(f'patient {patient_count} arrives at: {self.env.now:.3f}')
            # self.full_event_log.append(
            #     {'patient': patient_count,
            #      'pathway': 'Simplest',
            #      'event': 'arrival',
            #      'event_type': 'arrival_departure',
            #      'time': self.env.now}
            # )

            # Generate the patient
            new_patient = SimplePathway(patient_count, self.env, self.args, self.full_event_log)
            self.patients.append(new_patient)
            # start the pathway process for the patient
            self.env.process(new_patient.execute())
