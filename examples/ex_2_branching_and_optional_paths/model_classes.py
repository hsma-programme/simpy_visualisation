#!/usr/bin/env python
# coding: utf-8

'''
FirstTreatment: A health clinic based in the US.

This example is based on exercise 13 from Nelson (2013) page 170.
 
Nelson. B.L. (2013). Foundations and methods of stochastic simulation.
Patients arrive to the health clinic between 6am and 12am following a 
non-stationary poisson process. After 12am arriving patients are diverted 
elsewhere and remaining WIP is completed.  
On arrival, all patients quickly sign-in and are triaged.   

The health clinic expects two types of patient arrivals: 

**Trauma arrivals:**
patients with severe illness and trauma that must first be stablised in a 
trauma room. These patients then undergo treatment in a cubicle before being 
discharged.

**Non-trauma arrivals**
patients with minor illness and no trauma go through registration and 
examination activities. A proportion of non-trauma patients require treatment
in a cubicle before being dicharged. 

In this model treatment of trauma and non-trauma patients is modelled seperately 
'''

import itertools
import numpy as np
import pandas as pd
import simpy

from distribution_classes import (
    Exponential, Normal, Uniform, Bernoulli, Lognormal)

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

NSPP_PATH = 'resources/ed_arrivals.csv'

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


# Utility functions

def trace(msg):
    '''
    Utility function for printing a trace as the
    simulation model executes.
    Set the TRACE constant to False, to turn tracing off.

    Params:
    -------
    msg: str
        string to print to screen.
    '''
    if TRACE:
        print(msg)

class CustomResource(simpy.Resource):
    def __init__(self, env, capacity, id_attribute=None):
        super().__init__(env, capacity)
        self.id_attribute = id_attribute

    def request(self, *args, **kwargs):
        # Add logic to handle the ID attribute when a request is made
        # For example, you can assign an ID to the requester
        # self.id_attribute = assign_id_logic()
        return super().request(*args, **kwargs)

    def release(self, *args, **kwargs):
        # Add logic to handle the ID attribute when a release is made
        # For example, you can reset the ID attribute
        # reset_id_logic(self.id_attribute)
        return super().release(*args, **kwargs)
# def patch_resource(resource, pre=None, post=None):
#     """
#     Part of the required code for event-based auditing of resources (so records each time
#     utilisation of a resource changes in some way, as opposed to recording after
#     defined time intervals)

#     Patch *resource* so that it calls the callable *pre* before each
#      put/get/request/release operation and the callable *post* after each
#      operation.  The only argument to these functions is the resource
#      instance.

#      From
#      https://simpy.readthedocs.io/en/4.0.1/topical_guides/monitoring.html?highlight=monitor#resource-usage

#     Used for event-based monitoring (as opposed to interval-based monitoring)

#      """

#     def get_wrapper(func):
#         # Generate a wrapper for put/get/request/release
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             # This is the actual wrapper
#             # Call "pre" callback
#             if pre:
#                 pre(resource)

#             # Perform actual operation
#             ret = func(*args, **kwargs)

#             # Call "post" callback
#             if post:
#                 post(resource)

#             return ret
#         return wrapper

#     # Replace the original operations with our wrapper
#     for name in ['put', 'get', 'request', 'release']:
#         if hasattr(resource, name):
#             setattr(resource, name, get_wrapper(getattr(resource, name)))

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


class TraumaPathway:
    '''
    Encapsulates the process a patient with severe injuries or illness.

    These patients are signed into the ED and triaged as having severe injuries
    or illness.

    Patients are stabilised in resus (trauma) and then sent to Treatment.  
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
        self.wait_triage = -np.inf
        self.wait_trauma = -np.inf
        self.wait_treat = -np.inf
        self.total_time = -np.inf

        self.triage_duration = -np.inf
        self.trauma_duration = -np.inf
        self.treat_duration = -np.inf

    def execute(self):
        '''
        simulates the major treatment process for a patient

        1. request and wait for sign-in/triage
        2. trauma
        3. treatment
        '''
        # record the time of arrival and entered the triage queue
        self.arrival = self.env.now
        self.full_event_log.append(
            {'patient': self.identifier,
             'pathway': 'Trauma',
             'event_type': 'queue',
             'event': 'triage_wait_begins',
             'time': self.env.now}
        )

        ###################################################
        # request sign-in/triage
        triage_resource = yield self.args.triage.get()

        # record the waiting time for triage
        self.wait_triage = self.env.now - self.arrival

        trace(f'patient {self.identifier} triaged to trauma '
                f'{self.env.now:.3f}')
        self.full_event_log.append(
            {'patient': self.identifier,
             'pathway': 'Trauma',
             'event_type': 'resource_use',
             'event': 'triage_begins',
             'time': self.env.now,
             'resource_id': triage_resource.id_attribute
            }
        )

        # sample triage duration.
        self.triage_duration = self.args.triage_dist.sample()
        yield self.env.timeout(self.triage_duration)
        
        trace(f'triage {self.identifier} complete {self.env.now:.3f}; '
              f'waiting time was {self.wait_triage:.3f}')
        self.full_event_log.append(
            {'patient': self.identifier,
             'pathway': 'Trauma',
             'event_type': 'resource_use_end',
             'event': 'triage_complete',
             'time': self.env.now,
             'resource_id': triage_resource.id_attribute}
        )

        # Resource is no longer in use, so put it back in the store 
        self.args.triage.put(triage_resource) 
        ###################################################

        # record the time that entered the trauma queue
        start_wait = self.env.now
        self.full_event_log.append(
            {'patient': self.identifier,
             'pathway': 'Trauma',
             'event_type': 'queue',
             'event': 'TRAUMA_stabilisation_wait_begins',
             'time': self.env.now}
        )

        ###################################################
        # request trauma room
        trauma_resource = yield self.args.trauma.get()

        self.full_event_log.append(
            {'patient': self.identifier,
                'pathway': 'Trauma',
                'event_type': 'resource_use',
                'event': 'TRAUMA_stabilisation_begins',
                'time': self.env.now,
                'resource_id': trauma_resource.id_attribute
                }
        )

        # record the waiting time for trauma
        self.wait_trauma = self.env.now - start_wait

        # sample stablisation duration.
        self.trauma_duration = self.args.trauma_dist.sample()
        yield self.env.timeout(self.trauma_duration)

        trace(f'stabilisation of patient {self.identifier} at '
              f'{self.env.now:.3f}')
        self.full_event_log.append(
            {'patient': self.identifier,
             'pathway': 'Trauma',
             'event_type': 'resource_use_end',
             'event': 'TRAUMA_stabilisation_complete',
             'time': self.env.now,
             'resource_id': trauma_resource.id_attribute
            }
        )
        # Resource is no longer in use, so put it back in the store
        self.args.trauma.put(trauma_resource)
        
        #######################################################

        # record the time that entered the treatment queue
        start_wait = self.env.now
        self.full_event_log.append(
            {'patient': self.identifier,
             'pathway': 'Trauma',
             'event_type': 'queue',
             'event': 'TRAUMA_treatment_wait_begins',
             'time': self.env.now}
        )

        ########################################################
        # request treatment cubicle
        trauma_treatment_resource = yield self.args.cubicle_2.get()

        # record the waiting time for trauma
        self.wait_treat = self.env.now - start_wait
        trace(f'treatment of patient {self.identifier} at '
                f'{self.env.now:.3f}')
        self.full_event_log.append(
            {'patient': self.identifier,
                'pathway': 'Trauma',
                'event_type': 'resource_use',
                'event': 'TRAUMA_treatment_begins',
                'time': self.env.now,
                'resource_id': trauma_treatment_resource.id_attribute
                }
        )

        # sample treatment duration.
        self.treat_duration = self.args.trauma_dist.sample()
        yield self.env.timeout(self.treat_duration)

        trace(f'patient {self.identifier} treatment complete {self.env.now:.3f}; '
              f'waiting time was {self.wait_treat:.3f}')
        self.full_event_log.append(
            {'patient': self.identifier,
             'pathway': 'Trauma',
             'event_type': 'resource_use_end',
             'event': 'TRAUMA_treatment_complete',
             'time': self.env.now,
             'resource_id': trauma_treatment_resource.id_attribute}
        )
        self.full_event_log.append(
            {'patient': self.identifier,
            'pathway': 'Shared',
            'event': 'depart',
            'event_type': 'arrival_departure',
            'time': self.env.now}
        )

        # Resource is no longer in use, so put it back in the store
        self.args.cubicle_2.put(trauma_treatment_resource) 

        #########################################################

        # total time in system
        self.total_time = self.env.now - self.arrival

class NonTraumaPathway(object):
    '''
    Encapsulates the process a patient with minor injuries and illness.

    These patients are signed into the ED and triaged as having minor 
    complaints and streamed to registration and then examination. 

    Post examination 40% are discharged while 60% proceed to treatment.  
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

        # triage resource
        self.triage = args.triage

        # metrics
        self.arrival = -np.inf
        self.wait_triage = -np.inf
        self.wait_reg = -np.inf
        self.wait_exam = -np.inf
        self.wait_treat = -np.inf
        self.total_time = -np.inf

        self.triage_duration = -np.inf
        self.reg_duration = -np.inf
        self.exam_duration = -np.inf
        self.treat_duration = -np.inf

    def execute(self):
        '''
        simulates the non-trauma/minor treatment process for a patient

        1. request and wait for sign-in/triage
        2. patient registration
        3. examination
        4.1 40% discharged
        4.2 60% treatment then discharge
        '''
        # record the time of arrival and entered the triage queue
        self.arrival = self.env.now
        self.full_event_log.append(
            {'patient': self.identifier,
             'pathway': 'Non-Trauma',
             'event_type': 'queue',
             'event': 'triage_wait_begins',
             'time': self.env.now}
        )

        ###################################################
        # request sign-in/triage
        triage_resource = yield self.args.triage.get()

        # record the waiting time for triage
        self.wait_triage = self.env.now - self.arrival
        trace(f'patient {self.identifier} triaged to minors '
                f'{self.env.now:.3f}')
        self.full_event_log.append(
            {'patient': self.identifier,
                'pathway': 'Non-Trauma',
                'event_type': 'resource_use',
                'event': 'triage_begins',
                'time': self.env.now,
                'resource_id': triage_resource.id_attribute
                }
        )

        # sample triage duration.
        self.triage_duration = self.args.triage_dist.sample()
        yield self.env.timeout(self.triage_duration)

        trace(f'triage {self.identifier} complete {self.env.now:.3f}; '
                f'waiting time was {self.wait_triage:.3f}')
        self.full_event_log.append(
            {'patient': self.identifier,
                'pathway': 'Non-Trauma',
                'event_type': 'resource_use_end',
                'event': 'triage_complete',
                'time': self.env.now,
                'resource_id': triage_resource.id_attribute}
        )

        # Resource is no longer in use, so put it back in the store 
        self.args.triage.put(triage_resource) 
        #########################################################

        # record the time that entered the registration queue
        start_wait = self.env.now
        self.full_event_log.append(
            {'patient': self.identifier,
             'pathway': 'Non-Trauma',
             'event_type': 'queue',
             'event': 'MINORS_registration_wait_begins',
             'time': self.env.now}
        )

        #########################################################
        # request registration clerk
        registration_resource = yield self.args.registration.get()
            
        # record the waiting time for registration
        self.wait_reg = self.env.now - start_wait
        trace(f'registration of patient {self.identifier} at '
                f'{self.env.now:.3f}')
        self.full_event_log.append(
            {'patient': self.identifier,
                'pathway': 'Non-Trauma',
                'event_type': 'resource_use',
                'event': 'MINORS_registration_begins',
                'time': self.env.now,
                'resource_id': registration_resource.id_attribute
                }
        )

        # sample registration duration.
        self.reg_duration = self.args.reg_dist.sample()
        yield self.env.timeout(self.reg_duration)

        trace(f'patient {self.identifier} registered at'
                f'{self.env.now:.3f}; '
                f'waiting time was {self.wait_reg:.3f}')
        self.full_event_log.append(
            {'patient': self.identifier,
                'pathway': 'Non-Trauma',
                'event': 'MINORS_registration_complete',
                'event_type': 'resource_use_end',
                'time': self.env.now,
                'resource_id': registration_resource.id_attribute}
        )
        # Resource is no longer in use, so put it back in the store
        self.args.registration.put(registration_resource)
        ########################################################

        # record the time that entered the evaluation queue
        start_wait = self.env.now

        self.full_event_log.append(
            {'patient': self.identifier,
             'pathway': 'Non-Trauma',
             'event': 'MINORS_examination_wait_begins',
             'event_type': 'queue',
             'time': self.env.now}
        )

        #########################################################
        # request examination resource
        examination_resource = yield self.args.exam.get()

        # record the waiting time for examination to begin
        self.wait_exam = self.env.now - start_wait
        trace(f'examination of patient {self.identifier} begins '
                f'{self.env.now:.3f}')
        self.full_event_log.append(
            {'patient': self.identifier,
                'pathway': 'Non-Trauma',
                'event': 'MINORS_examination_begins',
                'event_type': 'resource_use',
                'time': self.env.now,
                'resource_id': examination_resource.id_attribute
                }
        )

        # sample examination duration.
        self.exam_duration = self.args.exam_dist.sample()
        yield self.env.timeout(self.exam_duration)

        trace(f'patient {self.identifier} examination complete '
                f'at {self.env.now:.3f};'
                f'waiting time was {self.wait_exam:.3f}')
        self.full_event_log.append(
            {'patient': self.identifier,
                'pathway': 'Non-Trauma',
                'event': 'MINORS_examination_complete',
                'event_type': 'resource_use_end',
                'time': self.env.now,
                'resource_id': examination_resource.id_attribute}
        )
        # Resource is no longer in use, so put it back in
        self.args.exam.put(examination_resource) 
        ############################################################################

        # sample if patient requires treatment?
        self.require_treat = self.args.nt_p_treat_dist.sample()  #pylint: disable=attribute-defined-outside-init

        if self.require_treat:

            self.full_event_log.append(
                {'patient': self.identifier,
                 'pathway': 'Non-Trauma',
                 'event': 'requires_treatment',
                 'event_type': 'attribute_assigned',
                 'time': self.env.now}
            )

            # record the time that entered the treatment queue
            start_wait = self.env.now
            self.full_event_log.append(
                {'patient': self.identifier,
                 'pathway': 'Non-Trauma',
                 'event': 'MINORS_treatment_wait_begins',
                 'event_type': 'queue',
                 'time': self.env.now}
            )
            ###################################################
            # request treatment cubicle

            non_trauma_treatment_resource = yield self.args.cubicle_1.get()

            # record the waiting time for treatment
            self.wait_treat = self.env.now - start_wait
            trace(f'treatment of patient {self.identifier} begins '
                    f'{self.env.now:.3f}')
            self.full_event_log.append(
                {'patient': self.identifier,
                    'pathway': 'Non-Trauma',
                    'event': 'MINORS_treatment_begins',
                    'event_type': 'resource_use',
                    'time': self.env.now,
                    'resource_id': non_trauma_treatment_resource.id_attribute
                }
            )

            # sample treatment duration.
            self.treat_duration = self.args.nt_treat_dist.sample()
            yield self.env.timeout(self.treat_duration)

            trace(f'patient {self.identifier} treatment complete '
                    f'at {self.env.now:.3f};'
                    f'waiting time was {self.wait_treat:.3f}')
            self.full_event_log.append(
                {'patient': self.identifier,
                    'pathway': 'Non-Trauma',
                    'event': 'MINORS_treatment_ends',
                    'event_type': 'resource_use_end',
                    'time': self.env.now,
                    'resource_id': non_trauma_treatment_resource.id_attribute}
            )

            # Resource is no longer in use, so put it back in the store
            self.args.cubicle_1.put(non_trauma_treatment_resource)
        ##########################################################################

        # Return to what happens to all patients, regardless of whether they were sampled as needing treatment
        self.full_event_log.append(
            {'patient': self.identifier,
            'pathway': 'Shared',
            'event': 'depart',
            'event_type': 'arrival_departure',
            'time': self.env.now}
        )

        # total time in system
        self.total_time = self.env.now - self.arrival


class TreatmentCentreModel:
    '''
    The treatment centre model

    Patients arrive at random to a treatment centre, are triaged
    and then processed in either a trauma or non-trauma pathway.

    The main class that a user interacts with to run the model is 
    `TreatmentCentreModel`.  This implements a `.run()` method, contains a simple
    algorithm for the non-stationary poission process for patients arrivals and 
    inits instances of `TraumaPathway` or `NonTraumaPathway` depending on the 
    arrival type.    

    '''

    def __init__(self, args):
        self.env = simpy.Environment()
        self.args = args
        self.init_resources()

        self.patients = []
        self.trauma_patients = []
        self.non_trauma_patients = []

        self.rc_period = None
        self.results = None

        self.full_event_log = []
        self.utilisation_audit = []

    def init_resources(self):
        '''
        Init the number of resources
        and store in the arguments container object

        Resource list:
        1. Sign-in/triage bays
        2. registration clerks
        3. examination bays
        4. trauma bays
        5. non-trauma cubicles (1)
        6. trauma cubicles (2)

        '''

        # sign/in triage
        # self.args.triage = CustomResource(self.env,
        #                                   capacity=self.args.n_triage)
        
        self.args.triage = simpy.Store(self.env)

        for i in range(self.args.n_triage):
            self.args.triage.put(
                CustomResource(
                    self.env,
                    capacity=1,
                    id_attribute = i+1)
                )

        # registration
        # self.args.registration = CustomResource(self.env,
        #                                         capacity=self.args.n_reg)

        self.args.registration = simpy.Store(self.env)

        for i in range(self.args.n_reg):
            self.args.registration.put(
                CustomResource(
                    self.env,
                    capacity=1,
                    id_attribute = i+1)
                )

        # examination
        # self.args.exam = CustomResource(self.env,
        #                                 capacity=self.args.n_exam)
        
        self.args.exam = simpy.Store(self.env)

        for i in range(self.args.n_exam):
            self.args.exam.put(
                CustomResource(
                    self.env,
                    capacity=1,
                    id_attribute = i+1)
                )

        # trauma
        # self.args.trauma = CustomResource(self.env,
        #                                   capacity=self.args.n_trauma)
        
        self.args.trauma = simpy.Store(self.env)

        for i in range(self.args.n_trauma):
            self.args.trauma.put(
                CustomResource(
                    self.env,
                    capacity=1,
                    id_attribute = i+1)
                )

        # non-trauma treatment
        # self.args.cubicle_1 = CustomResource(self.env,
        #                                      capacity=self.args.n_cubicles_1)
        
        self.args.cubicle_1 = simpy.Store(self.env)

        for i in range(self.args.n_cubicles_1):
            self.args.cubicle_1.put(
                CustomResource(
                    self.env,
                    capacity=1,
                    id_attribute = i+1)
                )

        # trauma treatment
        # self.args.cubicle_2 = CustomResource(self.env,
        #                                      capacity=self.args.n_cubicles_2)
        
        self.args.cubicle_2 = simpy.Store(self.env)

        for i in range(self.args.n_cubicles_2):
            self.args.cubicle_2.put(
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

        # setup the resource montioring process

        # self.env.process(
        #     self.interval_audit_utilisation(
        #         resources=self.args.triage,
        #         interval=1
        #     )
        # )

        resources_list = [
            {'resource_name': 'registration_clerks',
                'resource_object': self.args.registration},
            {'resource_name': 'triage_bays', 'resource_object': self.args.triage},

            {'resource_name': 'examination_bays',
                'resource_object': self.args.exam},
            {'resource_name': 'non_trauma_treatment_cubicle_type_1',
                'resource_object': self.args.cubicle_1},

            {'resource_name': 'trauma_bays', 'resource_object': self.args.trauma},
            {'resource_name': 'trauma_treatmentcubicle_type_2',
                'resource_object': self.args.cubicle_2}
        ]

        self.env.process(
            self.interval_audit_utilisation(
                resources=resources_list,
                interval=5
            )
        )

        # store rc period
        self.rc_period = results_collection_period

        # run
        self.env.run(until=results_collection_period)

        # self.utilisation_audit.append(

        #         {'resource': 'triage',
        #         'time': self.env.now,
        #         'capacity': self.args.triage.capacity,
        #         'usage': self.args.triage.count
        #         }

        #     )

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
                        'number_utilised': len(resources[i]['resource_object'].items),
                        'number_available': resources[i]['resource_object'].capacity,
                        # The number of queued processes
                        # 'number_queued': len(resources[i]['resource_object'].queue),
                    })

            else:
                self.utilisation_audit.append({
                    # 'simulation_time': resource._env.now,
                    'simulation_time': self.env.now,  # The current simulation time
                    'number_utilised': len(resources.items),  # The number of users
                    'number_available': resources.capacity,
                    # The number of queued processes
                    # 'number_queued': len(resources.queue),
                })

            # Trigger next audit after interval
            yield self.env.timeout(interval)

    def arrivals_generator(self):
        ''' 
        Simulate the arrival of patients to the model

        Patients either follow a TraumaPathway or
        NonTraumaPathway simpy process.

        Non stationary arrivals implemented via Thinning acceptance-rejection 
        algorithm.
        '''
        # while self.env.now < self.rc_period:
        # if int(round(self.env.now,0)) % 10 == 0:
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
            self.full_event_log.append(
                {'patient': patient_count,
                 'pathway': 'Shared',
                 'event': 'arrival',
                 'event_type': 'arrival_departure',
                 'time': self.env.now}
            )

            # sample if the patient is trauma or non-trauma
            trauma = self.args.p_trauma_dist.sample()

            if trauma:
                # create and store a trauma patient to update KPIs.
                new_patient = TraumaPathway(
                    patient_count, self.env, self.args, self.full_event_log)
                self.trauma_patients.append(new_patient)
            else:
                # create and store a non-trauma patient to update KPIs.
                new_patient = NonTraumaPathway(patient_count, self.env,
                                               self.args, self.full_event_log)
                self.non_trauma_patients.append(new_patient)

            # start the pathway process for the patient
            self.env.process(new_patient.execute())


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

    def process_run_results(self):
        '''
        Calculates statistics at end of run.
        '''
        self.results = {}

        if self.args.model == "simplest":


            self.patient_log = self.model.patients

            mean_treat_wait = self.get_mean_metric('wait_treat', self.model.patients)

            perc_treat_wait_target_met = self.get_perc_wait_target_met('wait_treat',
                                                                       self.model.patients,
                                                                       target=120)

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

    # def get_full_event_log(self):
    #     '''
    #     Returns run results as a pandas.DataFrame

    #     Returns:
    #     -------
    #     pd.DataFrame
    #     '''
    #     # append to results df

    #     df = pd.DataFrame(self.full_event_log)
    #     #df = df.T
    #     #df.index.name = 'rep'
    #     return df


# ## Executing a model

def single_run(scenario, rc_period=DEFAULT_RESULTS_COLLECTION_PERIOD,
               random_no_set=1,
               utilisation_audit_interval=1,
               return_detailed_logs=False
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
    if scenario.model == "full":
        model = TreatmentCentreModel(scenario)
    if scenario.model == "simplest":
        model = TreatmentCentreModelSimpleNurseStepOnly(scenario)
    if scenario.model == "simple_with_branch":
        model = TreatmentCentreModelSimpleBranchedPathway(scenario)

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
                          rc_period=DEFAULT_RESULTS_COLLECTION_PERIOD,
                          n_reps=5,
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

    # if return_full_log:
    #     results = [single_run(scenario,
    #                           rc_period,
    #                           random_no_set=(scenario.random_number_set)+rep,
    #                           return_full_log=True,
    #                           return_event_log=False)
    #                for rep in range(n_reps)]

    # format and return results in a dataframe
    # df_results = pd.concat(reesults)
    # df_results.index = np.arange(1, len(df_results)+1)
    # df_results.index.name = 'rep'
    # return df_results
    # return results

    if return_detailed_logs:
        results = [{'rep': rep+1,
                    'results': single_run(scenario,
                                          rc_period,
                                          random_no_set=(scenario.random_number_set)+rep,
                                          return_detailed_logs=True)}
                   #   .assign(Rep=rep+1)
                   for rep in range(n_reps)]

        # format and return results in a dataframe

        return results
    # {
        # {df_results = [pd.concat(result) for result in results] }
        # }
        # return results

    results = [single_run(scenario,
                          rc_period,
                          random_no_set=(scenario.random_number_set)+rep)
               for rep in range(n_reps)]

    # format and return results in a dataframe
    df_results = pd.concat(results)
    df_results.index = np.arange(1, len(df_results)+1)
    df_results.index.name = 'rep'
    return df_results


# ## Scenario Analysis

def get_scenarios():
    '''
    Creates a dictionary object containing
    objects of type `Scenario` to run.

    Returns:
    --------
    dict
        Contains the scenarios for the model
    '''
    scenarios = {}
    scenarios['base'] = Scenario()

    # extra triage capacity
    scenarios['triage+1'] = Scenario(n_triage=DEFAULT_N_TRIAGE+1)

    # extra examination capacity
    scenarios['exam+1'] = Scenario(n_exam=DEFAULT_N_EXAM+1)

    # extra non-trauma treatment capacity
    scenarios['treat+1'] = Scenario(n_cubicles_1=DEFAULT_N_CUBICLES_1+1)

    # swap over 1 exam room for extra treat cubicle
    scenarios['swap_exam_treat'] = Scenario(n_triage=DEFAULT_N_TRIAGE+1,
                                            n_exam=DEFAULT_N_EXAM-1)

    # scenario + 3 min short mean exam times.
    scenarios['short_exam'] = Scenario(n_triage=DEFAULT_N_TRIAGE+1,
                                       n_exam=DEFAULT_N_EXAM-1,
                                       exam_mean=12.0)
    return scenarios


def run_scenario_analysis(scenarios, rc_period, n_reps):
    '''
    Run each of the scenarios for a specified results
    collection period and replications.

    Params:
    ------
    scenarios: dict
        dictionary of Scenario objects

    rc_period: float
        model run length

    n_rep: int
        Number of replications

    '''
    print('Scenario Analysis')
    print(f'No. Scenario: {len(scenarios)}')
    print(f'Replications: {n_reps}')

    scenario_results = {}
    for sc_name, scenario in scenarios.items():

        print(f'Running {sc_name}', end=' => ')
        replications = multiple_replications(scenario, rc_period=rc_period,
                                             n_reps=n_reps)
        print('done.\n')

        # save the results
        scenario_results[sc_name] = replications

    print('Scenario analysis complete.')
    return scenario_results


def scenario_summary_frame(scenario_results):
    '''
    Mean results for each performance measure by scenario

    Parameters:
    ----------
    scenario_results: dict
        dictionary of replications.  
        Key identifies the performance measure

    Returns:
    -------
    pd.DataFrame
    '''
    columns = []
    summary = pd.DataFrame()
    for sc_name, replications in scenario_results.items():
        summary = pd.concat([summary, replications.mean()], axis=1)
        columns.append(sc_name)

    summary.columns = columns
    return summary



####################################################################
# Classes for 'using a simple resource' page
####################################################################

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


# UNFINISHED

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
        trace(f'treatment of patient {self.identifier} begins '
                f'{self.env.now:.3f}')
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

        trace(f'patient {self.identifier} nurse exam/treatment complete '
                f'at {self.env.now:.3f};'
                f'waiting time was {self.wait_treat:.3f}')
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






#####################################################################
# Classes for 'adding an optional step' page
#####################################################################

class TreatmentCentreModelSimpleBranchedPathway:
    '''
    The treatment centre model

    Patients arrive at random to a treatment centre, see a nurse, then either require treatment or depart the system immediately.

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

        # Create examination bays
        self.args.exam = simpy.Store(self.env)

        for i in range(self.args.n_exam):
            self.args.exam.put(
                CustomResource(
                    self.env,
                    capacity=1,
                    id_attribute = i+1)
                )

        # Create treatment bays   
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

        # store rc perio
        self.rc_period = results_collection_period

        # run
        self.env.run(until=results_collection_period)


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
            new_patient = SimpleBranchedPathway(patient_count, self.env, self.args, self.full_event_log)
            self.patients.append(new_patient)
            # start the pathway process for the patient
            self.env.process(new_patient.execute())


# UNFINISHED


class SimpleBranchedPathway(object):
    '''
    Encapsulates the process a patient with minor injuries and illness.

    These patients are arrived, then seen and treated by a nurse as soon as one is available.
    Some patients may then require treatment.

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
        self.wait_exam = -np.inf
        self.wait_treat = -np.inf
        self.total_time = -np.inf

        self.exam_duration = -np.inf
        self.treat_duration = -np.inf

    def execute(self):
        '''
        simulates the simplest minor treatment process for a patient

        1. Arrive
        2. Examined/treated by nurse when one available
        3. Sample to see whether treatment is required
        4. Treated if required
        5. Discharged either after examination or after treatment depending on outcome of 3
        '''
        # record the time of arrival and entered the triage queue
        self.arrival = self.env.now
        self.full_event_log.append(
            {'patient': self.identifier,
             'pathway': 'simple_with_branch',
             'event_type': 'arrival_departure',
             'event': 'arrival',
             'time': self.env.now}
        )

        # request examination resource
        start_wait = self.env.now
        self.full_event_log.append(
            {'patient': self.identifier,
             'pathway': 'simple_with_branch',
             'event': 'examination_wait_begins',
             'event_type': 'queue',
             'time': self.env.now}
        )

        #########################################################
        # All arrivals require examination
        examination_resource = yield self.args.exam.get()

        # record the waiting time for registration
        self.wait_exam = self.env.now - start_wait
        trace(f'treatment of patient {self.identifier} begins '
                f'{self.env.now:.3f}')
        self.full_event_log.append(
            {'patient': self.identifier,
                'pathway': 'simple_with_branch',
                'event': 'examination_begins',
                'event_type': 'resource_use',
                'time': self.env.now,
                'resource_id': examination_resource.id_attribute
                }
        )

        # sample examination duration.
        self.exam_duration = self.args.exam_dist.sample()
        yield self.env.timeout(self.exam_duration)

        trace(f'patient {self.identifier} nurse exam/treatment complete '
                f'at {self.env.now:.3f};'
                f'waiting time was {self.wait_treat:.3f}')
        self.full_event_log.append(
            {'patient': self.identifier,
                'pathway': 'simple_with_branch',
                'event': 'examination_complete',
                'event_type': 'resource_use_end',
                'time': self.env.now,
                'resource_id': examination_resource.id_attribute}
        )

        # Resource is no longer in use, so put it back in the store
        self.args.exam.put(examination_resource) 
        #########################################################

        # sample if patient requires treatment?
        self.require_treat = self.args.nt_p_treat_dist.sample()  #pylint: disable=attribute-defined-outside-init

        if self.require_treat:

            self.full_event_log.append(
                {'patient': self.identifier,
                 'pathway': 'simple_with_branch',
                 'event': 'requires_treatment',
                 'event_type': 'attribute_assigned',
                 'time': self.env.now}
            )

            # record the time that entered the treatment queue
            start_wait = self.env.now
            self.full_event_log.append(
                {'patient': self.identifier,
                 'pathway': 'simple_with_branch',
                 'event': 'treatment_wait_begins',
                 'event_type': 'queue',
                 'time': self.env.now}
            )
            ###################################################
            # request treatment cubicle

            non_trauma_treatment_resource = yield self.args.treatment.get()

            # record the waiting time for treatment
            self.wait_treat = self.env.now - start_wait
            trace(f'treatment of patient {self.identifier} begins '
                    f'{self.env.now:.3f}')
            self.full_event_log.append(
                {'patient': self.identifier,
                    'pathway': 'simple_with_branch',
                    'event': 'treatment_begins',
                    'event_type': 'resource_use',
                    'time': self.env.now,
                    'resource_id': non_trauma_treatment_resource.id_attribute
                }
            )

            # sample treatment duration.
            self.treat_duration = self.args.nt_treat_dist.sample()
            yield self.env.timeout(self.treat_duration)

            trace(f'patient {self.identifier} treatment complete '
                    f'at {self.env.now:.3f};'
                    f'waiting time was {self.wait_treat:.3f}')
            self.full_event_log.append(
                {'patient': self.identifier,
                    'pathway': 'simple_with_branch',
                    'event': 'treatment_ends',
                    'event_type': 'resource_use_end',
                    'time': self.env.now,
                    'resource_id': non_trauma_treatment_resource.id_attribute}
            )

            # Resource is no longer in use, so put it back in the store
            self.args.treatment.put(non_trauma_treatment_resource)

            self.full_event_log.append(
                {'patient': self.identifier,
                'pathway': 'simple_with_branch',
                'event': 'depart',
                'event_type': 'arrival_departure',
                'time': self.env.now}
            )
        ##########################################################################
        else:
            self.full_event_log.append(
                {'patient': self.identifier,
                 'pathway': 'simple_with_branch',
                 'event': 'does_not_require_treatment',
                 'event_type': 'attribute_assigned',
                 'time': self.env.now}
            )
            self.full_event_log.append(
                {'patient': self.identifier,
                'pathway': 'simple_with_branch',
                'event': 'depart',
                'event_type': 'arrival_departure',
                'time': self.env.now}
            )
        # total time in system
        self.total_time = self.env.now - self.arrival



