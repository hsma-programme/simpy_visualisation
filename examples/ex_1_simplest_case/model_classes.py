#!/usr/bin/env python
# coding: utf-8
'''
 patient arrives at the treatment centre, is seen, and then leaves
'''
import itertools
import numpy as np
import pandas as pd
import simpy

from examples.ex_1_simplest_case.distribution_classes import Exponential, Lognormal
from examples.simulation_utility_functions import trace, CustomResource

# Simulation model run settings

class Scenario:
    '''
    Container class for scenario parameters/arguments

    Passed to a model and its process classes
    '''

    def __init__(self,
                 random_number_set=42,  
                 n_streams = 20,               
                 
                 n_cubicles_1=2,

                 trauma_treat_mean=30,
                 trauma_treat_var=5,
                 
                 manual_arrival_rate=2

                 ):
        '''
        Create a scenario to parameterise the simulation model

        Parameters:
        -----------
        random_number_set: int, optional (default=DEFAULT_RNG_SET)
            Set to control the initial seeds of each stream of pseudo
            random numbers used in the model.

        n_cubicles_1: int
            The number of treatment cubicles

        trauma_treat_mean: float
            Mean of the trauma cubicle treatment distribution (Lognormal)

        trauma_treat_var: float
            Variance of the trauma cubicle treatment distribution (Lognormal)

         manual_arrival_rate: float
            Set the mean of the exponential distribution that is used to sample the 
            inter-arrival time of patients

        
        '''
        # sampling
        self.random_number_set = random_number_set
        self.n_streams = n_streams

        # store parameters for sampling
        
        self.trauma_treat_mean = trauma_treat_mean
        self.trauma_treat_var = trauma_treat_var
        
        self.manual_arrival_rate = manual_arrival_rate
        

        self.init_sampling()

        # count of each type of resource
        self.init_resource_counts(n_cubicles_1)

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

    def init_resource_counts(self, n_cubicles_1):
        '''
        Init the counts of resources to default values...
        '''
        self.n_cubicles_1 = n_cubicles_1

    def init_sampling(self):
        '''
        Create the distributions used by the model and initialise 
        the random seeds of each.
        '''
        # create random number streams
        rng_streams = np.random.default_rng(self.random_number_set)
        self.seeds = rng_streams.integers(0, 999999999, size=self.n_streams)

        # create distributions
        # treatment of trauma patients
        self.treat_dist = Lognormal(self.trauma_treat_mean,
                                    np.sqrt(self.trauma_treat_var),
                                    random_seed=self.seeds[5])

        self.arrival_dist = Exponential(self.manual_arrival_rate,  # pylint: disable=attribute-defined-outside-init
                                        random_seed=self.seeds[8])

# ## Patient Pathways Process Logic

class SimplePathway(object):
    '''
    Encapsulates the process for a patient with minor injuries and illness.

    These patients are arrived, then seen and treated by a nurse as soon as one is available.
    No place-based resources are considered in this pathway.

    Following treatment they are discharged.
    '''

    def __init__(self, identifier, env, args, event_log):
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
        self.event_log = event_log

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
        self.event_log.append(
            {'patient': self.identifier,
             'pathway': 'Simplest',
             'event_type': 'arrival_departure',
             'event': 'arrival',
             'time': self.env.now}
        )

        # request examination resource
        start_wait = self.env.now
        self.event_log.append(
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
        self.event_log.append(
            {'patient': self.identifier,
                'pathway': 'Simplest',
                'event': 'treatment_begins',
                'event_type': 'resource_use',
                'time': self.env.now,
                'resource_id': treatment_resource.id_attribute
                }
        )

        # sample treatment duration
        self.treat_duration = self.args.treat_dist.sample()
        yield self.env.timeout(self.treat_duration)

        self.event_log.append(
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
        self.event_log.append(
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

        self.event_log = []
        self.utilisation_audit = []

    def init_resources(self):
        '''
        Init the number of resources
        and store in the arguments container object

        Resource list:
            1. Nurses/treatment bays (same thing in this model)

        '''     
        self.args.treatment = simpy.Store(self.env)

        for i in range(self.args.n_cubicles_1):
            self.args.treatment.put(
                CustomResource(
                    self.env,
                    capacity=1,
                    id_attribute = i+1)
                )



    def run(self, results_collection_period=60*24*10):
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

            interarrival_time = 0.0

            interarrival_time += self.args.arrival_dist.sample()

            yield self.env.timeout(interarrival_time)

            trace(f'patient {patient_count} arrives at: {self.env.now:.3f}')

            # Generate the patient
            new_patient = SimplePathway(patient_count, self.env, self.args, self.event_log)
            self.patients.append(new_patient)
            # start the pathway process for the patient
            self.env.process(new_patient.execute())
