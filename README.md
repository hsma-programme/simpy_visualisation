WORK IN PROGRESS


# Introduction

Visual display of the outputs of discrete event simulations in simpy have been identified as one of the limitations of simpy, potentially hindering adoption of FOSS simulation in comparison to commercial modelling offerings or GUI FOSS alternatives such as JaamSim.

> When compared to commercial DES software packages that are commonly used in health research, such as Simul8, or AnyLogic, a limitation of our approach is that we do not display a dynamic patient pathway or queuing network that updates as the model runs a single replication. This is termed Visual Interactive Simulation (VIS) and can help users understand where process problems and delays occur in a patient pathway; albeit with the caveat that single replications can be outliers. A potential FOSS solution compatible with a browser-based app could use a Python package that can represent a queuing network, such as NetworkX, and displaying results via matplotlib. If sophisticated VIS is essential for a FOSS model then researchers may need to look outside of web apps; for example, salabim provides a powerful FOSS solution for custom animation of DES models.
> -  Monks T and Harper A. Improving the usability of open health service delivery simulation models using Python and web apps [version 2; peer review: 3 approved]. NIHR Open Res 2023, 3:48 (https://doi.org/10.3310/nihropenres.13467.2) 


This repository contains code allowing visually appealing, flexible visualisations of discrete event simulations to be created from simpy models, such as the example below: 

https://github.com/hsma-programme/Teaching_DES_Concepts_Streamlit/assets/29951987/1adc36a0-7bc0-4808-8d71-2d253a855b31

Plotly is leveraged to create the final animation, meaning that users can benefit from the ability to further customise or extend the plotly plot, as well as easily integrating with web frameworks such as Streamlit, Dash or Shiny for Python.

The code has been designed to be flexible and could potentially be used with alternative simulation packages such as ciw or simmer if it is possible to provide all of the required details in the logs that are output.

# Creating a visualisation from an existing model

Two key things need to happen to existing models to work with the visualisation code:
1. All simpy resources need to be changed to simpy stores containing a custom resource with an ID attribute 
2. Logging needs to be added at key points: **arrival, (queueing, resource use start, resource use end), departure**
where the steps in the middle can be repeated for as many queues and resource types as required

## All simpy resources need to be changed to simpy stores containing a custom resource with an ID attribute 

To allow the use of resources to be visualised correctly - with entities staying with the same resource throughout the time they are using it - it is essential to be able to identify and track individual resources. 

By default, this is not possible with Simpy resources. They have no ID attribute or similar. 

The easiest workaround which drops fairly painlessly into existing models is to use a simpy store with a custom resource class.

The custom resource is setup as follows:

```{python}
class CustomResource(simpy.Resource):
    def __init__(self, env, capacity, id_attribute=None):
        super().__init__(env, capacity)
        self.id_attribute = id_attribute

    def request(self, *args, **kwargs):
        # Add logic to handle the ID attribute when a request is made
        return super().request(*args, **kwargs)

    def release(self, *args, **kwargs):
        # Add logic to handle the ID attribute when a release is made
        return super().release(*args, **kwargs)
```

The creation of simpy resources is then replaced with the following pattern:
```{python}
beds = simpy.Store(environment)

for i in range(number_of_beds):
    beds.put(
        CustomResource(
            environment,
            capacity=1,
            id_attribute=i+1)
        )
```

Instead of requesting a resource in the standard way, you instead use the .get() method.

```{python}
req = beds.get()
```
or
```{python}
with beds.get() as req:
  ...CODE HERE THAT USES THE RESOURCE...
```

At the end, it is important to put the resource back into the store, even if you used the 'with' notation, so it can be made available to the next requester:
```{python}
beds.put(req)
```
This becomes slightly more complex with conditional requesting (for example, where a resource request is made but if it cannot be fulfilled in time, the requester will renege). This is demonstrated in example 3.

The benefit of this is that when we are logging, we can use the `.id_attribute` attribute of the custom resource to record the resource that was in use.
This can have wider benefits for monitoring individual resource utilisation within your model as well. 

## Logging needs to be added at key points

The animation function needs to be passed an event log with the following layout:

| patient | pathway  | event_type        | event                    | time | resource_id |
|---------|----------|-------------------|--------------------------|------|-------------|
| 15      | Primary  | arrival_departure | arrival                  | 1.22 |             |
| 15      | Primary  | queue             | enter_queue_for_bed      | 1.35 |             |
| 27      | Revision | arrival_departure | arrival                  | 1.47 |             |
| 27      | Revision | queue             | enter_queue_for_bed      | 1.58 |             |
| 12      | Primary  | resource_use_end  | post_surgery_stay_ends   | 1.9  | 4           |
| 15      | Revision | resource_use      | post_survery_stay_begins | 1.9  | 4           |

One easy way to achieve this is by appending dictionaries to a list at each important point in the process. For example:

```{python}
event_log = []
...
...
self.event_log.append(
      {'patient': id,
      'pathway': 'Revision',
      'event_type': 'resource_use',
      'event': 'post_surgery_stay_begins',
      'time': self.env.now,
      'resource_id': bed.id_attribute}
  )
```

# Models used as examples

## Emergency department (Treatment Centre) model:
Monks.T, Harper.A, Anagnoustou. A, Allen.M, Taylor.S. (2022) Open Science for Computer Simulation https://github.com/TomMonks/treatment-centre-sim

The layout code for the emergency department model: https://github.com/hsma-programme/Teaching_DES_Concepts_Streamlit

## The hospital efficiency project model
Harper, A., & Monks, T. Hospital Efficiency Project Orthopaedic Planning Model Discrete-Event Simulation [Computer software]. https://doi.org/10.5281/zenodo.7951080 
https://github.com/AliHarp/HEP/tree/main

