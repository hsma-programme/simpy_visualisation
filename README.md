WORK IN PROGRESS


# Introduction

Visual display of the outputs of discrete event simulations in simpy have been identified as one of the limitations of simpy, potentially hindering adoption of FOSS simulation in comparison to 

> When compared to commercial DES software packages that are commonly used in health research, such as Simul8, or AnyLogic, a limitation of our approach is that we do not display a dynamic patient pathway or queuing network that updates as the model runs a single replication. This is termed Visual Interactive Simulation (VIS) and can help users understand where process problems and delays occur in a patient pathway48; albeit with the caveat that single replications can be outliers. A potential FOSS solution compatible with a browser-based app could use a Python package that can represent a queuing network, such as NetworkX49, and displaying results via matplotlib. If sophisticated VIS is essential for a FOSS model then researchers may need to look outside of web apps; for example, salabim provides a powerful FOSS solution for custom animation of DES models.
-  Monks T and Harper A. Improving the usability of open health service delivery simulation models using Python and web apps [version 2; peer review: 3 approved]. NIHR Open Res 2023, 3:48 (https://doi.org/10.3310/nihropenres.13467.2) 


This repository contains code allowing visually appealing, flexible visualisations of discrete event simulations to be created, such as the example below: 

https://github.com/hsma-programme/Teaching_DES_Concepts_Streamlit/assets/29951987/1adc36a0-7bc0-4808-8d71-2d253a855b31


# Creating a visualisation from an existing model

Two key things need to happen to existing models to work with the visualisation code:
1. All simpy resources need to be changed to simpy stores containing a custom resource with an ID attribute 
2. Logging needs to be added at key points: **arrival, (queueing, resource use start, resource use end), departure**
where the steps in the middle can be repeated for as many queues and resource types as required

# Models used as examples

## Emergency department (Treatment Centre) model:
Monks.T, Harper.A, Anagnoustou. A, Allen.M, Taylor.S. (2022) Open Science for Computer Simulation https://github.com/TomMonks/treatment-centre-sim

The layout code for the emergency department model: https://github.com/hsma-programme/Teaching_DES_Concepts_Streamlit

## The hospital efficiency project model
Harper, A., & Monks, T. Hospital Efficiency Project Orthopaedic Planning Model Discrete-Event Simulation [Computer software]. https://doi.org/10.5281/zenodo.7951080 
https://github.com/AliHarp/HEP/tree/main

