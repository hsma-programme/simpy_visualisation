This example shows a very simple pathway:
    - patient arrivals are generated
    - the patient uses a resource for a variable period of time
    - the patient exits the system after this step

This demonstrates the use of custom simpy resources wthin a simpy store, and also all of the key logging steps required.

All key model logic is contained within `model_classes.py`.

The corresponding page for this in the Streamlit app is `pages/1_Simple_ED_Interactive``.
