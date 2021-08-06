# ContinuousGlucocseMonitoring

### Using Data Science algorithms and techniques  to find patterns, anomalies and correlation within the given CGM datasets to predict outcomes for given problem statements.


## Project outline
In this project we are considering the **Artificial Pancreas medical control system**, specifically the **Medtronic 670G** system. The Medtronic system consists of a continuous glucose monitor (CGM), the Guardian Sensor (12) is used to _collect blood glucose measurements every 5 minutes_.

The sensor is single use and can be used continuously for 7 days after which it has to be replaced. The replacement procedures include a recalibration process that requires the user to obtain blood glucose measurements using a Contour NextLink 2.4 glucosemeter®. Note that this process also requires manual intervention.

The Guardian Link Transmitter®, powers the CGM sensor and sends the data to the MiniMed 670G® insulin pump. The insulin pump utilizes the Smart Guard Technology, that modulates the insulin delivery based on the CGM data.

The SmartGuard Technology uses a Proportional, Integrative, and Derivative controller to derive small bursts of insulin also called **Micro bolus** to be delivered to the user. During meals, the user uses a BolusWizard to compute the amount of food _bolus required to maintain blood glucose levels_. The user manually estimates the amount of carbohydrate intake and enters it to the Bolus Wizard. The Bolus Wizard is pre-configured with correction factor, body weight and average insulin sensitivity of the subject calculates the bolus insulin to be delivered. The user can then program the MiniMed 670G infusion pump to deliver that amount. In addition to the bolus, the MiniMed 670Ginsulin pump can also provide a correction bolus. The correction bolus amount is provided only if the CGM reading is above a threshold (typically 120 mg/dL) and is a proportional amount with respect to the difference of the CGM reading and the threshold.
