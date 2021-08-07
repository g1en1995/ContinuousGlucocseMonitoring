## Assignment objective: 
1. Extract Meal Data and No Meal Data from the Data provided.
2. Extract discriminatory features from meal and no meal data.
3. Train a machine to recognise Meal or No Meal data.
4. Use k fold cross validation to evaluate the recognistion system.
5. Write a function to determine if the test data given is a Meal or No meal.

Meal Data : Meal Data comprises of 2 hr 30 min stretch of CGM data that starts from Tm-30 mins to Tm+2hrs where Tm is the meal cconsumption time.

No meal Data: 2 hrs of raw data that does not comprise meal intake.

### Steps:
Meal data:
a) There is no meal from time tm to time tm+2hrs, then use this stretch as meal data.
b) There is a meal at some time tp in between such that tp > tm and tp < tm + 2hrs, ignore the meal data at time tm and consider tm the meal at time instead.
c) There is a meal at time tm+2hrs, then consider the stretch from tm+1 30 hrs to tm + 4hrs as meal data.

No meal data:
d) No meal is where there 2 hr stretches of no meal intake.


### Features Extracted from Meal and No Meal Data
1. Meal absorption time  -> when the peak is reached.
2. cgm max - cgm min -> distinct values between meal and no meal.
3. Peak velocity -> find the gradient and find the peak velocity. 
4. Time of peak veocity -> when this peak happens.
5. Rolling Means
6. fft of meal data and no meal data provides different frequencies.

### Training and Testing
Afer the feature vector of both Meal Data and No Meal Data is formed. Training a model to distinguish between a Meal and No Meal using SVM and predicting test data to predict 
Meal and No Meal.



