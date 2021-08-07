## Assignment Objective:
1. Extract features from Meal data
2. Cluster Meal data based on the amount of carbohydrates in each meal.

1. Extract Ground Truth -> Discretize the meal amount in bins of size 20. Consider the meal data generated in Assignment MealNoMeal. 
   According to meal amount label put them in respective bins.
2. Now ignoring the ground truth, perform clustering on the meal data using the features generated in MealNoMeal assignment.
3. Perform DBScan and KMeans clustering.
4. Report accuracy of clustering based on Sum of Squared Error (SSE), Entropy, and Purity metrics.
