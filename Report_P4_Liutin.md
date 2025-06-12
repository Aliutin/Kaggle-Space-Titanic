# Spaceship Titanic: Project Report 🚀

This report walks through my approach to the Spaceship Titanic Kaggle competition, where the goal was to predict which passengers got zapped into another dimension. The project focused heavily on data preprocessing and feature engineering before diving into machine learning models.

My personal goal was to achieve 0.8+ minimum accuracy when submitting to Kaggle. The current top result is 0.82230, so I thought 0.8 was a reasonable target. With more time, I could potentially reach 0.81, but I set limits for myself to spend time on other projects rather than just this homework assignment.

## The Data Cleanup Marathon 🧹

### Building Better Features

Rather than just throwing raw data at a model and hoping for the best, I spent most of my time creating new, more meaningful features from what we had:

**Money Matters**
- Created a `Total_bill` column by summing up all passenger spending (RoomService + FoodCourt + ShoppingMall + Spa + VRDeck)
- Built spending ratio features to understand passenger behavior:
  - `Luxury_spending_ratio`: How much went to fancy stuff (Spa, VRDeck)
  - `Food_spending_ratio`: How much went to basic needs (RoomService, FoodCourt)

*[Graph Placeholder: Distribution of Total_bill by HomePlanet and CryoSleep status]*

**Cabin Intelligence** 
- Split the cryptic Cabin codes (like "G/734/S") into useful parts:
  - `Cabin_level`: Which deck they're on
  - `Cabin_number`: Their room number  
  - `Cabin_type`: Port or Starboard side
- Added `Is_premium_deck` flag for passengers on the fancy A, B, or C decks

*[Graph Placeholder: Heatmap showing Transported rate by Cabin_level and Cabin_type]*

**Social Connections**
- Extracted passenger groups from PassengerId patterns - this wasn't obvious at first since I initially assumed IDs were just for merging data. But they're actually in a "group_memberID" format, so we can use the first part as a group identifier.
- I noticed that some last names appear frequently, and most importantly, people with the same last names almost always came from the same home planet (used this insight later for recovery)
- Calculated `Group_size` and `Family_size` (by matching last names)
- Created `Solo_traveler` flag for lone wolves
- Split names into `First_name` and `Last_name` for family analysis

*[Graph Placeholder: Bar chart showing Transported rate by Group_size]*

**Demographics**
- Binned ages into logical groups (Child, Teen, Adult, etc.)



### Filling the Gaps (The Smart Way)

The NaN values in the data weren't really correlated with each other, which means we could work on filling them separately. I first focused on understanding how each variable relates to others, and recovered almost all CryoSleep and HomePlanet variables based on statistical analysis and logic.

*[Graph Placeholder: Missing data heatmap showing NaN patterns before and after imputation]*

Instead of just dropping rows with missing data or using boring averages, I got creative:

**Logic-Based Filling**
- **CryoSleep**: If someone spent money, they obviously weren't frozen. If they spent $0, they probably were (I verified this pattern on non-NaN values first, and it worked perfectly)
- **VIP Status**: Earth passengers were likely not VIPs, and frozen passengers definitely weren't, since all Earth passengers in the data were non-VIP
- **HomePlanet**: Used family name patterns - if the Smith family is consistently from Earth, missing Smiths probably are too

*[Graph Placeholder: Scatter plot showing Total_bill vs CryoSleep status (demonstrating the logic rule)]*

**KNN Imputation for Everything Else**
For the remaining gaps, I used K-Nearest Neighbors imputation. Basically, for each missing value, the algorithm finds the most similar passengers and uses their data to make an educated guess. Way smarter than just using averages across the whole dataset.

(Surprisingly though, filling by mode and mean gave almost identical results for this dataset - the logic-based filling and data analysis was more important than the specific algorithm used)

## The Machine Learning Bit 🤖

I always try to be picky about variables choice, thus I did not used all variables, I used which one looks most important for me: 
- Numerical Columns: ["RoomService", "FoodCourt","ShoppingMall",  "Spa", "VRDeck", "Total_bill", 'Group_size', 'Luxury_spending_ratio', 'Food_spending_ratio', 'Family_size','Cabin_number']
- Categorical Columns:  ['Age_group',"CryoSleep", 'Cabin_level','Cabin_type', "HomePlanet", "Destination", 'Solo_traveler','Is_premium_deck']

### Model Selection & Tuning

**Round 1: Random Forest**
- Started with a basic RandomForestClassifier
- Initial results showed classic overfitting (99.6% training accuracy, 78.5% validation - yikes!)
- Used Optuna to automatically find better parameters through 250 trials
- Best settings: 410 trees, max depth of 22, minimum 5 samples per leaf, entropy criterion

![RF_confusion_matrix](RF_consusion_matrix.png)
  
  And this model give me 0.795 which is not bad, and was top 50% result, but our goal 0.8!
  ![My_first_model](Submision_1.png)





**Round 1.5: Model Shopping**
After optimizing the Random Forest and getting results around 0.79, I decided to test other models. I ran comparisons across different algorithms on the same dataset to see which performed best. CatBoost and LightGBM were the leaders, but CatBoost was too slow and slightly worse, so I went with LightGBM.

| Classifier       | Validation Accuracy | Training Time (s) |
| :--------------- | :------------------ | :---------------- |
| LogisticRegression | 0.776865            | 0.051             |
| KNN              | 0.773250            | 0.012             |
| RandomForest     | 0.792639            | 0.093             |
| LGBM             | 0.799540            | 0.134             |
| CatBoost         | 0.796911            | 2.895             |
| NaiveBayes       | 0.780151            | 0.00              |



**Round 2: LightGBM** 
- Upgraded to LGBMClassifier for more power
- Did extensive hyperparameter tuning with Optuna (250 trials!)
- Used RepeatedStratifiedKFold cross-validation for rock-solid performance estimates
- Final model achieved ~81.3% cross-validation accuracy

*[Graph Placeholder: Optuna optimization history showing accuracy improvement over trials]*

And it gave me 0.804 points, which is more than 0.8! And I stopped here. But I am glad about my results. Of course, I could continue and achieve higher results (top results are very close, while 0.804 is only 497th place out of 1.5k, but again, my goal was my stop point).

![My_final_model](Final_submision.png)

## Results & Takeaways 📊

The heavy focus on feature engineering and smart data imputation paid off. The final tuned LGBM model performed consistently well with about 81.3% accuracy on cross-validation, easily hitting my 0.8+ target.

![My_final_model_matrix](matrix_final.png)

![My_final_model_ROC](Roc_Curve_final.png)

**Key Lessons:**
- **Feature engineering matters more than model choice** - Creating meaningful features from the raw data was where the real gains came from
- **Smart imputation beats simple approaches** - Using KNN and logical rules to fill missing data worked much better than basic methods, though for this specific dataset, the logic-based approach was the real game-changer
- **Proper validation prevents overconfidence** - Using robust cross-validation gave us realistic performance expectations
- **Domain knowledge trumps algorithms** - Understanding the story behind the data (like passenger groups hidden in IDs) was crucial

The project reinforced that in machine learning competitions, the data preprocessing and feature creation phase is often where you win or lose, not in fancy model selection. Sometimes the biggest breakthroughs come from just really understanding what your data is trying to tell you.

*[Graph Placeholder: Final accuracy comparison timeline showing improvement from baseline to final model]*


## After words

I noticed how important Cabin Number, thus decided to look better on it:
![My_final_model_features](Feature_importance_final.png)


Yeah, it looks like, it has not linear percentage of  survival based on this variable-> Will try to make it categorical and run models again.
![Bins_number](Bins_50_not_linear.png)