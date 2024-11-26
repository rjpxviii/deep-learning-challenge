
# Alphabet Soup Charity Analysis

## Overview

The purpose of this project is to assist Alphabet Soup, a non-profit organization, in predicting the success of funding applications. Using historical data, the goal was to build a machine learning model that identifies patterns and helps determine which applications are likely to succeed. The dataset includes various attributes of organizations, such as application type, classification, income levels, requested amounts, and special considerations.

This analysis includes preprocessing steps, designing and optimizing a neural network, and evaluating its performance. The ultimate aim was to achieve an accuracy of at least 75%.

---

## Data Preprocessing

### Target and Features
- **Target Variable:** `IS_SUCCESSFUL` (binary classification: 1 for success, 0 for failure).
- **Features:**
  - `APPLICATION_TYPE`
  - `AFFILIATION`
  - `CLASSIFICATION`
  - `USE_CASE`
  - `ORGANIZATION`
  - `INCOME_AMT`
  - `SPECIAL_CONSIDERATIONS`
  - `ASK_AMT`
- **Variable Removed:**
  - `EIN` and `NAME` (unique identifiers irrelevant for predictions).
  - `STATUS` was removed after the first model run, as it was determined to add minimal predictive value.

### Preprocessing Steps
1. **Combining Rare Categories:**
   - In `APPLICATION_TYPE`, types with fewer than 156 occurrences were grouped into a new category labeled `Other`.
   - In `CLASSIFICATION`, types with fewer than 1,883 occurrences were grouped as `Other`.
2. **Encoding Categorical Variables:**
   - Used `pd.get_dummies()` to encode categorical variables into numerical representations.
3. **Data Splitting:**
   - The data was split into training and testing sets using `train_test_split`.
4. **Scaling:**
   - Numerical features were scaled using `StandardScaler` to standardize the input data.

---

## Initial Model

### Model Design
1. **First Hidden Layer:**
   - Neurons: 80
   - Activation: ReLU
   - Input Dimensions: 43 (number of features).
2. **Second Hidden Layer:**
   - Neurons: 30
   - Activation: ReLU
3. **Output Layer:**
   - Neurons: 1
   - Activation: Sigmoid (binary classification).

### Training Details
- **Loss Function:** Binary cross-entropy
- **Optimizer:** Adam
- **Epochs:** 100

### Results
- **Loss:** 0.57
- **Accuracy:** 73%

---

## Optimized Models

### Attempt 1 (Baseline)
- **Layers:** Same as initial model.
- **Epochs:** 100
- **Results:** 
  - Loss: 0.56
  - Accuracy: 73%

### Attempt 2
- **Changes:**
  - Removed `STATUS` variable, reducing input dimensions to 42.
  - Increased neurons in the first layer to 90 and in the second layer to 40.
  - Reduced epochs to 60.
- **Results:**
  - Loss: 0.557
  - Accuracy: 72.5%

### Attempt 3
- **Changes:**
  - Added a third hidden layer with 10 neurons and ReLU activation.
  - Increased epochs to 70.
- **Results:**
  - Loss: 0.55
  - Accuracy: 72%

### Attempt 4 (Final Model)
- **Changes:**
  - Increased neurons:
    - First Layer: 100 neurons.
    - Second Layer: 50 neurons.
    - Third Layer: 20 neurons.
  - Increased epochs to 80.
- **Results:**
  - Loss: 0.58
  - Accuracy: 72.5%

---

## Summary of Results

Despite several attempts at optimization, the models consistently achieved accuracies around 72–73%. The target accuracy of 75% was not reached. The following observations were made:
- Grouping rare categories reduced feature dimensions and improved model efficiency but did not significantly affect accuracy.
- Increasing neurons and layers improved the model slightly, but the gains were marginal.

### Recommendations for Improvement
1. **Feature Engineering:**
   - Introduce additional quantitative features or refine existing ones.
   - Remove potentially irrelevant features.
2. **Alternative Models:**
   - A **Random Forest Classifier** or **Gradient Boosting Model** could be explored. These models often handle categorical data well and provide feature importance metrics.
   - These models may outperform neural networks in this case due to the categorical-heavy dataset.
3. **Data Augmentation:**
   - Incorporate more detailed data points to capture nuances in organizational success metrics.

---

### Deployment
The final optimized model was saved as `AlphabetSoupCharity_Optimization.h5` for future use in predicting the success of new applications. Further research and experimentation are recommended to meet the target accuracy.
