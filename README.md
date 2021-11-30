# Deep Learning Homework: Charity Funding Predictor

In this assignment, my goal was to create an algorithm to predict whether or not applicants for funding will be successful if funded by Alphabet Soup using the deep learning techniques we learned in class. 

I started with a CSV containing more than 34,000 organizations to have received funding from Alphabet Soup over the years. A number of columns capture metadata about each organization:

  - EIN and NAME—Identification columns
  - APPLICATION_TYPE—Alphabet Soup application type
  - AFFILIATION—Affiliated sector of industry
  - CLASSIFICATION—Government organization classification
  - USE_CASE—Use case for funding
  - ORGANIZATION—Organization type
  - STATUS—Active status
  - INCOME_AMT—Income classification
  - SPECIAL_CONSIDERATIONS—Special consideration for application
  - ASK_AMT—Funding amount requested
  - IS_SUCCESSFUL—Was the money used effectively


## Step 1: Preprocessing

I dropped the EIN and Name columns since they will not be used as features.
For the Application Type and Classification columns, I took the top values and put the others into an Other group for each column.
Then, I used pandas to get_dummies() to encode all of the categorical value columns. Finally I created my features variable X and target variable y, which is the IS_SUCCESSFUL column.

Finally, using sklearn, I split the data into training and testing and then used StandardScalar() to get our dataset ready for the model.

## Step 2: Compile, Train, and Evaluate the Model

Using Tensorflow, I defined a Sequential model that used kerastuner and activation relu for first and second layer since it is the most commonly used for hidden layers. Then,  sigmoid for the output layer since it is a classification model.  

Lastly, I compiled and trained the model and while doing so, saved the best model to AlphabetSoupCharity.hdf5.

## Results

The results from this model revealed a Loss of 0.5548 and Accuracy of 0.7244.

## Optimizer

In an effort to optimize the model and acheive a target predictive accuracy higher than 75%, I tried the following:

