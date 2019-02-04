# Dr-Suess Machine Learning LSTM

This project focuses on using LSTM machine learning models to create Dr Seuss like stories from a user input seed.

![Graph of Model Loss per Epoch](https://github.com/zen-gineer/Dr-Seuss-Machine-Learning-LSTM/blob/master/ModelLoss.PNG)

## Understanding LSTM

![](https://github.com/zen-gineer/Dr-Seuss-Machine-Learning-LSTM/blob/master/node/www/pages/2.png)
![](https://github.com/zen-gineer/Dr-Seuss-Machine-Learning-LSTM/blob/master/node/www/pages/3.png)

## Getting Started

word_based.ipnb is our experimentation with word based LSTM models. This has the advantage of outputing only words that it has seen, and will therefore be more readable. However this makes for less training data availability than the character based model. Our character based model can be found in seuss.ipnb or seuss.py 

### Prerequisites

The model is very big, and may require a dedicated GPU to run. AWS provides GPU's and can be experimented on with a free trial. 
You'll need:
npm, flask, and sufficient hardware for model utilization, or training should you want to play with the model. 

## Running the tests

The parameters of this model creation code are easily manipulatable for experiementation. When the code runs it will save model information, as well as training loss, accuracy and model output data in a json file for visibilty. You can see previous output files in AlldataX.json


## Built With
* [Dedicated GPU] (https://aws.amazon.com/ec2/instance-types/p3/) - Or similar


## Authors

* **Jennifer Shtaway** - *Model Training, web developement, graphical representations of results*


## Acknowledgments

* Hat tip to my team, Vish, Greyson, and Alper who collaborated with me on this.

