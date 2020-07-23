# Computer-Vision
This repository is only for studying and has low results. (Trust me, don't use this :D )

# Preparation
First of all, I prepared data for training. I prepared data to detect color, category and number of doors. To do this I only needed exterior of a car, so I used [vehicle_counting_tensorflow](https://github.com/ahmetozlu/vehicle_counting_tensorflow) and  modified it a little in order to only detect cars.

After this I filtered car images, which showed only exterior of cars and splitted data into training, validation and test.

# Modelling
I chose RESNET 18, which has 11M parameters. I used it because learning from a scratch is a big headache.

# Training
Because of huge amounts of data, I chose Google's colab and made my life easier. There is Generate_Car_Data.ipynb file, from where I generated train data.

### Color Training
For color training, I only chose 7 colors out of 16, because data was not enough for the other 9 colors and the model could not learn this. To sum up I chose 105K data for color training.
### Category Training
For category training, I only chose 4 categories out of 11, because data was not enough for the other 7 categories. For example, while Sedan was 125K, Limousine was only 55, so there was no way for model to learn anything about Limousines. I chose 50K data for category training. 

### Door Training
For door training, I chose every door in the data, even though >5 was only 1286, there was only 3 options, so I picked 9350 from 2/3 and 24400 from 4/5.

# Result
I have 3 IPYNB files, Evaluation_Doors, Evaluation_Colors, Evaluation_Categories. These are the results and how I got them.

#### GNU <3
