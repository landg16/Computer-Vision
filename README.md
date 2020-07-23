# Computer-Vision
This repository is only for studying and has low results. (Trust me, don't use this :D )

# Preparation
Firstly, i started to prepare data for train. I prepared data for color, category and doors number detection. For this i needed only outside of car, so i used [rovehicle_counting_tensorflowot](https://github.com/ahmetozlu/vehicle_counting_tensorflow) and modified it little bit to only detect cars.

Then i filtered car images, where only car outside visual are shown and splitted data into train, validation and test.

# Modelling
I chose RESNET 18, it has 11M parameters. Also learning from scratch is a big headache, so that's it.

# Training
Because of huge amount's of data, i chose Google's colab, to make my life easier. There is Generate_Car_Data.ipynb file, where i generated train data.
### Color Training
For color training, i chose only 7 color out of 16, because 9 color's data was too few for training. I chose 105K data for colors.
### Category Training
For category training, i chose 4 category out of 11, because 7 categories data was too few, while Sedan was 125K, Limousine was 55, so there was no way for model to learn anything about Limousines. I chose 50K data for that.
### Doors Training
For door training, i chose all doors, even tho >5 was only 1286, there was only 3 option, so i picked 9350 from 2/3 and 24400 from 4/5.

# Result
I have 3 IPYNB file, Evaluation_Doors, Evaluation_Colors, Evaluation_Categories. There are results and how i got them.
