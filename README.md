
<h1 align="center">
  <br>
COVID 19 Detection
  <br>
 </h1>
 </h1>
  <p align="center">
    <a • href="https://github.com/arielweizman1">Ariel Weizman</a> 
    <a • href="https://github.com/shavityaelshavit">Yael Shavit</a>
  </p>
In this project, we developed a Convolutional Neural Network algorithm to predict Covid-19 from chest X-Ray images. First, we trained the model to predict Pneumonia from the Pneumonia chest X-Ray dataset and then fine-tuned the model on the Covid data.


- [COVID 19 Detection](#covid-19-detection)
  * [Background](#background)
  * [Files in the repository](#files-in-the-repository)
  * [Data](data)
  * [Working with the model](#working-with-the-model)
  * [References](#references)

## Background
This project goal: Given an input chest X-ray image, the algorithm must detect whether the person has been infected with Covid-19 or not.
The first idea was to use Transfer learning and leverage the already existing labeled data of some related task or domain, in our case image classification.
We noticed that unlike the availability of a small number of X-ray images of COVID-19 patients, we found a bigger dataset of X-ray images of Pneumonia patients.
Hence we decided to divide our mission into 2 parts:

* The first part 
  * Given an input chest X-ray image, the algorithm must detect whether the person has been infected with Pneumonia or not.
The algorithm for this part use transfer learning. We take some familiar ConvNet and use it as a fixed feature extractor. we freeze the weights of the network except that of the final fully connected layer. This last fully connected layer is replaced with several new ones, with random weights and only the new layers are trained.
 
 
 ![image](https://user-images.githubusercontent.com/65540180/124584259-b467ef80-de5c-11eb-804e-287059c8643d.png)


* The second part: 
  * Given an input chest X-ray image, the algorithm must detect whether the person has been infected with Covid-19 or not.
Our algorithm is a form of fine-tuning method. We take the same model we developed in the first part as a pre-trained net. instead of random initialization, we initialize the network with the weights trained in the first part. next, we will do fine-tuning not on the whole Net, but on the layers, we added at the first part.

![image](https://user-images.githubusercontent.com/65540180/124584297-bcc02a80-de5c-11eb-8f70-3cef41438563.png)


## Files in the repository


|File name         | Purpsoe |
|----------------------|------|
|`COVID_19_Detection.ipynb`| main code for this project in Colaboratory format.|
|`COVID_19_Detection.py`| main code for this project in py format|
|`Creating_the_Covid_19_Dataset.ipynb`| code for fetching only PA X-ray images from the covid dataset. Colaboratory format|
|`Creating_the_Covid_19_Dataset.py`| code for fetching only PA X-ray images from the covid dataset. py format|
|`/COVID19/dataset/*`| folders contain the train and test images for covid-19|




## Data
We used 2 datasets from 2 sources.
* Pneumonia dataset: set of supervised X-ray images that have been labeled by radiologists as Normal / Pneumonia.
  * This dataset was loaded from [kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
* Covid dataset: we created this dataset usind 2 surces:
  *  Normal : we loaded images from the *test set* above
  *  Covid : we used [supervised X-ray images](https://www.kaggle.com/bachrr/covid-chest-xray) that have been labeld by radiologists as Covid infected.
     *   we used the `Creating_the_Covid_19_Dataset.ipynb` code to fetch only PA images.
     *   we split the data to train and test randomly. the final dataset used for this project can be found under the `/COVID19/dataset/` folders.
 
 We upload the dataset to google drive and accessed the drive while training.
 
## Working with the model
* We trained the model using Google Colab. At the beginning of the code, you can change the directories to work with during the training. more details in the code documentation.
* There is an option to change the depth of the classifier (version 0,1,2,3,4) and you can add more if you want.
you can also play with the pretrained model. we chose efficientNet-b0, but you can try other.


## References
[1] Pneumonia Dataset source: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

[2] Covid-19 Dataset source: https://www.kaggle.com/bachrr/covid-chest-xray

[3] Covid-19 detection source: https://www.kaggle.com/bachrr/detecting-covid-19-in-x-ray-images-with-tensorflow

[4] Pneumonia detection source: https://www.kaggle.com/teyang/pneumonia-detection-resnets-pytorch


