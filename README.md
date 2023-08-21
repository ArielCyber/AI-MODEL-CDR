# AI-MODEL-CDR


AI model Content Disarm and Reconstruction


## Acknowledgements

 - [Based on @gaborvecsei Github repository](https://github.com/gaborvecsei/Neural-Network-Steganography)
 
 - [Models were tested using the valdition dataset from imagenet dataset](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) 
 - [Tested with PyTorch TorchVision models](https://pytorch.org/vision/0.8/models.html)
 



## Usage/Examples

Download the needed libraries using the requirements.txt

requirements.txt genetrated using pipreqs


**Initialize**- 

If you wish to attack the model BITS_TO_USE_FOR_ATTACKING are the the bits amount used, if you do not wish to attack the model the value does not matter
```python
BITS_TO_USE_FOR_ATTACKING = 23
DATA_FILE_TYPE = '.JPEG'

my_model = ModelOp(BITS_TO_USE_FOR_ATTACKING,'/your/data/place/val',
file_type = DATA_FILE_TYPE,
csv_solution = '/your/data/place/val_solution.csv',
json_location = '/your/data/place/class_index.json')
```
**Loading a model** -

You can use some of PyTorch built in models with pretriand weights 

ResNet101, Vgg19, Vgg16, Inception, ResNet50, ResNet18, Mobilenet
```python
MODEL_TO_USE = "Vgg16"
my_model.load_model(MODEL_TO_USE)
```
OR

You can use a model serialized using pickle
```python
my_model.load_model('/your/data/place/pickeled_model.bin')
```

**Inserting data (Attacking)**-

Sometimes padding will be needed , if you get this error ValueError: Fraction should be 23 values bits you will need to add padding, happens when data length cannot be divided by BITS_TO_USE_FOR_ATTACKING, the data length of the file is printed when using the insert file function

Insert a string 
```python
MODEL_TO_USE = "Vgg16"
my_model.load_model(MODEL_TO_USE)
```
OR

Insert a file (you can also turn the file into a string)
```python
my_model.insert_file_data('/your/data/place/file.something','padding')
```

You can recover the string data 

```python
BITS_TO_SHOW = 1000
my_model.recover(BITS_TO_SHOW)
```


**Disarming the model**-

You can use n_random or quanti8 , 1 <= BITS_AMOUNT <= 23, when using quanti8 BITS_AMOUNT is not used
```python
strategy = "quanti8"
BITS_AMOUNT = 1
my_model.disarm(strategy,BITS_AMOUNT)
```

**Testing the model**-

Testing using the data you provided in the **Initialize** stage 
```python
my_model.test_model()

```

OR

You can take get model from the class 
```python
my_model.model

```
and test it in any other whey you test you PyTorch models, the model is a normal PyTorch model



## The experiment


We wanted to test the effect of the CDR and the attack on models, we choosed thoose model to test on ResNet101, Vgg19, Vgg16, Inception, ResNet50, ResNet18, Mobilenet.

We tested the models accurasy with the PyTorch pre-trained weights on the imagenet valdition dataset from kaggle (link in the start).

Then we tested the CDR effect on the models we used 23, 10 , 5 , 1 as the values for the random weights disarm, and also we tested the quanti8 option, we disarmed the model then tested the model.

Than we cheacked the effect of attacking the models with 2 kind of files a ransomware as a big file and vs-basic file as small file for every kind of file we attacked using 23, 12, 4 bits, we attacked the model then tested the model.

The last type of test was attacking the model and then disarm it, we used the small file with 4 bits as the 1st attack scenario and the big file with 12 and 23 bits as the 2nd and 3rd scenarios, we attacked each model and then used every kind of disarm(23,10,5,1,quanti8) and tested the model accurasy.

