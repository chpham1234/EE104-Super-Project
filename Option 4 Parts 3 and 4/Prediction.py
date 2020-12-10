from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

'''
2 image options for loading, Ex. airplane2 or car1
0: airplane
1: car
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck
'''

image = 'horse1'

#we Resized
filename = 'Resized/R_'+image+'.jpg'
#filename = 'option2pics/'+image+'.jpg'

# load and prepare the image
def load_image(filename):
	# load the image
    img = load_img(filename, target_size=(32, 32))
    plt.imshow(img)
	# convert to array
    img = img_to_array(img)
	# reshape into a single sample with 3 channels
    img = img.reshape(1, 32, 32, 3)
	# prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

def print_class(result):
    if (result == 0):
        print ("Airplane")
    if (result == 1):
        print ("Automobile")
    if (result == 2):
        print ("Bird")
    if (result == 3):
        print ("Cat")
    if (result == 4):
        print ("Deer")
    if (result == 5):
        print ("Dog")
    if (result == 6):
        print ("Frog")
    if (result == 7):
        print ("Horse")
    if (result == 8):
        print ("Ship")
    if (result == 9):
        print ("Truck")
        
#I still need to convert to float
def print_percent(img,model):
    percentage = model.predict(img)
    print("\nAirplane: {:0.2f}%".format(100*percentage[0, 0]))
    print("Automobile: {:0.2f}%".format(100*percentage[0, 1]))
    print("Bird: {:0.2f}%".format(100*percentage[0, 2]))
    print("Cat: {:0.2f}%".format(100*percentage[0, 3]))
    print("Deer: {:0.2f}%".format(100*percentage[0, 4]))
    print("Dog: {:0.2f}%".format(100*percentage[0, 5]))
    print("Frog: {:0.2f}%".format(100*percentage[0, 6]))
    print("Horse: {:0.2f}%".format(100*percentage[0, 7]))
    print("Ship: {:0.2f}%".format(100*percentage[0, 8]))
    print("Truck: {:0.2f}%\n".format(100*percentage[0, 9]))

    
# load an image and predict the class
def run_example(choosemodel):
	# load the image
	# load model
    model = load_model(choosemodel)
	# predict the class
    img = load_image(filename)
    result = model.predict_classes(img)
    img = load_image(filename)
    print("\nModel used:", choosemodel)
    print("Original image is", image)
    print_percent(img,model)
    print("The image is in class", result[0], "which is a:")
    print_class(result[0])

Final = 'finalmodel.h5'
Baseline = 'Baseline3.h5'

# entry point, run the example
run_example(Baseline)
run_example(Final)

