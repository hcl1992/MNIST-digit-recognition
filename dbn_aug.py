# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:00:55 2016

@author: huangchunlin
"""
""" This is a DBN model with data size augmentation
 by some simple translation
 """
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from nolearn.dbn import DBN

def train_test_prep():
    '''
    This function will load the MNIST data, scale it to a 0 to 1 range, and split it into test/train sets. 
    '''

    image_data = fetch_mldata('MNIST Original') # Get the MNIST dataset.

    basic_x = image_data.data
    basic_y = image_data.target # Separate images from their final classification. 

    min_max_scaler = MinMaxScaler() # Create the MinMax object.
    basic_x = min_max_scaler.fit_transform(basic_x.astype(float)) # Scale pixel intensities only.

    x_train, x_test, y_train, y_test = train_test_split(basic_x, basic_y, 
                            test_size = 0.14285, random_state = 0) # Split training/test.
    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = train_test_prep()

from scipy.ndimage import convolve, rotate

def random_image_generator(image):
    '''
    This function will randomly translate and rotate an image, producing a new, altered version as output.
    '''

    # Create our movement vectors for translation first. 

    move_up = [[0, 1, 0],
               [0, 0, 0],
               [0, 0, 0]]

    move_left = [[0, 0, 0],
                 [1, 0, 0],
                 [0, 0, 0]]

    move_right = [[0, 0, 0],
                  [0, 0, 1],
                  [0, 0, 0]]

    move_down = [[0, 0, 0],
                 [0, 0, 0],
                 [0, 1, 0]]

    # Create a dict to store these directions in.

    dir_dict = {1:move_up, 2:move_left, 3:move_right, 4:move_down}

    # Pick a random direction to move.

    direction = dir_dict[np.random.randint(1,5)]

    # Pick a random angle to rotate (10 degrees clockwise to 10 degrees counter-clockwise).

    angle = np.random.randint(-10,11)

    # Move the random direction and change the pixel data back to a 2D shape.

    moved = convolve(image.reshape(28,28), direction, mode = 'constant')

    # Rotate the image

    rotated = rotate(moved, angle, reshape = False)

    return rotated

def extra_training_examples(features, targets, num_new):
    '''
    This function will take the training set and increase it by artifically adding new training examples.
    We can also specify how many training examples we wish to add with the num_new parameter.
    '''

    # First, create empty arrays that will hold our new training examples.

    x_holder = np.zeros((num_new, features.shape[1]))
    y_holder = np.zeros(num_new)

    # Now, loop through our training examples, selecting them at random for distortion.

    for i in xrange(num_new):
        # Pick a random index to decide which image to alter.

        random_ind = np.random.randint(0, features.shape[0])

        # Select our training example and target.

        x_samp = features[random_ind]
        y_samp = targets[random_ind]

        # Change our image and convert back to 1D.

        new_image = random_image_generator(x_samp).ravel()

        # Store these in our arrays.

        x_holder[i,:] = new_image
        y_holder[i] = y_samp

    # Now that our loop is over, combine our original training examples with the new ones.

    combined_x = np.vstack((features, x_holder))
    combined_y = np.hstack((targets, y_holder))

    # Return our new training examples and targets.

    return combined_x, combined_y
x_train, y_train = extra_training_examples(x_train, y_train, 60000)

dbn_model = DBN([x_train.shape[1],300,300,10],
                learn_rates = 0.2,
                learn_rate_decays = 0.9,
                epochs = 100,
                verbose = 1)
dbn_model.fit(x_train, y_train)

from sklearn.metrics import classification_report, accuracy_score


y_true, y_pred = y_test, dbn_model.predict(x_test) # Get our predictions
print(classification_report(y_true, y_pred)) # Classification on each digit


print 'The accuracy is:', accuracy_score(y_true, y_pred)



