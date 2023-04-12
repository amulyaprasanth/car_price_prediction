import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense
from keras.optimizers import Adam


def build_model():
    "This model builds and compiles the model with mean absolute loss and adam optimizer and returns it"
    model = keras.Sequential([
        Dense(32, activation='relu', input_shape=(10,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(0.001))
    return model

# This code is taken from geeks for geeks page (https://www.geeksforgeeks.org/convert-the-number-from-international-system-to-indian-system/)


def convert(input):
    """This function converts numbers from international system to indian system numbers"""

    # Convert the input to string format for len method and slicing
    input = str(input)

    # Find the length of the
    # input string
    Len = len(input)

    # Removing all the separators(, )
    # from the input string
    i = 0
    while(i < Len):
        if(input[i] == ","):
            input = input[:i] + input[i + 1:]
            Len -= 1
            i -= 1
        elif(input[i] == " "):
            input = input[:i] + input[i + 1:]
            Len -= 1
            i -= 1
        else:
            i += 1
        # Reverse the input string
        input = input[::-1]

        # Declaring the output string
        output = ""

        # Process the input string
        for i in range(Len):

            # Add a separator(, ) after the
            # third number
            if(i == 2):
                output += input[i]
                output += ","

            # Then add a separator(, ) after
            # every second number
            elif(i > 2 and i % 2 == 0 and
                    i + 1 < Len):
                output += input[i]
                output += ","
            else:
                output += input[i]

        # Reverse the output string
        output = output[::-1]

        # Return the output string back
        # to the main function
        return output
