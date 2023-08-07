import numpy as np

# Passed-in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array(
    [
        [1, 1, 1]
    ]
)# 1x3

# There are 4 inputs and 3 neurons
# shape is 3 x 4 and must be Transponse
wights = np.array(
    [
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]
).T