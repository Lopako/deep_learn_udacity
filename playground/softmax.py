#!/usr/bin/env ipython
""" Softmax """

"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # [e^x_i / sum(e^x)]
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=0)


print(softmax(scores))



# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

# subplot (height 2, width 1, set first subplot active)
plt.subplot(2, 1, 1)

plt.plot(x, softmax(scores).T, linewidth=2)
plt.xlabel('x (?) (x axis label)')
plt.ylabel('softmax (y axis label)')
plt.title('Some Title')
plt.legend(['x', '1.0', '0.2'])


# set second subplot active
plt.subplot(2, 1, 2)
# plt.plot(x * 10, softmax(scores * 10).T, linewidth=2)
plt.plot(x / 10, softmax(scores / 10).T, linewidth=2)
plt.xlabel('x * 10 (?) (x axis label)')
plt.ylabel('softmax (y axis label)')
plt.title('Some Title * 10')
plt.legend(['x*10', '10.0', '2.0'])



plt.show()

