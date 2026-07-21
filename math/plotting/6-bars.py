#!/usr/bin/env python3
"""Plot quantities of fruit as stacked bars."""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

people = ['Farrah', 'Fred', 'Felicia']
positions = np.arange(3)
plt.bar(positions, fruit[0], width=0.5, color='red', label='apples')
plt.bar(positions, fruit[1], width=0.5, color='yellow', label='bananas',
        bottom=fruit[0])
plt.bar(positions, fruit[2], width=0.5, color='#ff8000', label='oranges',
        bottom=fruit[0] + fruit[1])
plt.bar(positions, fruit[3], width=0.5, color='#ffe5b4', label='peaches',
        bottom=fruit[0] + fruit[1] + fruit[2])
plt.xticks(positions, people)
plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80)
plt.yticks(np.arange(0, 81, 10))
plt.title('Number of Fruit per Person')
plt.legend()
plt.show()
