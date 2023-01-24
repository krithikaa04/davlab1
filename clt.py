# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 21:36:15 2023

@author: Krithika
"""

# Central Limit Theorem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from wand.image import Image
from wand.display import display
# 1000 simulations of die roll
n = 1000
# In each simulation, there is one trial more than the previous simulation
avg = []
for i in range(2,n):
    a = np.random.randint(1,7,i)
    avg.append(np.average(a))
print(avg[0:5])
# Function that will plot the histogram, where current is the latest figure
def clt(current):
    # if animation is at the last frame, stop it
    plt.cla()
    if current == 500:
        a.event_source.stop()
plt.hist(avg[0:current])
plt.gca().set_title('Expected value of die rolls')
    plt.gca().set_xlabel('Average from die roll')
    plt.gca().set_ylabel('Frequency')
plt.annotate('Die roll = {}'.format(current), [3,27])
fig = plt.figure()
a = animation.FuncAnimation(fig, clt, interval=5)
plt.show()