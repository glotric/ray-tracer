import numpy as np
import matplotlib.pyplot as plt

f = open("podatki.txt", 'r')
lines = f.readlines()

width, height = lines[1].split()
cam_pos = tuple(lines[2].split('=')[-1].strip('()\n ').split(','))

camera = np.array([float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])])
ratio = float(width) / float(height)
screen = (-1, 1 / ratio, 1, -1 / ratio) # L U R D

