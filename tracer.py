import numpy as np
import math
import matplotlib.pyplot as plt

f = open("podatki.txt", 'r')
lines = f.readlines()

width, height = lines[1].split()
cam_pos = tuple(lines[2].split('=')[-1].strip('()\n ').split(','))

camera = np.array([float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])])
ratio = float(width) / float(height)
screen = (-1, 1 / ratio, 1, -1 / ratio) # L U R D

class Vector:
    
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

class Vector(Vector):
    def __repr__(self):
        return 'Vector ({0}, {1}, {2})'.format(self.x, self.y, self.z)

    def __str__(self):
        return '({0}, {1}, {2})'.format(self.x, self.y, self.z)

    def norm(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        norm = self.norm()
        normalized = (self.x/norm, self.y/norm, self.z/norm)
        return Vector(*normalized)

    def dotproduct(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z

    def __add__(self, other):
        added = (self.x+other.x, self.y+other.y, self.z+other.z)
        return Vector(*added)

    def __sub__(self, other):
        subbed = (self.x-other.x, self.y-other.y, self.z-other.z)
        return Vector(*subbed)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def scale(self, alpha):
        scaled = (alpha*self.x, alpha*self.y, alpha*self.z)
        return Vector(*scaled)
