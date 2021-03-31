import numpy as np
import math
import matplotlib.pyplot as plt


#Definicija vektorjev

class Vector:
    
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return 'Vector({0}, {1}, {2})'.format(self.x, self.y, self.z)

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


#Definicija žarka

class Ray:

    def __init__(self, origin, direction): #origin in direction argumenta sta Vector
        self.origin = origin
        self.direction = direction.normalize()

    def __str__(self):
        return '{0} + s*{1}'.format(self.origin, self.direction)

    def __repr__(self):
        return 'Ray({0}, {1})'.format(self.origin, self.direction)

    def reflect(self, sphere):
        point = sphere.intersect_point(self)
        normal = sphere.intersect_normal(self)

        factor = self.direction.dotproduct(normal)
        scaled = normal.scale(factor)
        ref_direction = self.direction - scaled

        return Ray(point, ref_direction)


#Definicija krogel

class Sphere:

    def __init__(self, r, center, RGB, reflect):   #pos je Vector, RGB je touple, r in reflect sta int
        self.r = r
        self.center = center
        self.RGB = RGB
        self.reflect = reflect

    def __str__(self):
        return 'r = {0}, center = {1}, colour = {2}, reflect = {3}'.format(self.r, self.center, self.RGB, self.reflect)

    def __repr__(self):
        return 'Sphere({0},{1},{2},{3})'.format(self.r, self.center, self.RGB, self.reflect)

    def intersect(self, ray):
        intersection = {'in': 0, 'out': 0, 'ok': False}

        A = ray.direction.norm()**2
        B = 2 * ray.direction.dotproduct(ray.origin-self.center)
        C = (ray.origin-self.center).norm()**2 - self.r**2

        disc = B**2 - 4*A*C

        if disc > 0:
            lambda1 = (-B + np.sqrt(disc)) / (2*A)
            lambda2 = (-B - np.sqrt(disc)) / (2*A)

            if lambda1 >= 0 and lambda2 >= 0:
                if lambda1 < lambda2:
                    intersection['in'] = lambda1
                    intersection['out'] = lambda2
                else:
                    intersection['in'] = lambda2
                    intersection['out'] = lambda1
                intersection['ok'] = True

        return intersection

    def intersect_point(self, ray):
        point = ray.origin + ray.direction.scale(self.intersect(ray)['in'])
        return point

    def intersect_normal(self, ray):
        alpha = self.intersect(ray)['in']
        v1 = ray.origin + ray.direction.scale(alpha)

        v2 = v1 - self.center
        normal = v2.scale(1/self.r)

        return normal.normalize()



#Začne brat podatke

f = open("podatki.txt", 'r')
lines = f.readlines()

width, height = lines[1].split()
cam_pos = tuple(lines[2].split('=')[-1].strip('()\n ').split(','))

camera = Vector([float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])])
ratio = float(width) / float(height)
screen = (-1, 1, -1/ratio, 1/ratio) # L R D U

'''image = np.zeros((int(height), int(width), 3)) 
for i, y in enumerate(np.linspace(screen[2], screen[3], int(height))):
    for j, x in enumerate(np.linspace(screen[0], screen[1], int(width))):
        # image[i, j] = ...
        print("progress: %d/%d" % (i + 1, int(height)))

plt.imsave('image.png', image)'''


pos = Vector(2,4,0)
krogla = Sphere(5, pos, (123, 34, 5), 0.5)
zarek = Ray(Vector(-12,4,0), Vector(1, 0, 0))
print(krogla)
print(zarek)
print(krogla.intersect_point(zarek))
print(zarek.reflect(krogla))
