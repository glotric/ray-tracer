import numpy as np
import json
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
        if norm == 0:
            return self
        else:
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

        factor = 2*self.direction.dotproduct(normal)
        scaled = normal.scale(factor)
        ref_direction = self.direction - scaled

        return Ray(point, ref_direction)

    def shadow(self, light, sphere):
        point = sphere.intersect_point(self)
        return Ray(point, light-point)


#Definicija krogel

class Sphere:

    def __init__(self, r, center, ambient, diffuse, specular=np.array([1,1,1]), shininess=100, reflection=0.5):
        self.r = r
        self.center = center
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection

    def __str__(self):
        return 'r = {0}, center = {1}, colour = {2}, reflect = {3}'.format(self.r, self.center, self.ambient, self.reflect)

    def __repr__(self):
        return 'Sphere({0},{1},{2},{3})'.format(self.r, self.center, self.ambient, self.reflect)

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


#Blinn-Phong model senčenja

def blinnphong(light, ray, sphere):
    I = np.zeros(3)
    n = sphere.intersect_normal(ray)
    l = (light['pos'] - sphere.intersect_point(ray)).normalize()
    v = ray.direction.scale(-1)

    I = sphere.ambient * light['ambient']
    I += sphere.diffuse * light['diffuse'] * l.dotproduct(n)
    I += sphere.specular * light['specular'] * n.dotproduct((l+v).normalize()) ** (sphere.shininess / 4)
    return I


#Začne brat podatke

f = open("podatki.txt", 'r')
lines = f.readlines()

w_str, h_str = lines[1].split()
width, height = int(w_str), int(h_str)
cam_pos = tuple(lines[2].split('=')[-1].strip('()\n ').split(','))
source_pos = tuple(lines[3].split('=')[-1].strip('()\n ').split(','))

#objekti
krogla1 = Sphere(0.7, Vector(-0.2, 0, -1), np.array([0.1, 0, 0]), np.array([0.7, 0, 0]), np.array([1,1,1]), 100, 0.5)
krogla2 = Sphere(0.1, Vector(0.1, -0.3, 0), np.array([0.1, 0, 0.1]), np.array([0.7, 0, 0.7]), np.array([1,1,1]), 100, 0.5)
krogla3 = Sphere(0.15, Vector(-0.3, 0, 0), np.array([0, 0.1, 0]), np.array([0, 0.6, 0]), np.array([1,1,1]), 100, 0.5)
ravnina = Sphere(9000-0.7, Vector(0, -9000, 0), np.array([0.1, 0.1, 0.1]), np.array([0.6, 0.6, 0.6]), np.array([1,1,1]), 100, 0.5)

spheres = [krogla1, krogla2, krogla3, ravnina]

#definirana kamera in zaslon

camera = Vector(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
source = Vector(float(source_pos[0]), float(source_pos[1]), float(source_pos[2]))
ratio = width / height
screen = (-1, 1, -1/ratio, 1/ratio) # L R B T

light = {'pos': Vector(5,5,5), 'ambient': np.array([1,1,1]), 'diffuse': np.array([1,1,1]), 'specular': np.array([1,1,1])}

#sestavljanje slike

image = np.zeros((height, width, 3))

for i, y in enumerate(np.linspace(screen[3], screen[2], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[1], width)):
        min_lambda = float('inf')
        intersection_object = -1        
        shadow = False
        RGB = np.zeros(3)

        #izračun žarka
        current_pixel = Vector(x, y, 0)
        current_ray = Ray(camera, current_pixel - camera)

        #ali zadane objekt
        for n, c_sphere in enumerate(spheres):
            c_intersection = c_sphere.intersect(current_ray)
            if c_intersection['ok'] == True:
                if c_intersection['in'] < min_lambda:
                    min_lambda = c_intersection['in']
                    intersection_object = n
                    intersection = c_intersection

        #računanje barve pixla
        if intersection_object > -1:
            #image[i,j] = spheres[intersection_object].ambient * light['ambient']
            shadow_ray = current_ray.shadow(light['pos'], spheres[intersection_object])
            intersection_normal = spheres[intersection_object].intersect_normal(current_ray)
            for m, m_sphere in enumerate(spheres):
                #if m != intersection_object:
                if m_sphere.intersect(shadow_ray)['ok']:
                    shadow = True
        
            #če je v senci samo ambientna svetloba
            if shadow == True:
                RGB = spheres[intersection_object].ambient * light['ambient']

            #če ni v senci blinn-phong
            else:
                RGB = blinnphong(light, current_ray, spheres[intersection_object])

            #odsev ostalih teles



        image[i,j] = np.clip(RGB, 0, 1)

        print("preračunavam: {}%".format((i*width+j+1)/(height*width)*100))

plt.imsave('image.png', image)




#samo za testiranje
#pos = Vector(1,1,10)
#spheres = [Sphere(1, pos, (123, 34, 5), 0.5), Sphere(1, Vector(3, 6, 5), (0, 134, 50), 0.5)]
#zarek = Ray(Vector(-1,-1,0), Vector(1, 1, 0))
#print(source)
#print(camera)
#print(krogla)
#print(zarek)
#print(krogla.intersect_point(zarek))
#print(zarek.reflect(krogla))
