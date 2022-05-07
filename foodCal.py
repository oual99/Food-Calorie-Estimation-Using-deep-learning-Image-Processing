import cv2
import numpy as np
from foodArea import *

#density - gram / cm^3
density_dict = { 'Apple':0.609, 'Banana':0.94, 'Cucumber':0.641, 'Onion':0.513, 'Orange':0.482, 'Tomato':0.481 }
#kcal
calorie_dict = { 'Apple':52, 'Banana':89, 'Cucumber':16, 'Onion':40, 'Orange':47, 'Tomato':18 }
#skin of photo to real multiplier
skin_multiplier = 5*2.3

def getMacnutr(label):
	if label == 'Apple':
		T1= {0:96,1:3,2:1}
		T2= {0:13.8,1:0.2,2:0.3}
	if label == 'Banana':
		T1= {0:93,1:3,2:4}
		T2= {0:22.8,1:0.3,2:1.1}
	if label == 'Cucumber':
		T1= {0:80,1:5,2:15}
		T2= {0:3.6,1:0.1,2:0.7}
	if label == 'Onion':
		T1= {0:88,1:2,2:10}
		T2= {0:9.3,1:0.1,2:1.1}
	if label == 'Orange':
		T1= {0:91,1:2,2:7}
		T2= {0:11.7,1:0.1,2:0.9}
	if label == 'Tomato':
		T1= {0:75,1:9,2:16}
		T2= {0:3.6,1:0.2,2:0.9}
	
	return T1, T2

def getCalorie(label, volume): #volume in cm^3
	calorie = calorie_dict[label]
	density = density_dict[label]
	mass = volume*density*1.0
	calorie_tot = (calorie/100.0)*mass
	return mass, calorie_tot, calorie #calorie per 100 grams

def getCalorie1(label, mass): 
	calorie = calorie_dict[label]
	calorie_tot = (calorie/100.0)*mass
	return calorie_tot 


def getVolume(label, area, skin_area, pix_to_cm_multiplier, fruit_contour):
	area_fruit = (area/skin_area)*skin_multiplier #area in cm^2
	volume = 100
	if label == "Apple" or label == "Onion" or label == "Tomato" or label == "Orange" : #sphere-apple,tomato,orange,onion
		radius = np.sqrt(area_fruit/np.pi)
		volume = (4/3)*np.pi*radius*radius*radius
		#print (area_fruit, radius, volume, skin_area)
	
	if label == "Cucumber" or label == "Banana" or (label == 3 and area_fruit > 30): #cylinder like banana, cucumber, carrot
		fruit_rect = cv2.minAreaRect(fruit_contour)
		height = max(fruit_rect[1])*pix_to_cm_multiplier
		radius = area_fruit/(2.0*height)
		volume = np.pi*radius*radius*height
	
	return volume



def calories(result,img_path):
    fruit_areas,final_f,areaod,skin_areas, fruit_contours, pix_cm = getAreaOfFood(img_path)
    volume = getVolume(result, fruit_areas, skin_areas, pix_cm, fruit_contours)
    mass, cal, cal_100 = getCalorie(result, volume)
    fruit_volumes=volume
    fruit_calories=cal
    fruit_calories_100grams=cal_100
    fruit_mass=mass
    print("\nfruit_volumes:",fruit_volumes,"\nfruit_calories:",fruit_calories,"\nfruit_calories_100_grams: ",fruit_calories_100grams,"\nfruit_mass:",fruit_mass)
    return fruit_calories, fruit_mass

