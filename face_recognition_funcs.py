#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.express as px
import requests
import face_recognition
from typing import List
from tqdm.auto import tqdm

# # Test

# In[2]:


import plotly.graph_objects as go
import numpy as np

import cv2


def resize(image, dsize_min=512):
    min_axis_size = np.asanyarray(image.shape[:-1]).min()
    if min_axis_size < dsize_min:
        return image
    ratio = min_axis_size / dsize_min
    new_size = (np.asanyarray(image.shape[:-1]) / ratio).astype(int)
    return cv2.resize(image, dsize=new_size, interpolation=cv2.INTER_CUBIC)


def face_locations_effective(x):
    return (face_recognition.face_locations(resize(x, 256)) or
            face_recognition.face_locations(resize(x, 512)) or
            face_recognition.face_locations(resize(x, 1024)) or
            face_recognition.face_locations(resize(x, 256), model='cnn'))


def face_locations_to_axis_dict(face_location):
    (top, right, bottom, left) = face_location
    rect = dict(zip(('y0', 'x1', 'y1', 'x0'), (top, right, bottom, left)))
    return rect


def face_locations_to_position_dict(face_location):
    (top, right, bottom, left) = face_location
    rect = dict(zip(('top', 'right', 'bottom', 'left'), (top, right, bottom, left)))
    return rect


def plot_face_rect(image, locations, names=None, color='blue', fig=None):
    if (names is None) or len(names)!=len(locations):
        names = None
    if fig is None:
        fig = px.imshow(image)
    for i in range(len(locations)):
        rect = face_locations_to_axis_dict(locations[i])
        if names:
            fig.add_trace(go.Scatter(
                x=[(rect['x0'] + rect['x1'])/2],
                y=[rect['y0']-20],
                text=[names[i]],
                mode="text",
                textfont=dict(
                    color=color,
                    size=20,
                    family="Arial",
                )
            ))
        fig.add_shape(
            dict(type="rect", **rect, line_color=color),
            row="all",
            col="all",
    )
    return fig


def crop_faces(image, locations):
    cropped_images = []
    for location in locations:
        axis_dict = face_locations_to_axis_dict(location)
        cropped_image = image[axis_dict['y0']:axis_dict['y1'], axis_dict['x0']:axis_dict['x1']]
        cropped_images.append(cropped_image)
    return cropped_images


def plot_cropped_faces(image, locations):
    cropped_faces = crop_faces(image, locations)
    for cropped_face in cropped_faces:
        fig = px.imshow(cropped_face)
        fig.show()
    return cropped_faces


# In[3]:




def get_most_relevant_face(image, face_locations, how: "'center' | 'bigger'" = 'center'):
    if how == 'center':
        center = (image.shape[0] / 2, image.shape[1] / 2)
        distances = []
        for face_location in face_locations:
            axis_dict = face_locations_to_axis_dict(face_location)
            rect_center = ((axis_dict['y1'] + axis_dict['y0']) / 2, (axis_dict['x1'] + axis_dict['x0']) / 2)
            distance = np.linalg.norm(np.asanyarray(rect_center) - np.asanyarray(center))
            distances.append(distance)
        face = face_locations[np.argmin(distances)]
    else:
        sizes = []
        for face_location in face_locations:
            axis_dict = face_locations_to_axis_dict(face_location)
            size = (axis_dict['x1'] - axis_dict['x0']) * (axis_dict['y1'] - axis_dict['y0'])
            sizes.append(size)
        face = face_locations[np.argmax(sizes)]
    return face


# In[4]:


def images_to_encodings(urls: List[str], max_faces=3, min_faces=1, how='center', plot=False):
    results = {}
    for url in tqdm(urls):
        image = face_recognition.load_image_file(requests.get(url, stream=True).raw)
        locations = face_locations_effective(image)
        if len(locations) > max_faces or len(locations) < min_faces:
            print(url)
            # locations = [[0, image.shape[0], image.shape[1], 0]]
            # plot_cropped_faces(image, locations) if plot else None
            # encodings = face_recognition.face_encodings(image, locations)
            results[url] = {
                'total_faces_found': 0,
                'encodings': None,
                'rectangle': None
            }
        else:
            if len(locations) > 1:
                plot_face_rect(image, locations).show() if plot else None
                locations = [get_most_relevant_face(image, locations, how=how)]
            plot_cropped_faces(image, locations) if plot else None
            encodings = face_recognition.face_encodings(image, locations)
            results[url] = {
                'total_faces_found': len(locations),
                'encodings': encodings[0],
                'rectangle': face_locations_to_axis_dict(locations[0])
            }
    return results


def find_matching_image(url: str, encodings: dict, max_encodings=1, min_encodings=1, how='center'):
    image = face_recognition.load_image_file(requests.get(url, stream=True).raw)
    locations = face_locations_effective(image)
    if locations is None:
        return {
            'rectangle': None,
            'distance': None,
            'matching_url': None,
            'encodings': None,
            'total_faces_found': 0
        }
    location = get_most_relevant_face(image, locations, how=how)
    image_encodings = face_recognition.face_encodings(image, known_face_locations=[location])[0]
    encodings_series = pd.Series(encodings, name='encodings')
    distances = face_recognition.face_distance(np.stack(encodings_series.values), image_encodings)
    distance_series = pd.Series(data=distances, index=encodings_series.index, name='distances')
    return {
        'rectangle': face_locations_to_axis_dict(location),
        'distance': distance_series.min(),
        'matching_url': distance_series.idxmin(),
        'encodings': image_encodings,
        'total_faces_found': len(locations)
    }


def find_matching_user(url: str, users_encodings: List[List[str]]):
    pass

# In[5]:


# from typing import List
# def get_encodings(input_image, min_encodings=1, max_encodings=9999):    
#     locations = face_recognition.face_locations(image, model='hog') or face_recognition.face_locations(image, model='cnn')
#     print(len(locations))
#     if len(locations) >= min_encodings and len(locations)<=max_encodings:
#         encodings = face_recognition.face_encodings(image, locations)
#         return encodings

# def compare_faces(input_image: str, real_person_image: str):
#     person_encodings = get_encodings(real_person_image)
#     input_image_encodings = get_encodings(input_image)
#     if bool(input_image_encodings) and bool(person_encodings):
#         return face_recognition.face_distance(person_encodings, input_image_encodings[0])

# compare_faces(image4, image5)
