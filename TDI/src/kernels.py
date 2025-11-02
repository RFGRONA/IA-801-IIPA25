# kernels.py
import numpy as np

"""
Este archivo almacena las matrices (kernels) para los filtros de convolución,
basados en la presentación de la clase .
"""

KERNELS = {
    # Un kernel básico que no hace nada, solo devuelve la imagen original.
    "Identidad": np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]),
    
    # Kernel de Enfoque (Sharpen) 
    "Enfoque (Sharpen)": np.array([
        [0, -1, 0],
        [-1,  5, -1],
        [0, -1, 0]
    ]),
    
    # Kernel de Desenfoque (Box Blur) 
    "Desenfoque (Box Blur)": np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]),
    
    # Kernel de Detección de Bordes (Laplace) 
    "Detección de Bordes (Laplace)": np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ]),
    
    # Kernel de Repujado (Emboss) 
    "Repujado (Emboss)": np.array([
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2]
    ]),
    
    # Kernel de Filtro Sobel (Vertical) 
    "Filtro Sobel (Vertical)": np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]),
    
    # Kernel de Filtro Sobel (Horizontal)
    "Filtro Sobel (Horizontal)": np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]),
    
    # Filtro Norte 
    "Filtro Norte": np.array([
        [ 1,  1,  1],
        [ 1, -2,  1],
        [-1, -1, -1]
    ]),
    
    # Filtro Este 
    "Filtro Este": np.array([
        [-1, 1, 1],
        [-1, -2, 1],
        [-1, 1, 1]
    ]),
    
    # Filtro Gauss (5x5) - Copiado exacto de la diapositiva 
    "Filtro Gauss (5x5)": np.array([
        [1, 2, 3, 1, 1],
        [2, 7, 11, 7, 2],
        [3, 11, 17, 11, 3],
        [2, 7, 11, 7, 1],
        [1, 2, 3, 2, 1]
    ])
}