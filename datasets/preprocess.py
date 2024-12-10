import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(data, test_size=0.2, random_state=42):

    """
    Preprocesa los datos dividiendo en entrenamiento y prueba.

    Args:
        data (pd.DataFrame): DataFrame con columnas 'Celsius' y 'Fahrenheit'.
        test_size (float): Proporción de los datos para el conjunto de prueba.
        random_state (int): Semilla para la división aleatoria.

    Returns:
        tuple: Tupla que contiene:
            - X_train (np.ndarray): Características de entrenamiento.
            - X_test (np.ndarray): Características de prueba.
            - y_train (np.ndarray): Etiquetas de entrenamiento.
            - y_test (np.ndarray): Etiquetas de prueba.
    """
    x = data['Celsius'].values.reshape(-1, 1) #Convertir a matriz 2D
    y = data['Fahrenheit'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test