import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from models.cnn import CNN
from models.rnn import RNN
from models.simple_nn import SimpleNN

from datasets.data_loader import load_data
from datasets.preprocess import preprocess_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Oculta mensajes de INFO y WARNING

# Función para seleccionar el modelo
def select_model(model_type, input_shape):
    """
    Selecciona y devuelve el modelo deseado.

    Args:
        model_type (str): Tipo de modelo ('simple_nn', 'cnn', 'rnn').
        input_shape (tuple): Forma de entrada del modelo.

    Returns:
        tf.keras.Model: Modelo compilado.
    """
    if model_type == 'simple_nn':
        model = SimpleNN(input_shape=input_shape)
    elif model_type == 'cnn':
        model = CNN(input_shape=input_shape)
    elif model_type == 'rnn':
        model = RNN(input_shape=input_shape)
    else:
        raise ValueError("Modelo no válido. Selecciona entre 'simple_nn', 'cnn', o 'rnn'.")
    
    model.compile_model()
    return model.get_model()

# Cargar los datos desde el CSV
data = load_data('datasets/celsius_fahrenheit_data.csv')

# Preprocesar los datos
X_train, X_test, y_train, y_test = preprocess_data(data)

# Elegir el modelo (cambiar 'simple_nn', 'cnn', o 'rnn' según el caso)
model_type = 'simple_nn'  # Cambiar según el modelo deseado
input_shape = (1,) if model_type == 'simple_nn' else (None, 1)  # Ajustar según datos

model = select_model(model_type, input_shape)

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=500, verbose=False, validation_data=(X_test, y_test))
print("Finished training the model")

# Mostrar estadísticas del entrenamiento
plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

# Usar el modelo para predecir valores (solo para SimpleNN)
if model_type == 'simple_nn':
    prediction = model.predict(np.array([[100.0]]))  # Usar np.array para la entrada
    print(f"Predicted value for 100°C: {prediction[0][0]} °F")
    print("These are the layer variables: {}".format(model.get_weights()))