#Red convolucional

import tensorflow as tf

class CNN:
    """
    Clase para crear y compilar una red convolucional simple (CNN).
    """
    def __init__(self, input_shape=(28, 28, 1)):
        """
        Inicializa la arquitectura de la red convolucional.
        Args:
            input_shape (tuple): Forma de entrada para los datos de imágenes.
        """
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # Capa convolucional
            tf.keras.layers.MaxPooling2D((2, 2)),                  # Max pooling
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # Segunda capa convolucional
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),                             # Aplanamiento para capa densa
            tf.keras.layers.Dense(64, activation='relu'),          # Capa densa intermedia
            tf.keras.layers.Dense(10, activation='softmax')        # Salida con 10 clases
        ])

    def compile_model(self, learning_rate=0.001):
        """
        Compila el modelo con función de pérdida y optimizador.
        Args:
            learning_rate (float): Tasa de aprendizaje para el optimizador Adam.
        """
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=['accuracy']
        )

    def get_model(self):
        """
        Devuelve el modelo compilado.
        Returns:
            tf.keras.Model: Modelo compilado.
        """
        return self.model
