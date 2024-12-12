#Red recurrente

import tensorflow as tf

class RNN:
    """
    Clase para crear y compilar una red recurrente básica (RNN).
    """
    def __init__(self, input_shape=(None, 1)):
        """
        Inicializa la arquitectura de la red recurrente.
        Args:
            input_shape (tuple): Forma de entrada para datos secuenciales.
        """
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.SimpleRNN(64, activation='relu', return_sequences=True),  # Capa RNN
            tf.keras.layers.SimpleRNN(32, activation='relu'),                        # Segunda capa RNN
            tf.keras.layers.Dense(1)                                                # Salida
        ])

    def compile_model(self, learning_rate=0.001):
        """
        Compila el modelo con función de pérdida y optimizador.
        Args:
            learning_rate (float): Tasa de aprendizaje para el optimizador Adam.
        """
        self.model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate)
        )

    def get_model(self):
        """
        Devuelve el modelo compilado.
        Returns:
            tf.keras.Model: Modelo compilado.
        """
        return self.model
