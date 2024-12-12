import tensorflow as tf

class SimpleNN:
    """
    Clase para crear y compilar una red neuronal básica (fully connected).
    """
    def __init__(self, input_shape=(1,)):
        """
        Inicializa la arquitectura de la red neuronal.
        Args:
            input_shape (tuple): Forma de entrada para la red.
        """
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Dense(units=1)  # Capa densa con una salida
        ])

    def compile_model(self, learning_rate=0.1):
        """
        Compila el modelo con una función de pérdida y un optimizador.
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
