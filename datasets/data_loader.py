import pandas as pd
import os

def load_data(file_path):
    """
    Carga los datos desde un archivo CSV y los devuelve como un DataFrame.

    Args:
        file_path (str): Ruta al archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    
    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo no tiene las columnas necesarias.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")
    
    data = pd.read_csv(file_path)
    
    # Verificar que las columnas necesarias existan
    required_columns = ['Celsius', 'Fahrenheit']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"El archivo debe contener las columnas: {', '.join(required_columns)}")
    
    return data
