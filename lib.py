import pandas as pd


def read_datacsv(file_name, nrows=None, skip_inirows=None):
    """
    Esta funcion permite lleer un archivo .csv desde el directorio de trabajo actual.
    Trabaja sobre los datos y los entrega en formato de DataFrame de Pandas.
    Inputs:
        file_name:    Nombre del file.csv a leer
        nrows:  Filas a leer (largo de datos a leer)
        skip_inirows: Filas a saltarse al inicio
    Outputs:
        df: DataFrame con los datos leidos del csv
    """
    # Carga datos desde csv
    df = pd.read_csv(file_name, 
                       nrows=nrows,                             
                       skiprows=skip_inirows, 
                       header=0, 
                       names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'], 
                       index_col=False) 
    
    # Agregar zeros para el tiempo    
    df['time'] = df['time'].astype(str).str.zfill(4)

    # Formato datetime
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time']) 

    # Elimina columnas que no se usan       
    df = df.drop(['date', 'time'], axis=1) 

    # Ordena por Fechas                   
    df = df.sort_values(by=['datetime'], axis=0, ascending=True)     

    # Deja fecha como indice         
    df = df.set_index('datetime') 
                                          
    return df

# Funcion Calculo Volume Bars
def generate_volumebars(data, volume_threshold):
    """
    Genera un DataFrame con volume_bars, high es el maximo, low el minimo, close el ultimo.
    Inputs:
        data: Dataframe con datos de operaciones (date, ticks, price)
        
        volume_threshold: Tamano del volumen para calculo
        
    Output:
        volume_bars: DataFrame con los datos OHLCV calculados cada volume_threshold
    """
    df = data.copy()
    df['cumulative_volume'] = df['volume'].cumsum()
    df['volume_bin'] = (df['cumulative_volume'] // volume_threshold).astype(int)
    df['datetime'] = df.index

    volume_bars = df.groupby('volume_bin').agg(
        datetime=('datetime', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum')
    ).reset_index(drop=True)  

    volume_bars.index = volume_bars['datetime']
    
    volume_bars = volume_bars.drop(['datetime'], axis=1)  
    
    return volume_bars   