import pandas as pd
import numpy as np

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

def calculate_atr(data, period):
    """
    Calculate the Average True Range (ATR)

    Args:
        data (pd.DataFrame): DataFrame with 'high', 'low', and 'close' columns
        period (int): The period over which to calculate the ATR

    Returns:
        pd.Series: A pandas Series containing the ATR
    """
    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift()).abs()
    low_close = (data['low'] - data['close'].shift()).abs()

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()

    return atr

def labelling(data, length=None, dynamic=True, period=None, beta=None, height=None):
    """
    This function labels the data based on the given parameters.

    Args:
        data (pd.DataFrame): DataFrame with 'high', 'low', and 'close' columns
        length (int): The length of the barriers
        dynamic (bool): If True, the labelling is dynamic using ATR
        period (int): The period over which to calculate the ATR
        beta (float): Multiplier of ATR for calculating the upper and lower barriers
        height (float): The height for calculating the upper and lower bands when dynamic is False

    Returns:
        pd.DataFrame: A DataFrame with the labelled data
    """
    datacopy = data.copy()
    atr = calculate_atr(datacopy, period)
    output = list()

    if dynamic:
        
        # average true range
    
        for i in range(len(datacopy)-length):

            series = datacopy[i:i+length]
            uper_barrier = series.high.iloc[0] + beta*atr.iloc[i]
            lower_barrier = series.low.iloc[0] - beta*atr.iloc[i]
            pre_labels =  np.where(series.high >= uper_barrier, 1,
                           (np.where(series.low <= lower_barrier, -1, 0)))
        
        # Si hay alguna etiqueta distinta de 0,
        # quiero la primera en suceder.
            if any(pre_labels != 0):
                output.append(pre_labels[pre_labels != 0][0])
            else:
                output.append(0)
        datacopy.drop(datacopy.index[-length:],axis=0, inplace=True) # Pierdo los ultimos "length" datos
        datacopy["labels"] = np.array(output)    

        return datacopy
    
    else:
        for i in range(len(datacopy)-length):

            series = datacopy[i:i+length]
            upper_band = series.close.iloc[0]*(1+height/100) 
            lower_band = series.close.iloc[0]*(1-height/100)
            pre_labels =  np.where(series.high >= upper_band, 1,
                           (np.where(series.low <= lower_band, -1, 0)))

            if any(pre_labels != 0):
                output.append(pre_labels[pre_labels != 0][0])
            else:
                output.append(0)
            
        datacopy.drop(datacopy.index[-length:],axis=0, inplace=True) # Pierdo los ultimos "length" datos
        datacopy["labels"] = np.array(output)
    
        return datacopy
    
