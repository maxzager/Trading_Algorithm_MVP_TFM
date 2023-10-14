import pandas as pd
import numpy as np

def read_datacsv(file_name, nrows=None, skip_inirows=None, tick_data=False):
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
    if tick_data is True:
        df = pd.read_csv(file_name,
                         nrows=nrows,
                         skiprows=skip_inirows,
                         header=0,
                         names=['date', 'time', 'price', 'volume'],
                         index_col=False)
    else:
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
def generate_volumebars(data, volume_threshold, tick_data=False):
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

    if tick_data is True:
        volume_bars = df.groupby('volume_bin').agg(
            datetime=('datetime', 'first'),
            open=('price', 'first'),
            high=('price', 'max'),
            low=('price', 'min'),
            close=('price', 'last'),
            volume=('volume', 'sum')

        ).reset_index(drop=True)
    else:

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

class labeler:

    def __init__(self, data:pd.DataFrame, tick_data:bool, vol_thresholds:list, horizons:list, betas:list, periods:list):
        self.data = data
        self.tick_data = tick_data
        self.vol_thresholds = vol_thresholds
        self.horizons = horizons
        self.betas = betas
        self.periods = periods

    def labelling_ATR(self, df, horizon, beta, period):
        atr = calculate_atr(df, period)
        uper_barrier = df.high + beta*atr
        lower_barrier = df.low - beta*atr

        labels = list()
        labeled_data = df.copy()

        
        for i in range(len(df)-horizon):

            series = df[i:i+horizon]
            pre_labels =  np.where(series.high >= uper_barrier.iloc[i], 1,
                           (np.where(series.low <= lower_barrier.iloc[i], -1, 0)))
        
        # Si hay alguna etiqueta distinta de 0,
        # quiero la primera en suceder.
            if any(pre_labels != 0):
                labels.append(pre_labels[pre_labels != 0][0])
            else:
                labels.append(0)
        labeled_data.drop(labeled_data.index[-horizon:],axis=0, inplace=True) # Pierdo los ultimos "length" datos
        labeled_data["labels"] = np.array(labels)    

        return labeled_data
    
    def labelling_volat(self, df, horizon, beta, period):
        volat = df.close.pct_change()
        volat = volat.rolling(period).std()
        uper_barrier = df.high * (1+beta*volat)
        lower_barrier = df.low - (1+beta*volat)
        
        labels = list()
        labeled_data = df.copy()

        for i in range(len(df)-horizon):
            
            series = df[i:i+horizon]
            pre_labels =  np.where(series.high >= uper_barrier.iloc[i], 1,
                           (np.where(series.low <= lower_barrier.iloc[i], -1, 0)))
        # Si hay alguna etiqueta distinta de 0,
        # quiero la primera en suceder.
            if any(pre_labels != 0):
                labels.append(pre_labels[pre_labels != 0][0])
            else:
                labels.append(0)
        labeled_data.drop(labeled_data.index[-horizon:],axis=0, inplace=True) # Pierdo los ultimos "length" datos
        labeled_data["labels"] = np.array(labels)    

        return labeled_data
    

    def run_labeler(self):
        output = dict()
        for i in self.vol_thresholds:
            df = generate_volumebars(self.data, i, self.tick_data)

            ### ACA HAY QUE LLAMAR FUNCION QUE AGREGA LOS FEATURES

            for h in self.horizons:
                for b in self.betas:
                    for p in self.periods:
                        labels_atr = self.labelling_ATR(df, h, b, p)
                        output[f"v_{i}, h_{h}, b_{b}, p_{p}, ATR"] = labels_atr
                        labels_volat = self.labelling_volat(df, h, b, p)
                        output[f"v_{i}, h_{h}, b_{b}, p_{p}, volat"] = labels_volat
        
        return output
    
