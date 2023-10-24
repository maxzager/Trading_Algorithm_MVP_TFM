import pandas as pd
import numpy as np
import pandas_ta as ta

class DataHandler:
    def __init__(self, file_name, nrows=None, skip_inirows=None, tick_data=False):
        self.data = self.read_datacsv(file_name, nrows, skip_inirows, tick_data)

        
    def read_datacsv(self, file_name, nrows=None, skip_inirows=None, tick_data=False):
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
    def generate_volumebars(self, volume_threshold, tick_data=False):
        """
        Genera un DataFrame con volume_bars, high es el maximo, low el minimo, close el ultimo.
        Inputs:
            data: Dataframe con datos de operaciones (date, ticks, price)
            
            volume_threshold: Tamano del volumen para calculo
            
        Output:
            volume_bars: DataFrame con los datos OHLCV calculados cada volume_threshold
        """
        df = self.data.copy()
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
                open=('open', 'first'),
                high=('high', 'max'),
                low=('low', 'min'),
                close=('close', 'last'),
                volume=('volume', 'sum')
            ).reset_index(drop=True)  

        volume_bars.index = volume_bars['datetime']
        
        volume_bars = volume_bars.drop(['datetime'], axis=1)  
        
        return volume_bars   

class FeaturesGenerator:
    def __init__(self, data:pd.DataFrame, momentum:bool, volume:bool, volatility:bool):
        self.data = data
        self.momentum = momentum
        self.volume = volume
        self.volatility = volatility
        self.df = pd.DataFrame()
        self.df.index = self.data.index
        self.generate_features()
    
    def momentum_features(self):

        AO = ta.momentum.ao(
                self.data["high"],
                self.data["low"])
        self.df = self.df.join(AO)
                
        BOP = ta.momentum.bop(
                self.data["open"],
                self.data["high"],
                self.data["low"],
                self.data["close"])
        self.df = self.df.join(BOP)
        
        BRAR = ta.momentum.brar(
                self.data["open"],
                self.data["high"],
                self.data["low"],
                self.data["close"])
        self.df = self.df.join(BRAR)

        CCI = ta.momentum.cci(
                self.data["high"],
                self.data["low"],
                self.data["close"])
        self.df = self.df.join(CCI)
        
        CFO = ta.momentum.cfo(
                self.data["close"])
        self.df = self.df.join(CFO)
        
        CG = ta.momentum.cg(
                self.data["close"])
        self.df = self.df.join(CG)
        
        CMO = ta.momentum.cmo(
                self.data["close"])
        self.df = self.df.join(CMO)
        
        COPC = ta.momentum.coppock(
                self.data["close"])
        self.df = self.df.join(COPC)
        
        CTI = ta.momentum.cti(
                self.data["close"])
        self.df = self.df.join(CTI)

        DM = ta.momentum.dm(
                self.data["high"],
                self.data["low"])
        self.df = self.df.join(DM)

        ER = ta.momentum.er(
                self.data["close"])
        self.df = self.df.join(ER)
        
        ERI = ta.momentum.eri(
                self.data["high"],
                self.data["low"],
                self.data["close"])
        self.df = self.df.join(ERI)
        
        FISHER = ta.momentum.fisher(
                self.data["high"],
                self.data["low"])
        self.df = self.df.join(FISHER)
        
        INERTIA = ta.momentum.inertia(
                self.data["close"],
                self.data["high"],
                self.data["low"])
        self.df = self.df.join(INERTIA)
        
        KDJ = ta.momentum.kdj(
                self.data["high"],
                self.data["low"],
                self.data["close"])
        self.df = self.df.join(KDJ)
        
        KST = ta.momentum.kst(
                self.data["close"])
        self.df = self.df.join(KST)
        
        MACD = ta.momentum.macd(
                self.data["close"])
        self.df = self.df.join(MACD)
        
        MOM = ta.momentum.mom(
                self.data["close"])
        self.df = self.df.join(MOM)
        
        PGO = ta.momentum.pgo(
                self.data["high"],
                self.data["low"],
                self.data["close"])
        self.df = self.df.join(PGO)

        PPO = ta.momentum.ppo(
                self.data["close"])
        self.df = self.df.join(PPO)
        
        PSL = ta.momentum.psl(
                self.data["close"])
        self.df = self.df.join(PSL)
        
        QQE = ta.momentum.qqe(
                self.data["close"])
        self.df = self.df.join(QQE)
        
        RSI = ta.momentum.rsi(
                self.data["close"])
        self.df = self.df.join(RSI)
        
        RSX = ta.momentum.rsx(
                self.data["close"])
        self.df = self.df.join(RSX)
        
        RVGI = ta.momentum.rvgi(
                self.data["open"],
                self.data["high"],
                self.data["low"],
                self.data["close"])
        self.df = self.df.join(RVGI)
        
        SLOPE = ta.momentum.slope(
                self.data["close"])
        self.df = self.df.join(SLOPE)
        
        SMI = ta.momentum.smi(
                self.data["close"])
        self.df = self.df.join(SMI)
        
        SQUEEZE = ta.momentum.squeeze(
                self.data["high"],
                self.data["low"],
                self.data["close"])
        self.df = self.df.join(SQUEEZE)
        
        SQZPRO = ta.momentum.squeeze_pro(
                self.data["high"],
                self.data["low"],
                self.data["close"])
        self.df = self.df.join(SQZPRO)
              
        STOCH = ta.momentum.stoch(
                self.data["high"],
                self.data["low"],
                self.data["close"])
        self.df = self.df.join(STOCH)
        
        STOCHRSI = ta.momentum.stochrsi(
                self.data["close"])
        self.df = self.df.join(STOCHRSI)
        
        TRIX = ta.momentum.trix(
                self.data["close"])
        self.df = self.df.join(TRIX)
        
        TSI = ta.momentum.tsi(
                self.data["close"])
        self.df = self.df.join(TSI)
        
        UO = ta.momentum.uo(
                self.data["high"],
                self.data["low"],
                self.data["close"])
        self.df = self.df.join(UO)
        
        WILLR = ta.momentum.willr(
                self.data["high"],
                self.data["low"],
                self.data["close"])
        self.df = self.df.join(WILLR)
        
    def volume_features(self):
        pass

    def volatility_features(self):
        pass

    def generate_features(self):
        if self.momentum:
                self.momentum_features()
                        
        if self.volume:
                self.volume_features()
                
        if self.volatility:
                self.volatility_features()
                



class Labeler:

    def __init__(self, data:pd.DataFrame, horizons:list, betas:list, periods:list):
        self.data = data
        self.horizons = horizons
        self.betas = betas
        self.periods = periods

        self.df = pd.DataFrame()
        self.df.index = self.data.index
        self.run_labeler()

    def labelling_ATR(self, horizon, beta, period):
        atr = self.calculate_atr(self.data, period)
        uper_barrier = self.data.high + beta*atr
        lower_barrier = self.data.low - beta*atr

        labels = np.full(len(self.data), np.nan)

        
        for i in range(len(self.data) - horizon):
                series = self.data[i:i+horizon]
                pre_labels =  np.where(series.high >= uper_barrier.iloc[i], 1,
                            (np.where(series.low <= lower_barrier.iloc[i], -1, 0)))
            
            # Si hay alguna etiqueta distinta de 0,
            # quiero la primera en suceder.
                if any(pre_labels != 0):
                    labels[i] = pre_labels[pre_labels != 0][0]
                else:
                    labels[i] = 0
            
        return labels
    
    def labelling_volat(self, horizon, beta, period):
        volat = self.data.close.pct_change()
        volat = volat.rolling(period).std()
        uper_barrier = self.data.high * (1+beta*volat)
        lower_barrier = self.data.low * (1-beta*volat)
        
        labels = np.full(len(self.data), np.nan)

        for i in range(len(self.data)-horizon):
                series = self.data[i:i+horizon]
                pre_labels =  np.where(series.high >= uper_barrier.iloc[i], 1,
                            (np.where(series.low <= lower_barrier.iloc[i], -1, 0)))
            # Si hay alguna etiqueta distinta de 0,
            # quiero la primera en suceder.
                if any(pre_labels != 0):
                    labels[i] = pre_labels[pre_labels != 0][0]
                else:
                    labels[i] = 0

        return labels
    
    def run_labeler(self):

        for h in self.horizons:
            for b in self.betas:
                for p in self.periods:
                    labels_atr = self.labelling_ATR(h, b, p)
                    self.df[f"h_{h}, b_{b}, p_{p}, ATR"] = labels_atr
                    labels_volat = self.labelling_volat(h, b, p)
                    self.df[f"h_{h}, b_{b}, p_{p}, V"] = labels_volat
                
    
    #funcion auxiliar
    def calculate_atr(self, data, period):
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