import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
import matplotlib.pyplot as plt

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
		
		# Indicator: Awesome Oscillator (AO)
		AO = ta.momentum.ao(
				self.data["high"],
				self.data["low"])
		self.df = self.df.join(AO)

		# Indicator: Balance of Power (BOP)        
		BOP = ta.momentum.bop(
				self.data["open"],
				self.data["high"],
				self.data["low"],
				self.data["close"])
		self.df = self.df.join(BOP)
		
		# Indicator: BRAR (BRAR)
		BRAR = ta.momentum.brar(
				self.data["open"],
				self.data["high"],
				self.data["low"],
				self.data["close"])
		self.df = self.df.join(BRAR)

		# Indicator: Commodity Channel Index (CCI)
		CCI = ta.momentum.cci(
				self.data["high"],
				self.data["low"],
				self.data["close"])
		self.df = self.df.join(CCI)
		
		# Indicator: Chande Forcast Oscillator (CFO)
		CFO = ta.momentum.cfo(
				self.data["close"])
		self.df = self.df.join(CFO)
		
		# Indicator: Center of Gravity (CG)
		CG = ta.momentum.cg(
				self.data["close"])
		self.df = self.df.join(CG)
		
		# Indicator: Chande Momentum Oscillator (CMO)
		CMO = ta.momentum.cmo(
				self.data["close"])
		self.df = self.df.join(CMO)
		
		# Indicator: Coppock Curve (COPC)
		COPC = ta.momentum.coppock(
				self.data["close"])
		self.df = self.df.join(COPC)
		
		# Indicator: Correlation Trend Indicator (CTI)
		CTI = ta.momentum.cti(
				self.data["close"])
		self.df = self.df.join(CTI)

		# Indicator: Directional Movement (DM)
		DM = ta.momentum.dm(
				self.data["high"],
				self.data["low"])
		self.df = self.df.join(DM)

		# Indicator: Efficiency Ratio (ER)
		ER = ta.momentum.er(
				self.data["close"])
		self.df = self.df.join(ER)
		
		# Indicator: Elder Ray Index (ERI)
		ERI = ta.momentum.eri(
				self.data["high"],
				self.data["low"],
				self.data["close"])
		self.df = self.df.join(ERI)
		
		# Indicator: Fisher Transform 
		FISHER = ta.momentum.fisher(
				self.data["high"],
				self.data["low"])
		self.df = self.df.join(FISHER)
		
		# Indicator: Inertia (INERTIA)
		INERTIA = ta.momentum.inertia(
				self.data["close"],
				self.data["high"],
				self.data["low"])
		self.df = self.df.join(INERTIA)
		
		# Indicator: KDJ (KDJ)
		KDJ = ta.momentum.kdj(
				self.data["high"],
				self.data["low"],
				self.data["close"])
		self.df = self.df.join(KDJ)
		
		# Indicator: 'Know Sure Thing' (KST)
		KST = ta.momentum.kst(
				self.data["close"])
		self.df = self.df.join(KST)
		
		# Indicator: Moving Average, Convergence/Divergence (MACD)
		MACD = ta.momentum.macd(
				self.data["close"])
		self.df = self.df.join(MACD)
		
		# Indicator: Momentum (MOM)
		MOM = ta.momentum.mom(
				self.data["close"])
		self.df = self.df.join(MOM)
		
		# Indicator: Pretty Good Oscillator (PGO)
		PGO = ta.momentum.pgo(
				self.data["high"],
				self.data["low"],
				self.data["close"])
		self.df = self.df.join(PGO)

		# Indicator: Percentage Price Oscillator (PPO)
		PPO = ta.momentum.ppo(
				self.data["close"])
		self.df = self.df.join(PPO)
		
		#Indicator: Psychological Line (PSL)
		PSL = ta.momentum.psl(
				self.data["close"])
		self.df = self.df.join(PSL)
		
		# Indicator: Relative Strength Index (RSI)
		RSI = ta.momentum.rsi(
				self.data["close"])
		self.df = self.df.join(RSI)
		
		# Indicator: Relative Strength Xtra (inspired by Jurik RSX)
		RSX = ta.momentum.rsx(
				self.data["close"])
		self.df = self.df.join(RSX)
		
		# Indicator: Relative Vigor Index (RVGI)
		RVGI = ta.momentum.rvgi(
				self.data["open"],
				self.data["high"],
				self.data["low"],
				self.data["close"])
		self.df = self.df.join(RVGI)
		
		# Indicator: Slope
		SLOPE = ta.momentum.slope(
				self.data["close"])
		self.df = self.df.join(SLOPE)
		
		# Indicator: SMI Ergodic Indicator (SMIIO)
		SMI = ta.momentum.smi(
				self.data["close"])
		self.df = self.df.join(SMI)
		
		# Indicator: Squeeze Momentum (SQZ)
		SQUEEZE = ta.momentum.squeeze(
				self.data["high"],
				self.data["low"],
				self.data["close"])
		self.df = self.df.join(SQUEEZE)
		
		# Indicator: Squeeze Momentum (SQZ) PRO
		SQZPRO = ta.momentum.squeeze_pro(
				self.data["high"],
				self.data["low"],
				self.data["close"])
		self.df = self.df.join(SQZPRO)

		# Indicator: Stochastic Oscillator (STOCH)      
		STOCH = ta.momentum.stoch(
				self.data["high"],
				self.data["low"],
				self.data["close"])
		self.df = self.df.join(STOCH)
		
		# Indicator: Stochastic RSI Oscillator (STOCHRSI)
		STOCHRSI = ta.momentum.stochrsi(
				self.data["close"])
		self.df = self.df.join(STOCHRSI)
		
		# Indicator: Trix (TRIX)
		TRIX = ta.momentum.trix(
				self.data["close"])
		self.df = self.df.join(TRIX)
		
		# Indicator: True Strength Index (TSI)
		TSI = ta.momentum.tsi(
				self.data["close"])
		self.df = self.df.join(TSI)
		
		# Indicator: Ultimate Oscillator (UO)
		UO = ta.momentum.uo(
				self.data["high"],
				self.data["low"],
				self.data["close"])
		self.df = self.df.join(UO)
		
		# Indicator: William's Percent R (WILLR)
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

	def __init__(self, data:pd.DataFrame, horizons:list, betas:list, periods:list, direction:str):
		self.data = data
		self.horizons = horizons
		self.betas = betas
		self.periods = periods
		self.direction = direction
		assert self.direction in ['long', 'short', 'longshort']

		self.df = pd.DataFrame()
		self.df.index = self.data.index
		self.run_labeler()

	def labelling_ATR(self, horizon, beta, period):
		atr = self.calculate_atr(self.data, period)
		uper_barrier = self.data.close + beta*atr
		lower_barrier = self.data.close - beta*atr

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
		uper_barrier = self.data.close * (1+beta*volat) 
		lower_barrier = self.data.close * (1-beta*volat) 
		
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
		if self.direction == 'long':
			# Replace labels -1 with 0
			self.df[self.df == -1] = 0
		elif self.direction == 'short':
			# Replace labels 1 with 0
			self.df[self.df == 1] = 0
			# Replace labels -1 with 1
			self.df[self.df == -1] = 1
				
	
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
	

class DataPreprocessor:
	def __init__(self, ohlc:pd.DataFrame, features:pd.DataFrame, labels:pd.DataFrame, validation_size:float, test_size:float):
		self.ohlc = ohlc
		self.features = features
		self.labels = labels
		self.len_data = len(ohlc)
		self.len_validation = int(self.len_data * validation_size)
		self.len_test = int(self.len_data * test_size)
		
	def train_test_split(self, data):
		train = data.iloc[:-self.len_validation-self.len_test]
		val = data.iloc[-self.len_validation-self.len_test:-self.len_test]
		test = data.iloc[-self.len_test:]
		return train, val, test
		 
	def normalization(self, data):
		max_ohlc = data[['open', 'high', 'low', 'close']].max().max()
		min_ohlc = data[['open', 'high', 'low', 'close']].min().min()
		ohlc_norm = (data[['open', 'high', 'low', 'close']] - min_ohlc) / (max_ohlc - min_ohlc)
		data.loc[:,['open', 'high', 'low', 'close']] = ohlc_norm
		scaler = MinMaxScaler()
		not_ohlc = data.drop(['open', 'high', 'low', 'close'], axis=1)
		not_ohlc_scaled = scaler.fit_transform(not_ohlc)
		data.loc[:,not_ohlc.columns] = not_ohlc_scaled
		return data
		 
	def preprocess(self):
		#join ohlc with features to create X
		data = self.ohlc.join(self.features)
		data_cols = data.columns
		# clean nans from all data to harmonize
		data = data.join(self.labels)
		data = data.dropna(axis=0, how='any')
		# store back clean data separately
		self.labels = data[self.labels.columns]
		data = data[data_cols]
		#train test split
		X_train, X_val, X_test = self.train_test_split(data)
		y_train, y_val, y_test = self.train_test_split(self.labels)
		#normalization
		X_train = self.normalization(X_train)
		X_val = self.normalization(X_val)
		X_test = self.normalization(X_test)
		#
		return X_train, X_val, X_test, y_train, y_val, y_test
			  
class ModelsHandler:
	def __init__(self, model_name, model_type, direction, label_option):
		self.model_name = model_name
		self.model_type = model_type
		self.direction = direction
		self.label_option = label_option
		assert self.direction in ['long', 'short', 'longshort']
		self.metrics_df = None
	
	def preprocess_data_for_model(self, data):
		# Preprocess data based on model type
		if self.model_type == 'Dense':
			return data
		elif self.model_type == 'CNN':
			return data
		elif self.model_type == 'LSTM':
			return data
		else:
			raise ValueError(f'Unknown model type: {self.model_type}')
		
	def train_model(self, build_model_func, X_train, y_train, X_val, y_val, epochs, batch_size, patience, input_shape):
		
		# Preprocess data based on model type
		X_train_processed, y_train_processed = self.preprocess_data_for_model((X_train, y_train))
		X_val_processed, y_val_processed = self.preprocess_data_for_model((X_val, y_val))
		
		# Build model
		model = build_model_func(input_shape=input_shape)
		
		# Compile model
		if self.direction == 'long' or self.direction == 'short':
			loss = 'binary_crossentropy'
		elif self.direction == 'longshort':
			loss = 'categorical_crossentropy'
		model.compile(optimizer="adam",
					loss=loss,
					metrics=['accuracy'])

		# Define callbacks
		checkpoint_path = f"model_weights/{self.model_name}_{self.label_option}_best_weights.h5"
		checkpoint = ModelCheckpoint(
			checkpoint_path, 
			monitor='val_accuracy', 
			verbose=1, 
			save_best_only=True, 
			mode='max')
		early_stopping = EarlyStopping(
			monitor='val_accuracy', 
			patience=patience, 
			verbose=1, 
			mode='max')
		history_callback = History()

		# Train the model
		self.history = model.fit(X_train_processed, y_train_processed,
							validation_data=(X_val_processed, y_val_processed),
							epochs=epochs, 
							batch_size=batch_size,
							callbacks=[checkpoint, early_stopping, history_callback],
							verbose=1)
	
	def plot_learning_curves(self):
		# Plot for loss
		plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot = top subplot

		# Plot training loss
		plt.plot(self.history.history['loss'], label='Training Loss')

		# Plot validation loss
		plt.plot(self.history.history['val_loss'], label='Validation Loss')

		plt.title(f'Learning Curves for {self.model_name} - {self.label_option}')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()

		# Plot for accuracy
		plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot = bottom subplot

		# Plot training accuracy
		plt.plot(self.history.history['accuracy'], label='Training Accuracy')

		# Plot validation accuracy
		plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')

		plt.title(f'Accuracies for {self.model_name} - {self.label_option}')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.legend()

		# Save the combined figure
		plt.tight_layout()
		plt.savefig(f'model_learning_curves/{self.model_name}_{self.label_option}_plots.jpg')
		plt.close()

