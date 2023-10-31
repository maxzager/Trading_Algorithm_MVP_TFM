from lib import *
import os
import importlib
import tensorflow as tf

def run_pipeline(data_path:str, bar_sizes:list, horizons:list, betas:list, periods:list, direction:str, epochs:int, batch_size:int, patience:int):
    
    # Load data and create candlesticks
    for bar_size in bar_sizes:

        #load data and create candlesticks
        print(f'Generating {bar_size} volume bars...')
        dh = DataHandler(data_path, 
                        tick_data=False)

        vol_bars = dh.generate_volumebars(bar_size)

        # Create Features
        print('Creating features...')
        fg = FeaturesGenerator(vol_bars,
                            momentum=True, 
                            volume=False, 
                            volatility=False)
        features = fg.df
        
        # Create Labels
        print('Creating labels...')
        lb = Labeler(vol_bars,
                    horizons=horizons,
                    betas=betas,
                    periods=periods,
                    direction=direction)
        labels = lb.df

        # Pre process data
        print('Preprocessing data...')
        dpp = DataPreprocessor(vol_bars, 
                        features, 
                        labels, 
                        validation_size=0.2, 
                        test_size=0.3)

        X_train, X_val, X_test, y_train, y_val, y_test = dpp.preprocess()
        input_shape = X_train.shape[1:] # defining input shape for models

        # Training bucle
        model_config_files = [f[:-3] for f in os.listdir('model_configs') if f.endswith('.py') and f != '__init__.py']
        

        for config_file in model_config_files:
            
            i=0
            config_module = importlib.import_module(f'model_configs.{config_file}')
            model_name = config_module.CONFIG['model_name']
            model_type = config_module.CONFIG['model_type']
            model = config_module.build_model

            for i in range(y_train.shape[1]):
                print(f'Training {model_name} for {y_train.columns[i]}, iteration {i}...')
                label_option = y_train.columns[i]

                mh = ModelsHandler(model_name=model_name, 
                                model_type=model_type,
                                direction=direction,
                                label_option=label_option)
                
                mh.train_model(build_model_func=model,
                            X_train=X_train,
                            y_train=y_train.iloc[:,i],
                            X_val=X_val,
                            y_val=y_val.iloc[:,i],
                            epochs=epochs,
                            batch_size=batch_size,
                            patience=patience,
                            input_shape=input_shape
                            )

                mh.plot_learning_curves()

                # store metrics
                mh.store_metrics(bar_size, model_name, model_type, label_option)
                

                i+=1

        