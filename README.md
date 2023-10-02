
## Memoria

### Busqueda de los datos

Mayo 15, 2023 - Max:
    Se arma la funcion de Web Scrapping para tener todos los activos presentes y pasados del SP500.

    Se prueba la API gratuita de Alpaca.
    Si bien cuenta con los datos de activos deslistados, no cuenta con los datos
    que han salido del SP500 antes del 2015 (fecha de inicio de los datos de la API).
    Nuestra lista actualmente es los tickers desde 1957.
    Los datos no estan ajustados por SPLIT.

Mayo 17, 2023 - Max:
    Se prueba bajar los datos de Yahoo Finance.
    De la lista de 810 activos, solo consigo descargar 677, y no encuentro datos para
    133 activos.
    De momento no identifico porque no se pueden bajar los datos de estos activos.