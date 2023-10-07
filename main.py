import pandas as pd
import logging
from train_model import train_w2v
from clean_data import preprocess
from datetime import datetime

#logging configs
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.DEBUG)
logging.error("Hi")
logging.info('is when this event was logged.')
logging.debug("hshshsh")



#read data
df = pd.read_csv("./test.csv")

#pre-process data
#w2v_df is a type 'Series'
start_time = datetime.now()
logging.info("Pre-processing started")
w2v_df = preprocess(df,"Description")
logging.info(f"Pre-processing finished in time: {datetime.now() - start_time} ")

start_time = datetime.now()
logging.info("Model training started")
w2v_model = train_w2v(w2v_df)
logging.info(f"Model training finished in time: {datetime.now() - start_time} ")

#to print length and values of vector embedding
# print(len(w2v_model.wv))
# print(w2v_model.wv["great"])