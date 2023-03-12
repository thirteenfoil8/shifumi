from shifumi.models.train import Train
from dotenv import dotenv_values
config = dotenv_values(".env")
train = Train()
train.model= train.load_model()
train.plot_history()