import pickle
import random
import blackjack_ml_classes_and_functions

training_bot = AI_Player()


for i in range(50000):
    
    training_bot.train_module()

with open("training_bot.pkl", "wb") as bot_file:

    pickle.dump(training_bot, bot_file)
    print("file check")

