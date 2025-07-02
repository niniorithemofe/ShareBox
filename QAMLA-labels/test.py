import torch
import torch.nn as nn
import model
import dataloader
import tqdm
import os
import shutil
import time
import pandas as pd
import numpy as np
import pickle
import argparse

def main(model): #make dynamic here 
	t1 = time.time()

	in_features = 300
	hidden_size = 256
	layer_num = 2

	print("\n")
	print(" Loading test Data ... ")
	print("="*30)
	print("\n")
	test_items = ["Dark Chocolate Bar w/ Caramelized Orange Rinds - L'exclusif - 80 g"]
	test_dl, tst = dataloader.test_loader(test_items)


	print(" Got test_dataloader ... ")
	print("="*30)
	print("\n")

	print(" Loading LSTM Model ...")
	print("="*30)
	print("\n")
	model = model.Rnn_Lstm(in_features, hidden_size, layer_num, 391, phase='Test')

	print(" Loading Weights on the Model ...")
	print("="*30)
	print("\n")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model.to(device)

	state_dict = torch.load('QAMLA\Multi-Label_Text_classification_for_Household_grocery_Items-master\Model_Checkpoints\checkpoint_5.pth')
	model.load_state_dict(state_dict)




	model.eval()

	print(" Predicting on the test data ...")
	print("="*30)
	print("\n")

	predictions = []
	
	for x, _ in tqdm.tqdm(test_dl):
		x = x.to(device)
		preds = model(x)
		predictions.extend(preds.cpu().detach().numpy())

	with open("QAMLA\Multi-Label_Text_classification_for_Household_grocery_Items-master\Data\labels.txt", "rb") as fp:
		labels = pickle.load(fp)
		
	test_df = pd.DataFrame(test_items, columns=["titles"])
	
	result_df = pd.DataFrame(data=predictions, columns=labels)
	test_results = pd.concat([test_df, result_df], axis=1)
	print(f"test result: {test_results} ")


	print("\n Saving Results to test_results.csv .")
	

	print("Completed \n")


if __name__=='__main__':


	main(model)

	    
