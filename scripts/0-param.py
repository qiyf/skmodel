import pickle

numSpin = 20
beta = 2.0

with open('../param/params.pkl','wb') as f:
	pickle.dump({'numSpin':numSpin, 'beta':beta},f)
