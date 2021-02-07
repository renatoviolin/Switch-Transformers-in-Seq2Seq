import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('results_1', 'rb') as fp:
    res_1 = pickle.load(fp)

with open('results_2', 'rb') as fp:
    res_2 = pickle.load(fp)

# ======== PLOT RESULTS =======================
x = np.arange(len(res_1))
plt.title(f'Seq2Seq Transformer')
plt.plot(x[150::50], res_1[150::50], label='Standard Transformer')
plt.plot(x[150::50], res_2[150::50], label='Switch Transformer: 16 FFN Layers')
plt.legend()
plt.xlabel('train step')
plt.ylabel('train loss')
plt.savefig('loss.jpg')
