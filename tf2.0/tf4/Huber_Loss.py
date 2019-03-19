import matplotlib.pyplot as plt
import numpy as np

def sm_mae(true, pred, delta):
    """
    true: array of true values    
    pred: array of predicted values
    
    returns: smoothed mean absolute error loss
    """
    loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
    return np.sum(loss)

fig, ax1 = plt.subplots(1,1, figsize = (7,5))

target = np.repeat(0, 1000)
pred = np.arange(-10,10, 0.02)

delta = [0.1, 1, 10]

losses_huber = [[sm_mae(target[i], pred[i], q) for i in range(len(pred))] for q in delta]

# plot
for i in range(len(delta)):
    ax1.plot(pred, losses_huber[i], label = delta[i])
ax1.set_xlabel('Predictions')
ax1.set_ylabel('Loss')
ax1.set_title("Huber Loss/ Smooth MAE Loss vs. Predicted values (Color: Deltas)")
ax1.legend()
ax1.set_ylim(bottom=-1, top = 15)

fig.tight_layout()
plt.show()
