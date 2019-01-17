import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import dill as pickle

filename = '../model/dl_sa/model_Wed_Jan_16_23_32_48_2019_fitting_history.pkl'
histories = pickle.load(open(filename, 'rb'))
history = histories[-1]

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()