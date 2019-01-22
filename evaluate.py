import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import dill as pickle
import itertools

path = '../model/dl_sa/'
filename = f'{path}/model_Thu_Jan_17_12_42_20_2019_fitting_history.pkl'
histories = pickle.load(open(filename, 'rb'))
history = histories[-1]

def plot_accnloss(history):
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

plot_accnloss(history)

filename = f'{file_dir}/model_Thu_Jan_17_12:42:20_2019.hdf5'
model = load_model(filename, custom_objects={'AttentionWithContext': att})

x_test = pickle.load(open(f'{file_dir}/x_test.pkl', 'rb'))
y_test = pickle.load(open(f'{file_dir}/y_test.pkl', 'rb'))

get_val = lambda arr: list(map(lambda x: x[1], arr))

y_true = y_test
y_true = get_val(y_true)
y_predict = model.predict_classes(x_test)

# print(metrics.classification_report(y_true, y_predict, target_names = ['negative', 'positive']))

def plot_confusion_matrix(cm, classes,
	normalize=False,
	title='Confusion matrix',
	cmap=plt.cm.Blues):
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	tick_marks = [0, 1]
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	plt.grid(False)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	plt.show()

plot_confusion_matrix(confusion_matrix(y_true, y_predict, labels=[1,0]), 
					  classes=['positive', 'negative'], 
					  normalize=True,
					  title='Confusion matrix')
