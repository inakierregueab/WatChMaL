import numpy as np
from analysis.plot_utils import *

base_path = './../outputs/2022-03-17/15-35-46/outputs/'

disp_learn_hist(base_path, title=None, losslim=None, axis=None, show=True)

labels = np.load(base_path + '/labels.npy')
predictions = np.load(base_path + '/predictions.npy')
class_names = ('gamma', 'e')
plot_confusion_matrix(labels, predictions, class_names)
# TODO: confusion to two classes

softmax_out = np.load(base_path + '/softmax.npy')
labels_dict = {'gamma': 0, 'e': 1}
fpr, tpr, thr = compute_roc(softmax_out, labels, labels_dict['e'], labels_dict['gamma'])
plot_roc(fpr, tpr, thr, "e", "gamma", fig_list=None, xlims=None, ylims=None, axes=None, linestyle=None, linecolor=None, plot_label='plot', show=False)
