import glob
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import *
from functools import reduce


def basic_metrics(base_path, labels, mode):
    pred = np.load(base_path + mode + '/predictions.npy')
    prob = np.load(base_path + mode + '/softmax.npy')

    fpr, tpr, thresholds = roc_curve(labels, prob[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    metrics = {
        'model': mode,
        'acc': accuracy_score(labels, pred),
        'f1': f1_score(labels, pred),
        'log_loss': log_loss(labels, prob),
        'auc': roc_auc,
        'opt_threshold': optimal_threshold
    }
    return metrics


def draw_roc_curve(base_path, ground_truth, modes, pos_label=1):
    plt.figure(figsize=(10, 7))
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')

    for mode in modes:
        predictions = np.load(base_path + mode + '/softmax.npy')[:, 1]

        fpr, tpr, thresholds = roc_curve(ground_truth, predictions, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=1, label=f'{mode}: AUC = %0.2f' % roc_auc)

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate', fontsize='large')
    plt.ylabel('True Positive Rate', fontsize='large')
    plt.title('Receiver Operating Characteristic', fontsize='x-large')
    plt.legend(loc="lower right", fontsize='large')
    plt.show()


def disp_learn_hist(location, title=None, losslims=None, axis=None, show=True):
    """
    Purpose : Plot the loss and accuracy history for a training session

    Args:
        location    ... output directory containing log files
        title       ... the title for the plot
        losslims    ... sets bound on y axis of loss
        axis        ... axis to plot on
        show        ... if true then display figure, otherwise return figure
    """
    val_log = location + '/log_val.csv'
    val_log_df = pd.read_csv(val_log)

    train_log_df = get_aggregated_train_data(location)

    if axis is None:
        fig, ax1 = plt.subplots(figsize=(12, 8), facecolor='w')
    else:
        ax1 = axis

    line11 = ax1.plot(train_log_df.epoch, train_log_df.loss, linewidth=2, label='Train loss', color='b', alpha=0.3)
    line12 = ax1.plot(val_log_df.epoch, val_log_df.loss, marker='o', markersize=3, linestyle='',
                      label='Validation loss', color='blue')

    if losslims is not None:
        ax1.set_ylim(losslims[0], losslims[1])

    ax2 = ax1.twinx()
    line21 = ax2.plot(train_log_df.epoch, train_log_df.accuracy, linewidth=2, label='Train accuracy', color='r',
                      alpha=0.3)
    line22 = ax2.plot(val_log_df.epoch, val_log_df.accuracy, marker='o', markersize=3, linestyle='',
                      label='Validation accuracy', color='red')

    ax1.set_xlabel('Epoch', fontweight='bold', fontsize=24, color='black')
    ax1.tick_params('x', colors='black', labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold', color='b')
    ax1.tick_params('y', colors='b', labelsize=18)

    ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold', color='r')
    ax2.tick_params('y', colors='r', labelsize=18)
    ax2.set_ylim(0., 1.05)
    ax2.set_yticks(np.arange(0, 1.1, 0.1))

    # added these four lines
    lines = line11 + line12 + line21 + line22
    labels = [l.get_label() for l in lines]
    leg = ax2.legend(lines, labels, fontsize=14, loc=3, numpoints=1)
    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')

    if title is not None:
        ax1.set_title(title, fontsize=20)

    if show:
        plt.grid()
        plt.show()
        return

    if axis is None:
        return fig


def get_aggregated_train_data(location):
    """
    Aggregate training logs from all processes into a single set of data

    Args:
        location    ... path to outputs directory containing training logs

    Returns: pandas dataframe containing aggregated data
    """
    # get all training data files
    base_log_path = location + '/log_train_[0-9]*.csv'
    log_paths = glob.glob(base_log_path)

    print("Found training logs: ", log_paths)

    log_dfs = []
    for log_path in log_paths:
        log_dfs.append(pd.read_csv(log_path))
        log_dfs.append(pd.read_csv(log_path))

    # combine all files into one dataframe
    train_log_df = pd.DataFrame(0, index=np.arange(len(log_dfs[0])), columns=log_dfs[0].columns)
    for idx, df_vals in enumerate(zip(*[log_df.values for log_df in log_dfs])):
        iteration = df_vals[0][0]
        epoch = df_vals[0][1]
        loss = sum([df_val[2] for df_val in df_vals]) / len(df_vals)
        accuracy = sum([df_val[3] for df_val in df_vals]) / len(df_vals)

        output_df_vals = (iteration, epoch, loss, accuracy)
        train_log_df.iloc[idx] = output_df_vals

    return train_log_df


def data_splitting(split_idxs):
    splitting = {}
    for file in split_idxs.files:
        splitting[file] = split_idxs[file]
    n_test = len(splitting['test_idxs'])
    n_val = len(splitting['val_idxs'])
    n_train = len(splitting['train_idxs'])
    n_total = n_train + n_val + n_test
    print('Total num. of events: ', n_total)
    print('Events for testing: %1.3f' % (n_test / n_total * 100))
    print('Events for validation: %1.3f' % (n_val / n_total * 100))
    print('Events for training: %1.3f' % (n_train / n_total * 100))


def plot_classifier_response(softmaxes, labels, particle_names, label_dict,
                             bins=None, linestyles=None, legend_locs=None,
                             extra_panes=[], xlim=None, label_size=14, legend_label_dict=None, show=True):
    '''
    Plot classifier likelihoods over different classes for events of a given particle type

    Args:
        softmaxes           ... 2d array with first dimension n_samples
        labels              ... 1d array of particle labels to use in every output plot, or list of 4 lists of particle names to use in each respectively
        particle_names      ... list of string names of particle types to plot. All must be keys in 'label_dict'
        label_dict          ... dictionary of particle labels, with string particle name keys and values corresponsing to values taken by 'labels'
        bins                ... optional, number of bins for histogram
        legend_locs         ... list of 4 strings for positioning the legends
        extra_panes         ... list of lists of particle names, each of which contains the names of particles to use in a joint response plot
        xlim                ... limit the x-axis
        label_size          ... font size
        legend_label_dict   ... dictionary of display symbols for each string label, to use for displaying pretty characters
        show                ... if true then display figure, otherwise return figure
    author: Calum Macdonald
    June 2020
    '''
    if legend_label_dict is None:
        legend_label_dict = {}
        for name in particle_names:
            legend_label_dict[name] = name

    legend_size = label_size

    num_panes = softmaxes.shape[1] + len(extra_panes)

    fig, axes = plt.subplots(1, num_panes, figsize=(5 * num_panes, 5), facecolor='w')
    inverse_label_dict = {value: key for key, value in label_dict.items()}

    label_dict = {k: v for k, v in sorted(label_dict.items(), key=lambda item: item[1])}
    softmaxes_list = separate_particles([softmaxes], labels, label_dict, [name for name in label_dict.keys()])[0]

    if isinstance(particle_names[0], str):
        particle_names = [particle_names for _ in range(num_panes)]

    # generate single particle plots
    for independent_particle_label, ax in enumerate(axes[:softmaxes.shape[1]]):
        print(label_dict)
        dependent_particle_labels = [label_dict[particle_name] for particle_name in
                                     particle_names[independent_particle_label]]
        for dependent_particle_label in dependent_particle_labels:
            ax.hist(softmaxes_list[dependent_particle_label][:, independent_particle_label],
                    label=f"{legend_label_dict[inverse_label_dict[dependent_particle_label]]} Events",
                    alpha=0.7, histtype=u'step', bins=bins, density=True,
                    linestyle=linestyles[dependent_particle_label] if linestyles is not None else 'solid', linewidth=2)
        ax.legend(loc=legend_locs[independent_particle_label] if legend_locs is not None else 'best',
                  fontsize=legend_size)
        ax.set_xlabel('P({})'.format(legend_label_dict[inverse_label_dict[independent_particle_label]]),
                      fontsize=label_size)
        ax.set_ylabel('Normalized Density', fontsize=label_size)
        ax.set_yscale('log')

    ax = axes[-1]

    # generate joint plots
    for n, extra_pane_particle_names in enumerate(extra_panes):
        pane_idx = softmaxes.shape[1] + n
        ax = axes[pane_idx]
        dependent_particle_labels = [label_dict[particle_name] for particle_name in particle_names[pane_idx]]
        for dependent_particle_label in dependent_particle_labels:
            ax.hist(reduce(lambda x, y: x + y,
                           [softmaxes_list[dependent_particle_label][:, label_dict[pname]] for pname in
                            extra_pane_particle_names]),
                    label=legend_label_dict[particle_names[-1][dependent_particle_label]],
                    alpha=0.7, histtype=u'step', bins=bins, density=True,
                    linestyle=linestyles[dependent_particle_label] if linestyles is not None else 'solid', linewidth=2)
        ax.legend(loc=legend_locs[-1] if legend_locs is not None else 'best', fontsize=legend_size)
        xlabel = ''
        for list_index, independent_particle_name in enumerate(extra_pane_particle_names):
            xlabel += 'P({})'.format(legend_label_dict[independent_particle_name])
            if list_index < len(extra_pane_particle_names) - 1:
                xlabel += ' + '
        ax.set_xlabel(xlabel, fontsize=label_size)
        ax.set_ylabel('Normalized Density', fontsize=label_size)
        ax.set_yscale('log')

    plt.tight_layout()

    if show:
        plt.show()
        return

    return fig


def separate_particles(input_array_list, labels, index_dict, desired_labels=['gamma', 'e', 'mu']):
    '''
    Separates all arrays in a list by indices where 'labels' takes a certain value, corresponding to a particle type.

    Args:
        input_array_list    ... list of arrays to be separated, must have same length and same length as 'labels'
        labels              ... list of labels, taking any of the three values in index_dict.values()
        index_dict          ... dictionary of particle labels, must have 'gamma','mu','e' keys pointing to values taken by 'labels',
                                        unless desired_labels is passed
        desired_labels      ... optional list specifying which labels are desired and in what order. Default is ['gamma','e','mu']

    Returns: a list of tuples, each tuple contains section of each array corresponsing to a desired label
    author: Calum Macdonald
    June 2020
    '''
    idxs_list = [np.where(labels == index_dict[label])[0] for label in desired_labels]

    separated_arrays = []
    for array in input_array_list:
        separated_arrays.append(tuple([array[idxs] for idxs in idxs_list]))

    return separated_arrays


def compute_roc(softmax_out_val, labels_val, true_label, false_label):
    """
    Compute ROC metrics from softmax and labels for given particle labels

    Args:
        softmax_out_val     ... array of softmax outputs
        labels_val          ... 1D array of actual labels
        true_label          ... label of class to be used as true binary label
        false_label         ... label of class to be used as false binary label

    Returns:
        fpr, tpr, thr       ... false positive rate, true positive rate, thresholds used to compute scores
    """
    labels_val_for_comp = labels_val[np.where((labels_val == false_label) | (labels_val == true_label))]
    softmax_out_for_comp = softmax_out_val[np.where((labels_val == false_label) | (labels_val == true_label))][:,
                           true_label]

    fpr, tpr, thr = roc_curve(labels_val_for_comp, softmax_out_for_comp, pos_label=true_label)

    return fpr, tpr, thr


def plot_roc(fpr, tpr, thr, true_label_name, false_label_name, fig_list=None, xlims=None, ylims=None, axes=None,
             linestyle=None, linecolor=None, plot_label=None, show=False):
    """
    Plot ROC curves for a classifier that has been evaluated on a validation set with respect to given labels

    Args:
        fpr, tpr, thr           ... false positive rate, true positive rate, thresholds used to compute scores
        true_label_name         ... name of class to be used as true binary label
        false_label_name        ... name of class to be used as false binary label
        fig_list                ... list of indexes of ROC curves to plot
        xlims                   ... xlims to apply to plots
        ylims                   ... ylims to apply to plots
        axes                    ... axes to plot on
        linestyle, linecolor    ... line style and color
        plot_label              ... string to use in title of plots
        show                    ... if true then display figure, otherwise return figure
    """
    # Compute additional parameters
    rejection = 1.0 / (fpr + 1e-10)
    roc_AUC = auc(fpr, tpr)

    if fig_list is None:
        fig_list = list(range(3))

    figs = []
    # Plot results
    if axes is None:
        if 0 in fig_list:
            fig0, ax0 = plt.subplots(figsize=(12, 8), facecolor="w")
            figs.append(fig0)
        if 1 in fig_list:
            fig1, ax1 = plt.subplots(figsize=(12, 8), facecolor="w")
            figs.append(fig1)
        if 2 in fig_list:
            fig2, ax2 = plt.subplots(figsize=(12, 8), facecolor="w")
            figs.append(fig2)
    else:
        print(axes)
        axes_iter = iter(axes)
        if 0 in fig_list:
            ax0 = next(axes_iter)
        if 1 in fig_list:
            ax1 = next(axes_iter)
        if 2 in fig_list:
            ax2 = next(axes_iter)

    if xlims is not None:
        xlim_iter = iter(xlims)
    if ylims is not None:
        ylim_iter = iter(ylims)

    if 0 in fig_list:
        ax0.tick_params(axis="both", labelsize=20)
        ax0.plot(fpr, tpr,
                 label=plot_label if plot_label + ', AUC={:.3f}'.format(
                     roc_AUC) is not None else r'{} VS {} ROC, AUC={:.3f}'.format(true_label_name, false_label_name,
                                                                                  roc_AUC),
                 linestyle=linestyle if linestyle is not None else None,
                 color=linecolor if linecolor is not None else None)
        ax0.set_xlabel('FPR', fontsize=20)
        ax0.set_ylabel('TPR', fontsize=20)
        ax0.legend(loc="lower right", prop={'size': 16})

        if xlims is not None:
            xlim = next(xlim_iter)
            ax0.set_xlim(xlim[0], xlim[1])
        if ylims is not None:
            ylim = next(ylim_iter)
            ax0.set_ylim(ylim[0], ylim[1])

    if 1 in fig_list:
        ax1.tick_params(axis="both", labelsize=20)
        ax1.set_yscale('log')
        ax1.grid(b=True, which='major', color='gray', linestyle='-')
        ax1.grid(b=True, which='minor', color='gray', linestyle='--')
        ax1.plot(tpr, rejection,
                 label=plot_label + ', AUC={:.3f}'.format(
                     roc_AUC) if plot_label is not None else r'{} VS {} ROC, AUC={:.3f}'.format(true_label_name,
                                                                                                false_label_name,
                                                                                                roc_AUC),
                 linestyle=linestyle if linestyle is not None else None,
                 color=linecolor if linecolor is not None else None)

        xlabel = f'{true_label_name} Signal Efficiency'
        ylabel = f'{false_label_name} Background Rejection'
        title = '{} vs {} Rejection'.format(true_label_name, false_label_name)

        ax1.set_xlabel(xlabel, fontsize=20)
        ax1.set_ylabel(ylabel, fontsize=20)
        ax1.set_title(title, fontsize=24)
        ax1.legend(loc="upper right", prop={
            'size': 16})  # bbox_to_anchor=(1.05, 1), loc='upper left') #loc="upper right",prop={'size': 16})

        if xlims is not None:
            xlim = next(xlim_iter)
            ax1.set_xlim(xlim[0], xlim[1])
        if ylims is not None:
            ylim = next(ylim_iter)
            ax1.set_ylim(ylim[0], ylim[1])

    if 2 in fig_list:
        ax2.tick_params(axis="both", labelsize=20)
        # plt.yscale('log')
        # plt.ylim(1.0,1)
        ax2.grid(b=True, which='major', color='gray', linestyle='-')
        ax2.grid(b=True, which='minor', color='gray', linestyle='--')
        ax2.plot(tpr, tpr / np.sqrt(fpr),
                 label=plot_label + ', AUC={:.3f}'.format(
                     roc_AUC) if plot_label is not None else r'{} VS {} ROC, AUC={:.3f}'.format(true_label_name,
                                                                                                false_label_name,
                                                                                                roc_AUC),
                 linestyle=linestyle if linestyle is not None else None,
                 color=linecolor if linecolor is not None else None)
        ax2.set_xlabel('efficiency', fontsize=20)
        ax2.set_ylabel('~significance', fontsize=20)
        ax2.legend(loc="upper right", prop={'size': 16})

        if xlims is not None:
            xlim = next(xlim_iter)
            ax2.set_xlim(xlim[0], xlim[1])
        if ylims is not None:
            ylim = next(ylim_iter)
            ax2.set_ylim(ylim[0], ylim[1])

    if show:
        plt.show()
        return

    if axes is None:
        return tuple(figs)


def multi_compute_roc(softmax_out_val_list, labels_val_list, true_label, false_label):
    """
    Call compute_roc on multiple sets of data

    Args:
        softmax_out_val_list    ... list of arrays of softmax outputs
        labels_val_list         ... list of 1D arrays of actual labels
        true_label              ... label of class to be used as true binary label
        false_label             ... label of class to be used as false binary label

    """
    fprs, tprs, thrs = [], [], []
    for softmax_out_val, labels_val in zip(softmax_out_val_list, labels_val_list):
        fpr, tpr, thr = compute_roc(softmax_out_val, labels_val, true_label, false_label)
        fprs.append(fpr)
        tprs.append(tpr)
        thrs.append(thr)

    return fprs, tprs, thrs

def multi_plot_roc(fprs, tprs, thrs, true_label_name, false_label_name, fig_list=None, xlims=None, ylims=None, axes=None, linestyles=None, linecolors=None, plot_labels=None, show=False):
    '''
    Plot multiple ROC curves of background rejection vs signal efficiency. Can plot 'rejection' (1/fpr) or 'fraction' (tpr).

    Args:
        fprs, tprs, thrs        ... list of false positive rate, list of true positive rate, list of thresholds used to compute scores
        true_label_name         ... name of class to be used as true binary label
        false_label_name        ... name of class to be used as false binary label
        fig_list                ... list of indexes of ROC curves to plot
        xlims                   ... xlims to apply to plots
        ylims                   ... ylims to apply to plots
        axes                    ... axes to plot on
        linestyle, linecolor    ... lists of line styles and colors
        plot_labels             ... list of strings to use in title of plots
        show                    ... if true then display figures, otherwise return figures
    '''
    rejections = [1.0/(fpr+1e-10) for fpr in fprs]
    AUCs = [auc(fpr,tpr) for fpr, tpr in zip(fprs, tprs)]

    num_panes = len(fig_list)
    fig, axes = plt.subplots(num_panes, 1, figsize=(12,8*num_panes))
    if num_panes > 1:
        fig.suptitle("ROC for {} vs {}".format(true_label_name, false_label_name), fontweight='bold',fontsize=32)

    # Needed for 1 plot case
    axes = np.array(axes).reshape(-1)

    for idx, fpr, tpr, thr in zip(range(len(fprs)), fprs, tprs, thrs):
        figs = plot_roc(fpr, tpr, thr,
        true_label_name, false_label_name,
        axes=axes, fig_list=fig_list, xlims=xlims, ylims=ylims,
        linestyle=linestyles[idx]  if linestyles is not None else None,
        linecolor=linecolors[idx] if linecolors is not None else None,
        plot_label=plot_labels[idx] if plot_labels is not None else None,
        show=False)

    return figs

# TODO: missing comparison utils
