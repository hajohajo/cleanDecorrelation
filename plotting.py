import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_curve, auc
import os
import shutil

sns.set()
sns.set_style("whitegrid")

sns.set()
sns.set_style("whitegrid")

SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from matplotlib import rcParams
from matplotlib import font_manager
font_manager._rebuild()
rcParams['font.serif'] = "Latin Modern Sans"
rcParams['font.family'] = "serif"


path_to_plots = "./"
def initialize_plotting():
    listOfFolders = [path_to_plots+"plots", path_to_plots+"plots/distortionPlots"]
    for dir in listOfFolders:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

#Production worthy
def classifierVsX(dataframe, variable, binningToUse, plotName):
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2, 1]}, figsize=(8, 6))

    fig.suptitle("Mean output with respect to variable of interest")
    variableName = variable
    binning = binningToUse
    binCenters = binning+(binning[1]-binning[0])/2.0
    colors = [sns.xkcd_rgb["cerulean"], sns.xkcd_rgb["rouge"]]
    labels = ["background", "signal"]
    signal = [0, 1]
    for type in signal:
        isSignal = type
        mva = dataframe.loc[(dataframe.isSignal == isSignal), "prediction"]
        variable = dataframe.loc[(dataframe.isSignal == isSignal), variableName]
        indices = np.digitize(variable, binning, right=True)
        indices[indices == 0] = 1
        indices = indices - 1

        binMean = -np.ones(len(binning))
        binStd = np.zeros(len(binning))

        for i in np.unique(indices):
            if(np.sum(indices == i)<2):
                continue
            mean, std = norm.fit(mva[indices == i])

            binMean[i] = mean
            binStd[i] = std

        up = np.add(binMean, binStd)
        down = np.add(binMean, -binStd)

        nonzeroPoints = (binMean > -1)
        if(isSignal==0 or isSignal==1):
            ax0.fill_between(binCenters[nonzeroPoints], up[nonzeroPoints], down[nonzeroPoints], alpha=0.6, color=colors[isSignal]) #label="$\pm 1\sigma$",
            ax0.plot(binCenters[nonzeroPoints], up[nonzeroPoints], c=colors[isSignal], linewidth=1.0, alpha=0.8)
            ax0.plot(binCenters[nonzeroPoints], down[nonzeroPoints], c=colors[isSignal], linewidth=1.0, alpha=0.8)
            ax0.scatter(binCenters[nonzeroPoints], binMean[nonzeroPoints], c=colors[isSignal], label=labels[isSignal], linewidths=1.0, edgecolors='k')
        else:
            ax0.fill_between(binCenters[nonzeroPoints], up[nonzeroPoints], down[nonzeroPoints],
                             alpha=0.6, color=colors[isSignal])
            ax0.plot(binCenters[nonzeroPoints], up[nonzeroPoints], c=colors[isSignal], linewidth=1.0, alpha=0.7)
            ax0.plot(binCenters[nonzeroPoints], down[nonzeroPoints], c=colors[isSignal], linewidth=1.0, alpha=0.7)
            ax0.scatter(binCenters[nonzeroPoints], binMean[nonzeroPoints], c=colors[isSignal],
                        linewidths=1.0, edgecolors='k')


    ax0.set_ylim(-0.05, 1.35)
    locs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax0.set_yticks(locs)  # Set locations and labels
    ax0.set_xlim(binCenters[0], binCenters[-1])
    ax0.set_ylabel("Mean DNN output")

    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles[::-1], labels[::-1], loc='upper right')

    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.grid(False, axis='x')

    sig = dataframe.loc[(dataframe.isSignal == 1), variableName]
    bkg = dataframe.loc[(dataframe.isSignal != 1), variableName]
    ax1.hist((bkg, sig), bins=binning, label=labels, color=colors, stacked=True, edgecolor="black", linewidth=1.0, alpha=0.7)

    ax1.set_xlabel("Variable of interest")
    ax1.set_ylabel("Number of events/bin")

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.grid(False, axis='x')

    plt.tight_layout(pad=2.2)
    fig.align_ylabels((ax0, ax1))
    plt.savefig(path_to_plots+"plots/"+plotName+".pdf")
    plt.clf()
    plt.close()


def createROCplot(dataframe, name):
    falsePositiveRate, truePositiveRate, thresholds = roc_curve(dataframe.loc[:, "isSignal"], dataframe.loc[:, "prediction"])

    _auc = auc(falsePositiveRate, truePositiveRate)

    plt.plot(truePositiveRate, 1-falsePositiveRate, label="ROC area = {:.3f}".format(_auc))
    plt.legend()
    plt.title("ROC curves")
    plt.ylabel("Fake rejection")
    plt.xlabel("True efficiency")
    plt.savefig(path_to_plots+"plots/"+name+"_ROC.pdf")
    plt.clf()
    plt.close()

def createMVADistribution(dataframe, name):
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("MVA distributions for signal and background")

    binning = np.linspace(0.0, 1.0, 41)

    colors = [sns.xkcd_rgb["rouge"], sns.xkcd_rgb["cerulean"]]
    isSignal = dataframe.loc[:, "isSignal"]
    plt.hist(dataframe.loc[(isSignal == 1), "prediction"], bins=binning, label="Signal", density=True, alpha=0.7, edgecolor="black", linewidth=1.0, color=colors[0])
    plt.hist(dataframe.loc[(isSignal == 0), "prediction"], bins=binning, label="Background", density=True, alpha=0.7, edgecolor="black", linewidth=1.0, color=colors[1])

    plt.legend()
    plt.xlabel("MVA output")
    plt.ylabel("Events")

    plt.savefig(path_to_plots+"plots/" + name + "_MVADistribution.pdf")
    plt.clf()
    plt.close()


def create2DMVAdistribution(dataframe, name):
    fig = plt.figure(figsize=(8, 6))
    # fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2, 1]}, figsize=(8, 6))
    fig.suptitle("MVA vs. m$_{T}$ for background")

    binning_x = np.linspace(0.0, 500.0, 26)
    binning_y = np.linspace(0.0, 1.0, 21)

    mt = dataframe.loc[dataframe.isSignal == 0, "TransverseMass"]
    mva = dataframe.loc[dataframe.isSignal == 0, "prediction"]

    content, x_, y_ = np.histogram2d(mt, mva, bins=[binning_x, binning_y])
    # H_norm_rows = (content.T / np.sum(content, axis=0)).T
    for column in range(content.shape[0]):
        content[column][:] = content[column][:]/np.sum(content[column][:])

    plt.imshow(content.T, origin="lower", interpolation="nearest", extent=[binning_x[0], binning_x[-1], 0, 1], aspect=binning_x[-1]) #, binning_y[0], binning_y[-1]])
    plt.ylabel("MVA output")
    plt.xlabel("m$_{T}$")
    plt.grid(b=None)
    plt.colorbar()

    plt.savefig(path_to_plots+"plots/" + name + "_MVAvsMt.pdf")
    plt.clf()