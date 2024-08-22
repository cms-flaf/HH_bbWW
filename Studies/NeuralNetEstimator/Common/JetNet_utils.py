import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


@tf.function
def MXLossFunc(target, output):
    H_WW_px = target[:, 0]
    H_WW_py = target[:, 1]
    H_WW_pz = target[:, 2]
    H_WW_E = target[:, 3]    
        
    X_mass = target[:, 4]
    
    H_bb_px = output[:, 0]
    H_bb_py = output[:, 1]
    H_bb_pz = output[:, 2]
    H_mass = 125.0
    H_bb_E_sqr = H_mass*H_mass + H_bb_px*H_bb_px + H_bb_py*H_bb_py + H_bb_pz*H_bb_pz
    H_bb_E = tf.sqrt(H_bb_E_sqr)
    
    X_mass_pred = tf.sqrt((H_bb_E + H_WW_E)**2 - (H_bb_px + H_WW_px)**2 - (H_bb_py + H_WW_py)**2 - (H_bb_pz + H_WW_pz)**2)
    return (X_mass_pred - X_mass)**2


@tf.function
def H_WW_MXLossFunc(target, output):
    H_bb_px = target[:, 0]
    H_bb_py = target[:, 1]
    H_bb_pz = target[:, 2]
    H_bb_E = target[:, 3]    
        
    X_mass = target[:, 4]
    
    H_WW_px = output[:, 0]
    H_WW_py = output[:, 1]
    H_WW_pz = output[:, 2]
    H_mass = 125.0
    H_WW_E_sqr = H_mass*H_mass + H_WW_px*H_WW_px + H_WW_py*H_WW_py + H_WW_pz*H_WW_pz
    H_WW_E = tf.sqrt(H_WW_E_sqr)
    
    X_mass_pred = tf.sqrt((H_bb_E + H_WW_E)**2 - (H_bb_px + H_WW_px)**2 - (H_bb_py + H_WW_py)**2 - (H_bb_pz + H_WW_pz)**2)
    return (X_mass_pred - X_mass)**2


def PlotLoss(history):
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.png')
    plt.clf()


def PlotPrediction(pred_labels, test_labels):
    # for predicting H->bb take H->VV from test labels
    # for predicting H->VV take H->bb from test labels

    H_VV_px = test_labels['H_VV_px']
    H_VV_py = test_labels['H_VV_py']
    H_VV_pz = test_labels['H_VV_pz']
    H_VV_E = test_labels['H_VV_E']

    # H_WW_px = pred_labels[:, 0]
    # H_WW_py = pred_labels[:, 1]
    # H_WW_pz = pred_labels[:, 2]
    # H_mass = 125.0
    # H_WW_E_sqr = H_mass*H_mass + H_WW_px*H_WW_px + H_WW_py*H_WW_py + H_WW_pz*H_WW_pz
    # H_WW_E = np.sqrt(H_WW_E_sqr)

    # H_bb_px = test_labels['H_bb_px']
    # H_bb_py = test_labels['H_bb_py']
    # H_bb_pz = test_labels['H_bb_pz']
    # H_bb_E = test_labels['H_bb_E']

    # H_bb_px_true = test_labels['genbjet1_px'] + test_labels['genbjet2_px']
    # H_bb_py_true = test_labels['genbjet1_py'] + test_labels['genbjet2_py']
    # H_bb_pz_true = test_labels['genbjet1_pz'] + test_labels['genbjet2_pz']
    # H_bb_E_true = test_labels['genbjet1_E'] + test_labels['genbjet2_E']

    H_bb_px = pred_labels[:, 0]
    H_bb_py = pred_labels[:, 1]
    H_bb_pz = pred_labels[:, 2]
    H_mass = 125.0
    H_bb_E_sqr = H_mass*H_mass + H_bb_px*H_bb_px + H_bb_py*H_bb_py + H_bb_pz*H_bb_pz
    H_bb_E = np.sqrt(H_bb_E_sqr)

    X_mass_pred = np.sqrt((H_bb_E + H_VV_E)**2 - (H_bb_px + H_VV_px)**2 - (H_bb_py + H_VV_py)**2 - (H_bb_pz + H_VV_pz)**2)

    width = PredWidth(X_mass_pred)
    peak = PredPeak(X_mass_pred)

    plt.hist(X_mass_pred, bins=100)
    plt.title('JetNet prediction')
    plt.xlabel('X mass [GeV]')
    plt.ylabel('Count')
    plt.figtext(0.75, 0.8, f"peak: {peak:.2f}")
    plt.figtext(0.75, 0.75, f"width: {width:.2f}")
    plt.grid(True)
    plt.savefig('JetNet_prediction.png', bbox_inches='tight')
    plt.clf()

    return X_mass_pred


def PredWidth(pred_mass):
    q_84 = np.quantile(pred_mass, 0.84)
    q_16 = np.quantile(pred_mass, 0.16)
    width = q_84 - q_16
    return width 


def PredPeak(pred_mass):
    counts = np.bincount(pred_mass)
    peak = np.argmax(counts)
    return peak 