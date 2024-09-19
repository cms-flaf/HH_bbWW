import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# out_* - output of the neural net
# target_* - true vaules of quantities that nn predicts
@tf.function
def GetMXPred(out_px, out_py, out_pz, target_px, target_py, target_pz, target_E, target_mass):
    H_mass = 125.0
    out_E_sqr = H_mass*H_mass + out_px*out_px + out_py*out_py + out_pz*out_pz
    out_E = tf.sqrt(out_E_sqr)
    
    pred_mass = tf.sqrt((out_E + target_E)**2 - (out_px + target_px)**2 - (out_y + target_py)**2 - (out_pz + target_pz)**2)
    return pred_mass


@tf.function
def MXLossFunc(target, output):
    X_mass = target[:, 4]
    X_mass_pred = GetMXPred(output[:, 0], output[:, 1], output[:, 2], 
                            target[:, 0], target[:, 1], target[:, 2], target[:, 3], target[:, 4])
    return (X_mass_pred - X_mass)**2


def PlotLoss(history):
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.pdf', bbox_inches='tight')
    plt.clf()


def PlotPrediction(label_df, predicted_df):
    # label_df contains p4 of H->WW
    # predicted_df contains p4 of H->bb 

    masspoints = label_df['X_mass'].unique()
    X_mass_true = np.array(label_df['X_mass'])
    X_mass_pred = np.sqrt((label_df['H_VV_E'] + predicted_df['H_bb_E'])**2 -
                          (label_df['H_VV_px'] + predicted_df['H_bb_px'])**2 -
                          (label_df['H_VV_py'] + predicted_df['H_bb_py'])**2 -
                          (label_df['H_VV_pz'] + predicted_df['H_bb_pz'])**2)
    mass_df = pd.DataFrame({"X_mass_true": X_mass_true, "X_mass_pred": X_mass_pred})

    for mp in masspoints:
        df = mass_df[mass_df['X_mass_true'] == mp]

        width = PredWidth(df['X_mass_pred'])
        peak = PredPeak(df['X_mass_pred'])

        plt.hist(df['X_mass_pred'], bins=100)
        plt.title('JetNet prediction')
        plt.xlabel('X mass [GeV]')
        plt.ylabel('Count')
        plt.figtext(0.75, 0.8, f"peak: {peak:.2f}")
        plt.figtext(0.75, 0.75, f"width: {width:.2f}")
        plt.grid(True)
        plt.savefig(f"X_mass_pred_{mp}_GeV.pdf", bbox_inches='tight')
        plt.clf()


def PredWidth(pred_mass):
    q_84 = np.quantile(pred_mass, 0.84)
    q_16 = np.quantile(pred_mass, 0.16)
    width = q_84 - q_16
    return width 


def PredPeak(pred_mass):
    counts = np.bincount(pred_mass)
    peak = np.argmax(counts)
    return peak 