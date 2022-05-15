import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
import matplotlib.pyplot as plt


def test_and_plot(models,
                  test_data_1, test_data_10, test_data_100,
                  combine_method=None,
                  output_activation="sigmoid",
                  save_preds=0,
                  csv_name="preds",
                  dir_path="."):

    """

    This function test models, plots their precision-recall curves and saves predictions.

    @param models: list - tf.keras.Model instances
    @param test_data_1: pandas.DataFrame - test data ratio 1:1
    @param test_data_10: pandas.DataFrame - test data ratio 1:10
    @param test_data_100: pandas.DataFrame - test data ratio 1:100
    @param combine_method: string - defines if models' predictions are combined, one of 'majority', 'mean'
    @param output_activation: string - 'sigmoid', 'softmax'
    @param save_preds: int - 0 no predictions are save, 1 individual models' predictions are saved, 2 combined
    predictions are saved
    @param csv_name: string - name of created csv
    @param dir_path: string - path where csv is stored
    @return:
    """

    f, ax = plt.subplots(1, 3, figsize=(25, 5))
    f.suptitle("Precision-Recall curve", fontsize=16)
    _ = ax[0].set_title(f"1:100 ratio")
    _ = ax[1].set_title(f"1:10 ratio")
    _ = ax[2].set_title(f"1:1 ratio")

    labels = {
        "100": np.array(list(test_data_100[1])),
        "10": np.array(list(test_data_10[1])),
        "1": np.array(list(test_data_1[1]))
    }

    data = {
        "100": np.array(list(test_data_100[0])),
        "10": np.array(list(test_data_10[0])),
        "1": np.array(list(test_data_1[0]))
    }

    for ax_num, test_ratio in enumerate(["100", "10", "1"]):

        # Store predictions for current ratio
        preds = []

        for i, m in enumerate(models):

            pred = m.predict(data[test_ratio], batch_size=2048, verbose=False)

            if output_activation == "softmax":
                pred_label = np.argmax(pred, axis=1)
                pred_f = np.zeros(len(pred))
                pred_f[pred_label == 1] = pred[pred_label == 1][:, 1]
                pred_f[pred_label == 0] = 1 - pred[pred_label == 0][:, 0]
                pred = pred_f

            preds.append(pred)

            print(f"{m.name}\tAP{test_ratio} =", average_precision_score(labels[test_ratio], pred))

            _ = PrecisionRecallDisplay.from_predictions(
                labels[test_ratio],
                pred,
                ax=ax[ax_num],
                name=m.name,
                linestyle=':')

            if save_preds == 1:
                pd.DataFrame(data={"labels": labels[test_ratio], "preds": pred})\
                    .to_csv(f"{dir_path}/{csv_name}_{test_ratio}.csv", index=False)

        if combine_method is not None:

            if combine_method == "majority":
                ensemble_preds = np.sum(np.array(preds) > 0.5, axis=0)
            else:
                ensemble_preds = np.mean(preds, axis=0)

            _ = PrecisionRecallDisplay.from_predictions(
                labels[test_ratio],
                ensemble_preds,
                ax=ax[ax_num],
                name=f"Ensemble of {len(models)}",
                linestyle=':')

            if save_preds == 2:
                pd.DataFrame(data={"labels": labels[test_ratio], "preds": ensemble_preds})\
                    .to_csv(f"{dir_path}/{csv_name}_{test_ratio}.csv", index=False)
