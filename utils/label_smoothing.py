from tqdm.notebook import tqdm
import numpy as np


def get_instances_hardness(model, df, data_col=0, label_col=1, batch_size=150000):

    """
    This function takes a model, gets prediction for the whole training pool (all training data) and returns
    instance hardness.

    @param model: tf.keras.Model instance
    @param df: pd.DataFrame with encoded samples
    @param data_col: string / int - label of column with data
    @param label_col: string / int - label of column with labels
    @param batch_size: int - batch size used for splitting df
    @return: list - instance hardness for each sample in df
    """

    num_batches = (len(df) // batch_size) + 1
    preds = []

    for i in tqdm(list(range(num_batches))):
        preds.append(np.squeeze(model.predict(np.array(list(df[i*batch_size:(i+1)*batch_size][data_col])),
                                              batch_size=128*8,
                                              verbose=1)))

    preds = np.concatenate(preds)

    # For negative samples, compute their probability
    preds[df[label_col] == 0] = 1 - preds[df[label_col] == 0]

    # Return instance hardness as 1 - model's prediction
    return 1 - preds


def get_soft_labels(df, borders, min_major=0.10, increase=0.05, ih_col="ih", label_col=1):

    """
    Sets soft labels for samples in df. Changing the 'borders' attribute results in more precise or less precise
    label smoothing. When values in the 'df[ih_col]' were produced by a bigger model for instance, we recommend using
    borders = np.arange(0.0, 1.1, 0.1); if IH was produced by a small scouts borders = np.arange(0.0, 0.6, 0.1) are
    recommended. Borders can also be estimated by np.hist .

    @param df: pandas.DataFrame - (training) dataset
    @param borders: list - points defining the increase of the smoothing factor
    @param min_major: float - minimal label smoothing factor
    @param increase: float - increase of a label smoothing factor
    @param ih_col: string - column with computed IH
    @param label_col: string / int - column with label
    @return: pandas.DataFrame (df) with smooth labels
    """

    # Set default label for the minority class
    df.loc[df[label_col] == 1, ['s_0', 's_1']] = [0.05, 0.95]

    for i in range(1, len(borders)):
        smoothing_factor = min_major + (i - 1) * increase
        new_label = [round(1. - smoothing_factor, 2), round(0. + smoothing_factor, 2)]
        df.loc[(df[label_col] == 0) & (df[ih_col] >= borders[i - 1]), ['s_0', 's_1']] = new_label

    # Join s_0 and s_1 into a single column
    df['soft_label'] = df.apply(lambda row: [row['s_0'], row['s_1']], axis=1, result_type="reduce")

    del df['s_0']
    del df['s_1']

    return df
