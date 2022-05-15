import numpy as np
import pandas as pd
from pandarallel import pandarallel


def encode_ohe_matrix_2d(miRNA, gene, tensor_dim=(50, 20, 1)):
    """
    fun transform input database to one hot encoding numpy array.

    parameters:
    miRNA
    gene
    tensor_dim= 2d matrix shape

    output:
    2d dot matrix, labels as np array
    """

    # Check if input sequences have the expected length
    if (len(gene) > tensor_dim[0]) or (len(miRNA) > tensor_dim[1]):
        return None

    # initialize dot matrix with zeros
    ohe_matrix_2d = np.zeros(tensor_dim, dtype="float32")

    # compile matrix with watson-crick interactions.
    for bind_index, bind_nt in enumerate(gene):
        for mirna_index, mirna_nt in enumerate(miRNA):
            if (bind_nt == "A" and mirna_nt == "T") or (bind_nt == "T" and mirna_nt == "A") or (bind_nt == "G" and mirna_nt == "C") or (bind_nt == "C" and mirna_nt == "G"):
                ohe_matrix_2d[bind_index, mirna_index, 0] = 1

    return ohe_matrix_2d


def encode_and_label(row, label_col="label", one_hot_encode=False, sample_weight_col=None):
    """

    @param row: pandas.DataFrame Series object (dataFrame row)
    @param label_col: string - name of the column containing label
    @param one_hot_encode: boolean - defines if label should be returned one hot encoded
    @param sample_weight_col: string - name of the column containing sample weight
    @return: encoded miRNA-gene pair, label
    """

    ohe_matrix, label, weight = None, None, None

    # Encode miRNA - gene pair
    ohe_matrix = encode_ohe_matrix_2d(row['miRNA'], row['gene'])

    # Get label
    label = row[label_col]
    if one_hot_encode is True:
        label = [1., 0.] if label == 0 else [0., 1.]

    # Get weight
    if sample_weight_col is not None:
        weight = row[sample_weight_col]
        return ohe_matrix, label, weight

    return ohe_matrix, label


def get_test_datasets(data_path, dataset_type="evaluation"):

    """
    Function used namely to load evaluation and test datasets.

    @param data_path: string - path to directory containing datasets
    @param dataset_type: string - 'evaluation' or 'test'
    @return: pandas.DataFrame, pandas.DataFrame, pandas.DataFrame (datasets in 3 ratios)
    """

    pandarallel.initialize(progress_bar=True)

    # Load
    df_100 = pd.read_csv(f"{data_path}/{dataset_type}_set_1_100_CLASH2013_paper.tsv", sep="\t")
    df_10 = pd.read_csv(f"{data_path}/{dataset_type}_set_1_10_CLASH2013_paper.tsv", sep="\t")
    df_1 = pd.read_csv(f"{data_path}/{dataset_type}_set_CLASH2013_paper.tsv", sep="\t")

    # Encode
    print("Encoding set 1:100")
    data_100 = df_100[['miRNA', 'gene', 'label']].parallel_apply(encode_and_label, axis=1, result_type='expand')
    print("Encoding set 1:10")
    data_10 = df_10[['miRNA', 'gene', 'label']].parallel_apply(encode_and_label, axis=1, result_type='expand')
    print("Encoding set 1:1")
    data_1 = df_1[['miRNA', 'gene', 'label']].parallel_apply(encode_and_label, axis=1, result_type='expand')

    return data_1, data_10, data_100
