import matplotlib

from deg_project.NN.NN_IG_imp import get_integrated_gradients
from deg_project.general.sequence_utilies import create_DNA_logo
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deg_project.NN import NN_utilies, NN_load_datasets

model_path_110_4 = "C:\\Users\\YuvalFroman\\PycharmProjects\\DeepUTR/files/saved_models_8_disjoint/ensemble/dynamics_-_CNN\\A_minus_model_8_points_model_8_points_id_1___20201127-005706.h5"
model_path_110_5 = "C:\\Users\\YuvalFroman\\PycharmProjects\\DeepUTR/files/saved_model_new_input/ensemble/dynamics_-_CNN\\dynamics_-_model_8_points_id_1___20220402-081403.h5"
initial_value_path = "C:\\Users\\YuvalFroman\\PycharmProjects\\DeepUTR\\files\\dataset\\validation_A_minus_normalized_levels.csv"
seq_path = "C:\\Users\\YuvalFroman\\PycharmProjects\\DeepUTR\\files\\dataset\\validation_seq.csv"
output_pdf_name = "C:\\Users\\YuvalFroman\\PycharmProjects\\DeepUTR\\result_IG.pdf"
model_type = 'dynamics'
data_type = '-'
multiple_samples = True
model_path = model_path_110_5
Logo_letters = ['A', 'C', 'G', 'T']
secondary_color = False
starting_index = 0
ending_index = 4

if type(model_path) is list:
    model_list = [tf.keras.models.load_model(model_path_item, custom_objects={'tf_pearson': NN_utilies.tf_pearson})
                  for model_path_item in model_path]
else:
    model_list = [tf.keras.models.load_model(model_path, custom_objects={'tf_pearson': NN_utilies.tf_pearson})]

validation_set = NN_load_datasets.load_dataset_model_type(seq_path=seq_path, labels_path_minus=initial_value_path,
                                                          labels_path_plus=None, model_type=model_type,
                                                          data_type=data_type, split=False)
(initial_values_features, one_hot_features, _) = validation_set
one_hot = one_hot_features[:, :, 0:4]
lunps = one_hot_features[:, :, 4:5]
explained_seq_fetures, _ = get_integrated_gradients(model_list[0], [one_hot, lunps, initial_values_features],
                                                    multiple_samples=multiple_samples, const_inputs=[1, 2])
explained_seq_fetures_letters = explained_seq_fetures[0][:, :, starting_index:ending_index] if multiple_samples else \
explained_seq_fetures[0][:, starting_index:ending_index]

# save the resutls in a pdf file if needed
if (output_pdf_name is not None):
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_pdf_name)
    explained_seq_fetures_letters_list = explained_seq_fetures_letters if multiple_samples else [
        explained_seq_fetures_letters]
    x = 0
    for explained_seq_fetures_letters_item in explained_seq_fetures_letters_list:
        # create Logo object
        explained_seq_fetures_letters_item = pd.DataFrame(explained_seq_fetures_letters_item, columns=Logo_letters)
        IG_logo = create_DNA_logo(PWM_df=explained_seq_fetures_letters_item,
                                  secondary_color=secondary_color)
        pdf.savefig(IG_logo.ax.figure)
        plt.close('all')
    pdf.close()
