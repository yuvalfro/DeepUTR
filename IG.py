import matplotlib
from deg_project.NN.NN_IG_imp import get_integrated_gradients
from deg_project.general.sequence_utilies import create_DNA_logo
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deg_project.NN import NN_utilies, NN_load_datasets
from PyPDF3 import PdfFileWriter, PdfFileReader
from PyPDF3.pdf import PageObject

model_path_110_4 = "C:\\Users\\YuvalFroman\\PycharmProjects\\DeepUTR/files/saved_models_8_disjoint/ensemble/dynamics_-_CNN\\A_minus_model_8_points_model_8_points_id_1___20201127-005706.h5"
model_path_110_5 = "C:\\Users\\YuvalFroman\\PycharmProjects\\DeepUTR/files/saved_model_new_input/ensemble/dynamics_-_CNN\\dynamics_-_model_8_points_id_1___20220402-081403.h5"
initial_value_path = "C:\\Users\\YuvalFroman\\PycharmProjects\\DeepUTR\\files\\dataset\\validation_A_minus_normalized_levels.csv"
seq_path = "C:\\Users\\YuvalFroman\\PycharmProjects\\DeepUTR\\files\\dataset\\validation_seq.csv"
output_pdf_name1 = "result_IG - lunps in const.pdf"
output_pdf_name2 = "result_IG - one hot in const.pdf"
model_type = 'dynamics'
data_type = '-'
multiple_samples = True
model_path = model_path_110_5
Logo_letters = ['A', 'C', 'G', 'T']
Logo_symbol = ['I']
secondary_color = False
starting_index = 0
ending_index = 4


def load_data():
    seq = pd.read_csv(seq_path)["seq"].values.tolist()
    seq = np.array([np.array(list(s)) for s in seq])
    lunps = pd.read_csv("lunps_results.csv").values.tolist()
    for lunp in lunps:
        del lunp[0:29]  # Delete index+28 first elements
        del lunp[110:138]  # Delete last 28 elements according to new index after delete the first 28

    lunps = np.array([np.array(lunp) for lunp in lunps])
    features = np.concatenate((seq, lunps))
    return features


###########################
# THIS CODE CREATE THE EXCEL FILE
def create_excel():
    features = load_data()
    df = pd.DataFrame(features, columns=range(110))
    # table = df.loc[[0,58]]
    # table.to_excel("test.xlsx")
    df_list = []
    keys = []
    for i in range(int(len(features) / 2)):
        table = df.loc[[i, i + 58]]
        df_list.append(table)
        keys.append(f"Sequence {i + 1}")
    df_to_excel = pd.concat(df_list, keys=keys)
    df_to_excel.to_excel("test.xlsx")


#########################


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

##### LUNPS is CONST, ONE HOT is INTERPULATE
explained_seq_fetures1, _ = get_integrated_gradients(model_list[0], [one_hot, lunps, initial_values_features],
                                                     multiple_samples=multiple_samples, const_inputs=[1, 2])
explained_seq_fetures_letters1 = explained_seq_fetures1[0][:, :, starting_index:ending_index] if multiple_samples else \
    explained_seq_fetures1[0][:, starting_index:ending_index]

##### ONE HOT is CONST, LUNPS is INTERPULATE
_, explained_seq_fetures2 = get_integrated_gradients(model_list[0], [one_hot, lunps, initial_values_features],
                                                     multiple_samples=multiple_samples, const_inputs=[0, 2])
explained_seq_fetures_letters2 = explained_seq_fetures2[1][:, :, 0:1] if multiple_samples else \
    explained_seq_fetures2[1][:, 0:1]

# save the results in a pdf file
pdf1 = matplotlib.backends.backend_pdf.PdfPages(output_pdf_name1)
pdf2 = matplotlib.backends.backend_pdf.PdfPages(output_pdf_name2)

explained_seq_fetures_letters_list1 = explained_seq_fetures_letters1 if multiple_samples else [
    explained_seq_fetures_letters1]
explained_seq_fetures_letters_list2 = explained_seq_fetures_letters2 if multiple_samples else [
    explained_seq_fetures_letters2]

for explained_seq_fetures_letters_item1, explained_seq_fetures_letters_item2 in zip(explained_seq_fetures_letters_list1,
                                                                                    explained_seq_fetures_letters_list2):
    # create Logo object
    explained_seq_fetures_letters_item = pd.DataFrame(explained_seq_fetures_letters_item1, columns=Logo_letters)
    IG_logo1 = create_DNA_logo(PWM_df=explained_seq_fetures_letters_item,
                               secondary_color=secondary_color)

    pdf1.savefig(IG_logo1.ax.figure)

    explained_seq_fetures_letters_item2 = explained_seq_fetures_letters_item2[
        explained_seq_fetures_letters_item2 != 0].reshape(-1, 1)
    explained_seq_fetures_letters_item = pd.DataFrame(explained_seq_fetures_letters_item2, columns=Logo_symbol)
    IG_logo2 = create_DNA_logo(PWM_df=explained_seq_fetures_letters_item,
                               secondary_color=True)

    pdf2.savefig(IG_logo2.ax.figure)

    plt.close('all')
pdf1.close()
pdf2.close()

# merge both pdf to have two graphs in the same page
input1 = PdfFileReader(open(output_pdf_name1, "rb"), strict=False)
input2 = PdfFileReader(open(output_pdf_name2, "rb"), strict=False)

outputpdf = open("result_IG - merged.pdf", "wb")
output = PdfFileWriter()

for i in range(58):
    page1 = input1.getPage(i)
    page2 = input2.getPage(i)
    total_width = max([page1.mediaBox.upperRight[0], page2.mediaBox.upperRight[0]])
    total_height = page1.mediaBox.upperRight[1] + page2.mediaBox.upperRight[1]

    new_page = PageObject.createBlankPage(None, total_width, total_height)

    # Add first page at the 0,0 position
    new_page.mergePage(page2)
    # Add second page under the first page
    new_page.mergeTranslatedPage(page1, page1.mediaBox.lowerLeft[0], page1.mediaBox.upperLeft[1], True)
    output.addPage(new_page)
output.write(outputpdf)

test_results = NN_utilies.IG_test("pearson", [explained_seq_fetures_letters1, explained_seq_fetures_letters2],
                                  'model_8_points_id_1', 'dynamics', None, True)
print(test_results)
