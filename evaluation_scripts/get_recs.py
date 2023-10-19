"""
This file is used to retrieve the recommendations from the each model.
"""


# Import libraries.
import ast
import numpy as np
import pandas as pd
import sys


import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


sys.path.append("/home/rsaha/projects/similarity-engine/src/")
from utils.data_loading import load_processed_parquet

from constants import category_mappings, random_category_mappings, model_order, \
    model_order_list, choice_mapping_order, recs_images_list

model_name = 'distilbert-base-uncased'

file_name = f'/home/rsaha/projects/similarity-engine/data/sep_15_{model_name}_full_finetune_df_sim_records.csv'
num_recs = 5



# The random_category_mappings is a dictionary with the key as the category and the value as a list of uuids.
seed_records_uuids = []
for key, value in random_category_mappings.items():
    seed_records_uuids.extend(value)
    
print("uuids_set length: ", len(seed_records_uuids))

# TODO: Select the records from the seed_records_uuids list.
# seed_records_uuids = ['bbe5770e-a82c-4a8f-a344-0e895f3e2c40', 'uuid2', 'uuid3', 'uuid4', 'uuid5','uuid6', 'uuid7', 'uuid8', 'uuid9', 'uuid10',
#                         'uuid11', 'uuid12', 'uuid13', 'uuid14', 'uuid15','uuid16', 'uuid17', 'uuid18', 'uuid19', 'uuid20',
#                         'uuid21', 'uuid22', 'uuid23', 'uuid24', 'uuid25']



# Load the sim_records.csv file.

# NOTE: Do this for each model. Change this using the sim_records variable.
for model_num in model_order.keys():
    print("model_num: ", model_num)
          
    if model_num == 0:
        sim_records = pd.read_csv(f"/home/rsaha/projects/similarity-engine/data/sep_15_distilbert-base-uncased_full_finetune_df_sim_records.csv")
    elif model_num == 1:
        sim_records = pd.read_csv(f"/home/rsaha/projects/similarity-engine/data/sep_15_distilroberta-base_full_finetune_df_sim_records.csv")
    elif model_num == 2:
        sim_records = pd.read_csv(f"/home/rsaha/projects/similarity-engine/data/sep_17_bert-base-uncased_pretrained_df_sim_records.csv")
    elif model_num == 3:
        sim_records = pd.read_csv(f"/home/rsaha/projects/similarity-engine/data/sep_17_roberta-base_pretrained_df_sim_records.csv")
        
        
    all_sim_records = []
    all_titles = []
    for i, uuid in enumerate(seed_records_uuids):
        sim_records_for_uuid = []
        # Get the top k uuids.
        seed_title = sim_records[sim_records['features_properties_id'] == uuid]['features_properties_title_en'].values[0]
        all_titles.append(seed_title)
        top_k_uuids = ast.literal_eval(sim_records[sim_records['features_properties_id'] == uuid]['top_20_similar_uuid'].values[0])[:num_recs]
        for k_uuid in top_k_uuids:
            # Get the title for each uuid. Sorted in descending order.
            title = sim_records[sim_records['features_properties_id'] == k_uuid]['features_properties_title_en'].values[0]
            sim_records_for_uuid.append(title)
        print("sim_records_for_uuid: ", sim_records_for_uuid)
        all_sim_records.append(sim_records_for_uuid)
        
        

        # Create an image with the titles of the top_k_titles on each separate line.
        # Create a white background image
      # Adjust the figure size as needed

        # Define the number of lines and text for each line
        num_lines = num_recs
        
        # Doing one question at a time.
        line_texts = sim_records_for_uuid  # Reverse it so the most similar is on top.
        
        split_lines = {}
        line_spacing = 1  # Adjust as needed
        # text_height = 0.05  # Adjust as needed
        max_line_len = 45 # characters.
        plt.clf()
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Set axis limits and hide the axis
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Preprocess the line texts so that the long lines are split into multiple lines.
        flag = 0
        for line_index, line_text in enumerate(line_texts):
            y = (num_recs - line_index - line_spacing) / (num_recs + 1) - flag + 0.5
            if len(line_text) > max_line_len:
                words = line_text.split()
                current_line = ''
                lines = []
                for word in words:
                    if len(current_line) + len(word) + 1 <= max_line_len:
                        current_line += word + ' '
                    else:
                        lines.append(current_line)
                        current_line = word + ' '
                if len(current_line) > 0:
                    lines.append(current_line)
                # print("lines: ", lines)
                # lines.append(line_text[:max_line_len])
                # lines.append(line_text[max_line_len:])
                
                ax.text(0.05, y, f"{str(line_index + 1)}. " + lines[0], fontsize=20, ha='left', va='top')
                sub_y = 0
                for line in lines[1:]:
                    flag += 0.1
                    sub_y = sub_y + 0.1
                    ax.text(0.05, y - sub_y, "    " + line, fontsize=20, ha='left', va='top')                
                # flag = flag_count
            else:
                ax.text(0.05, y, f"{str(line_index + 1)}. " + line_text, fontsize=20, ha='left', va='top')
                # flag = 0
            
                

        # Save the image
        plt.savefig(f"/home/rsaha/projects/similarity-engine/rec_images_for_gform/q_{i}_option{model_num}.png", bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    #     # Display the image (optional)
    #     plt.show()




### Make a apndas dataframe with questions and answers.

form_template_df = pd.read_excel("/home/rsaha/projects/similarity-engine/form_builder_data/Form builder template for model voting.xlsx")

all_questions = []
question_prefix = "What is the most relevant group of recommendations for the following record? \n"
for _, title in enumerate(all_titles):
    question = f"Question {_+1}: " + question_prefix + f"\n \n {title}\n \n"
    all_questions.append(question)
    
# Replace the 'Question' column in the form_template_df with the all_questions list.
form_template_df['Question'] = all_questions  # NOTE: Doing 5 for test.
# Drop the images column.




# Now change the choice values with the recommendations but use the order defined for each question in the choice_mapping_order.
for k, v in choice_mapping_order.items():
    print("k: ", k)
    print("v: ", v)
    print("recs_images_list[k]: ", recs_images_list[k])
    form_template_df.iloc[k, 4] = f"1|{recs_images_list[k][v[0]]}|1"
    form_template_df.iloc[k, 5] = f"2|{recs_images_list[k][v[1]]}|2"
    form_template_df.iloc[k, 6] = f"3|{recs_images_list[k][v[2]]}|3"
    form_template_df.iloc[k, 7] = f"4|{recs_images_list[k][v[3]]}|4"
    
form_template_df.drop(columns=['Image'], inplace=True)

# print("form_template_df: ", form_template_df.head())

# Finally, save the form_template_df as a excel file.
form_template_df.to_excel("/home/rsaha/projects/similarity-engine/form_builder_data/form_builder_edited.xlsx", index=False)


