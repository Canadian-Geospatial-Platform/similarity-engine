# These are manually chosen uuids for each category.
category_mappings = {
    'foundation': ['8b23b631-8261-4076-81b2-3cd3f27f5ba0', '446b4220-6928-42eb-8d95-da0c67f22bc8'],
    'administration': ['9e1507cd-f25c-4c64-995b-6563bf9d65bd','65d3db23-b83c-4f49-ab93-65c59ee0e6aa'],
    'economy': ['05f72a28-01f3-4d8e-b6e8-71d2d494ab22', '11bccdd9-6de7-40fb-916f-021f5eff1683'],
    'emergency': ['f5c63b7b-7d05-49df-907a-910d178466d9', '4cedd37e-0023-41fe-8eff-bea45385e469'],
    'environment': ['f2d6263a-8b65-4350-9515-345875c6bebf', 'c3413230-b784-4e4a-befd-d325840235cc'],
    'imagery': ['65720669-2903-418b-a149-199bba914aad', 'ccmeo-root-Franklin-STAC-API'],
    'infrastructure': ['f7edef49-70da-1be8-5785-5f354ff6b7c1', '023ef1fd-878e-425d-a369-15c3e27a67f9'],
    'legal': ['f242b881-75e3-40bb-a148-63410b4ce2af', '0313f880-492c-4f4e-95ef-f53e4216576d'],
    'science': ['085024ac-5a48-427a-a2ea-d62af73f2142', 'd7f4628f-3abe-41fc-a61e-e10a7ab2be3a'],
    'society': ['69429e9f-74af-45d1-959c-7ed69564989a', 'bbe5770e-a82c-4a8f-a344-0e895f3e2c40']
}



# These uuids are truly random obtained from main.py inside src.
random_category_mappings = {
    # NOTE: the foundation category is taken from the category_mappings above. So it's not truly random.
    'foundation':['8b23b631-8261-4076-81b2-3cd3f27f5ba0', '446b4220-6928-42eb-8d95-da0c67f22bc8'],
    'administration': ['94ab3675-1b4c-7795-77e8-636f02cd2cf7', '69f4f070-d70a-43c8-b224-354758f4f492'],
    'economy':['b6129853-993a-4151-8806-946adf12bb98', '86d3ac53-60dc-0cbf-557a-249642376698'],
    'emergency': ['8c4f9a92-dfe7-4c9b-9e6e-10e66af9a769', 'ba580518-59e8-4d1c-b3ef-41d2658e6965'],
    'environment':['6d565968-2381-45fb-8059-06563d889b6f', 'd00389e0-66da-4895-bd56-39a0dd64aa78'],
    'imagery': ['ccmeo-hrdem-lidar-NB-3_BATHURST-1m', 'ccmeo-hrdem-lidar-QC-600020_36_Vaudreuil-Soulanges_MTM8_2020-21-1m'],
    'infrastructure': ['66c708e4-fd14-4bc7-980a-46749255372e', '22cd436e-607c-4cdc-b5bf-4726e09e9dec'],
    'science':['46dc6fa1-75e3-4d41-897b-ad3176a47265', '97283d35-6eb6-461c-bbf3-e1a8efa11d33'],
    'legal': ['6251438a-654e-4f2a-95c0-06917235e122', 'e5a936e5-35e3-413c-8b11-5d495f2bdb70'],
    'society':['8aab5e6e-6a02-3371-c206-ed1f0f9d7c85', '3d52693e-39d1-4e7f-b4fa-b72ccb605006']
}


model_order = {
    0: 'distilbert-base-uncased',
    1: 'distilroberta-base',
    2: 'bert-base-uncased',
    3: 'roberta-base',
}

model_order_list = [model_order[_] for _ in range(len(model_order))]

# This is used to randomized the order of the images in the form builder.
# This will also be used to obtain the original mappings before the statistical analysis.
choice_mapping_order = {
    0: [0, 1, 2, 3],
    1: [1, 0, 2, 3],
    2: [0, 2, 1, 3],
    3: [1, 2, 0, 3],
    4: [2, 0, 1, 3],
    5: [2, 1, 0, 3],
    6: [3, 0, 1, 2],
    7: [3, 1, 0, 2],
    8: [0, 3, 1, 2],
    9: [1, 3, 0, 2],
    10: [2, 3, 0, 1],
    11: [2, 0, 3, 1],
    12: [3, 2, 0, 1],
    13: [3, 0, 2, 1],
    14: [0, 2, 3, 1],
    15: [1, 2, 3, 0],
    16: [0, 1, 2, 3],
    17: [1, 0, 2, 3],
    18: [0, 2, 1, 3],
    19: [1, 2, 0, 3],
}



# This is just a placeholder dictionary which will be used to fill out the spreadsheet.
# Later on, the values will be replaced with the actual image urls.
recs_images_list = {
    0: ['q_0_option0.png', 'q_0_option1.png', 'q_0_option2.png', 'q_0_option3.png'],
    1: ['q_1_option0.png', 'q_1_option1.png', 'q_1_option2.png', 'q_1_option3.png'],
    2: ['q_2_option0.png', 'q_2_option1.png', 'q_2_option2.png', 'q_2_option3.png'],
    3: ['q_3_option0.png', 'q_3_option1.png', 'q_3_option2.png', 'q_3_option3.png'],
    4: ['q_4_option0.png', 'q_4_option1.png', 'q_4_option2.png', 'q_4_option3.png'],
    5: ['q_5_option0.png', 'q_5_option1.png', 'q_5_option2.png', 'q_5_option3.png'],
    6: ['q_6_option0.png', 'q_6_option1.png', 'q_6_option2.png', 'q_6_option3.png'],
    7: ['q_7_option0.png', 'q_7_option1.png', 'q_7_option2.png', 'q_7_option3.png'],
    8: ['q_8_option0.png', 'q_8_option1.png', 'q_8_option2.png', 'q_8_option3.png'],
    9: ['q_9_option0.png', 'q_9_option1.png', 'q_9_option2.png', 'q_9_option3.png'],
    10: ['q_10_option0.png', 'q_10_option1.png', 'q_10_option2.png', 'q_10_option3.png'],
    11: ['q_11_option0.png', 'q_11_option1.png', 'q_11_option2.png', 'q_11_option3.png'],
    12: ['q_12_option0.png', 'q_12_option1.png', 'q_12_option2.png', 'q_12_option3.png'],
    13: ['q_13_option0.png', 'q_13_option1.png', 'q_13_option2.png', 'q_13_option3.png'],
    14: ['q_14_option0.png', 'q_14_option1.png', 'q_14_option2.png', 'q_14_option3.png'],
    15: ['q_15_option0.png', 'q_15_option1.png', 'q_15_option2.png', 'q_15_option3.png'],
    16: ['q_16_option0.png', 'q_16_option1.png', 'q_16_option2.png', 'q_16_option3.png'],
    17: ['q_17_option0.png', 'q_17_option1.png', 'q_17_option2.png', 'q_17_option3.png'],
    18: ['q_18_option0.png', 'q_18_option1.png', 'q_18_option2.png', 'q_18_option3.png'],
    19: ['q_19_option0.png', 'q_19_option1.png', 'q_19_option2.png', 'q_19_option3.png']
}