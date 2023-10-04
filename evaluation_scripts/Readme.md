# Workflow.

## Getting recommendations and setting up spreadsheet
1. Run the get_recs.py file to get the recommendations for each model.
2. Save the form_builder_edited.xlsx to desktop file in the form_builder_data folder.
3. Upload the rec_images_for_gform folder to the google drive.
4. Run the get_gdrive_urls.py script to get the urls from the google drive and modify the form_builder_edited.xlsx file accordinly.
5. Import the finalized_form_builder.xlsx file into google sheets.
6. Run the google forms extension tool (`form builder plus`) to create the form from the spreadsheet.


# After collecting the form responses
1. Run the code in process_responses.py to change the responses to the original values.
2. Then upload the modified excel sheet onto google sheets.
3. Calculate the total scores for each model for each user.
4. Conduct a statistical test to see if the scores are significantly different from each other.