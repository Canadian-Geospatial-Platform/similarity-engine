from __future__ import print_function

import os.path

import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from constants import choice_mapping_order, model_order, recs_images_list

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive']


def main():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Make sure to add the credentials.json file in the same directory. Download this from the Google Drive API.
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    images_name_and_urls = {}
    try:
        service = build('drive', 'v3', credentials=creds)

        # Call the Drive v3 API
        results = service.files().list(
            q="	mimeType = 'application/vnd.google-apps.folder' and name='rec_images_for_gform'",
            pageSize=100, fields="nextPageToken, files(id, name)").execute()
        folder = results.get('files', [])

        # Get files
        results = service.files().list(q="'" + folder[0].get('id') + "' in parents", pageSize=100,
                                     fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])



        # Change permissions to get shareable link
        request_body = {
            'role': 'reader',
            'type': 'anyone'
        }
        for item in items:

            response = service.permissions().create(
                fileId=item['id'],
                body=request_body
            ).execute()
            url = service.files().get(
                fileId=item['id'],
                fields='webViewLink'
            ).execute()
            print(url)

            images_name_and_urls[item['name']] = url['webViewLink']

        if not items:
            print('No files found.')
            return
    except HttpError as error:
        # TODO(developer) - Handle errors from drive API.
        print(f'An error occurred: {error}')


    # Now regroup the files based on the key.
    #
    form_build_edited = pd.read_excel("/home/rsaha/projects/similarity-engine/form_builder_data/form_builder_edited.xlsx")


    for k, v in choice_mapping_order.items():
        print("k: ", k)
        print("v: ", v)
        print("recs_images_list[k]: ", recs_images_list[k])
        form_build_edited.iloc[k, 3] = f"1|{images_name_and_urls[recs_images_list[k][v[0]]]}|1-{recs_images_list[k][v[0]]}"
        form_build_edited.iloc[k, 4] = f"2|{images_name_and_urls[recs_images_list[k][v[1]]]}|2-{recs_images_list[k][v[1]]}"
        form_build_edited.iloc[k, 5] = f"3|{images_name_and_urls[recs_images_list[k][v[2]]]}|3-{recs_images_list[k][v[2]]}"
        form_build_edited.iloc[k, 6] = f"4|{images_name_and_urls[recs_images_list[k][v[3]]]}|4-{recs_images_list[k][v[3]]}"

    form_build_edited.to_excel("/home/rsaha/projects/similarity-engine/form_builder_data/finalized_form_builder.xlsx", index=False)


if __name__ == '__main__':
    main()