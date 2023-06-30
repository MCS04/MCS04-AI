"""
Extracting data from the Google Drive (Google Drive is our home)
Requirements:
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""
from __future__ import print_function
import io
import os.path
import google.auth

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload


SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly',
          "https://www.googleapis.com/auth/drive.readonly"]


def find_file_ids(d, creds):
    items = None
    dataset_folder_id = "1QkXA-nHZDEvHyGyAlP1ufnAhba-hO6FI"
    drive_id = "0AP5OUkOYBj0_Uk9PVA"

    try:
        service = build('drive', 'v3', credentials=creds)

        for key in d.keys():
            q_param = f"'{d[key]}' in parents and trashed=false"
            results = service.files().list(pageSize=10, q=q_param, supportsAllDrives=True,
                                           includeItemsFromAllDrives=True, fields="nextPageToken, files(id, name)").execute()
            items = results.get('files', [])
            if not items:
                print('No files found.')
            print('Files:')
            for idx, item in enumerate(items):
                print(u'{0} ({1})'.format(item['name'], item['id']))
                download_file(creds, item['id'], key,
                              idx, idx == 0, item['name'])

    except HttpError as error:
        # TODO(developer) - Handle errors from drive API.
        print(f'An error occurred: {error}')

    return items


def download_file(creds, real_file_id, key, index, directory_flag, name):
    try:
        # create drive api client
        service = build('drive', 'v3', credentials=creds)

        file_id = real_file_id

        # IMPORTANT: YOUR DIRECTORY HAS TO BE THE SAME AS HOW THE GITHUB SHOULD BE, IE: D:\Year_3_Sem_1\FIT3161\Deep Learning\ebn-classification> with the ebn-classification at the end
        current_dir = os.getcwd()
        directory = os.path.join(
            current_dir, f"{current_dir}\EBN-Grade-Classification\data\{key}")

        if directory_flag and not os.path.exists(directory):
            os.makedirs(directory)

        file_path = f"{current_dir}/EBN-Grade-Classification/data/{key}/{name}"
        if os.path.exists(file_path):
            return

        request = service.files().get_media(fileId=file_id)
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(F'Download {int(status.progress() * 100)}.')

    except HttpError as error:
        print(F'An error occurred: {error}')
        file = None

    if file is not None:
        result = file.getvalue()

        with open(f"{current_dir}/EBN-Grade-Classification/data/{key}/{name}", "wb") as f:
            f.write(result)
            print("Saved")
    else:
        print("File downloaded failed!")


def get_credentials():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds


def start_extracting():
    creds = get_credentials()
    d = {"a": "1vRzFoK6DQGx3_88slAR5FgIZ3JFl29HB", "b": "1xeBvqFRJYFv88qoPp5KgK3t9EoTrJqaW",
         "c": "1Hxytx8_7E2oc6KZ-uebkwqguR5gVKsZz", "d": "1FRULxbpH4tDA_LQDmocbvnLk6-XqrZB5", "others": "1iZnH8CBrtfnPMJHYneQ24iFEQkoYJRI1"}

    # search for all the files in the respective folders
    find_file_ids(d, creds)


if __name__ == '__main__':
    # IMPORTANT: YOUR DIRECTORY HAS TO BE THE SAME AS HOW THE GITHUB SHOULD BE, IE: D:\Year_3_Sem_1\FIT3161\Deep Learning\ebn-classification> with the ebn-classification at the end
    start_extracting()
