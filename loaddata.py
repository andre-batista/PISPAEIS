import gdown

# This script downloads data from a Google Drive folder to a local directory named 'data'.
url = "https://drive.google.com/drive/folders/1N0iAjMFQ9kaG4rDx6kGvklwJoJYoEFti?usp=drive_link"

# Download the data from Google Drive
gdown.download_folder(url, output='data', use_cookies=False)
print("Data downloaded successfully to 'data' folder")