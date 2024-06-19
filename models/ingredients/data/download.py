import requests
import os

file_meta = {
    "train.csv": {
        "id": "12NEGV4V9R75YuCefQrRkYZ-DnyAHkj10QsiDhsDvbBk",
        "gid": "819132495",
    },
    "validate.csv": {
        "id": "12NEGV4V9R75YuCefQrRkYZ-DnyAHkj10QsiDhsDvbBk",
        "gid": "805177748",
    },
}


def download_g_sheet(file_name):
    meta = file_meta[file_name]
    url = f"https://docs.google.com/spreadsheets/d/{meta['id']}/export?format=csv&gid={meta['gid']}"

    response = requests.get(url)

    if response.status_code == 200:
        with open(f"models/ingredients/data/{file_name}", "wb") as file:
            file.write(response.content)
    else:
        raise Exception("Failed to download file")


# Check if the directory exists and is accessible
print("Current Working Directory:", os.getcwd())


download_g_sheet("train.csv")
download_g_sheet("validate.csv")
