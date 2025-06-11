from src.googledrivedownloader import download_file_from_google_drive

_crossing_id = '1H1ett7yg-TdtTt6mj2jwmeGZaC8iY1CH'
_200mb_file = '11TTYpqLJ0KAn103ozpD7XlMiRrwxAi8F'
_zip_issue_file = '1wEZ81eTtOQLGXBoGqGATWqWkxNM8F-x6'
_docs = '13nD8T7_Q9fkQzq9bXF2oasuIZWao8uio'
_20mb_file = '1iaqXel5NIsykC-_NYewXwZZwqK0F9h3h'


def main():
    download_file_from_google_drive(file_id=_20mb_file, dest_path='data/20mb-file.dat')
    # download_file_from_google_drive(file_id=_crossing_id, dest_path='data/crossing.jpg')
    # download_file_from_google_drive(file_id=_zip_issue_file, dest_path='data/zip-of-zips.zip')
    # download_file_from_google_drive(file_id=_docs, dest_path='data/docs.zip')
    # download_file_from_google_drive(file_id=_200mb_file, dest_path='data/200mb-file.dat')
    pass


if __name__ == '__main__':
    main()
