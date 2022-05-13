import wget
import zipfile
import glob
import glob2
import os


def get_data(src,dst):
  """
  Download a file from a remote source to a local destination
  """
  #Define the remote file to retrieve
  #Exemple : remote_url = "https://dsti-aws-class-website-ravand.s3.eu-west-1.amazonaws.com/kagglecatsanddogs_5340.zip"
  remote_url = src
  #Define the local filename to save data
  # Exemple : local_dir = '/content/dog_cat_classification/data/raw'
  local_dir = dst
  wget.download(remote_url, local_dir)

def unzip_data(my_dir):
  """Unpack the first zip file in a directory"""
  local_dir = my_dir
  my_file = glob2.glob(f'{local_dir}/*.zip')[0]
  with zipfile.ZipFile(my_file,'r') as zip_ref:
    zip_ref.extractall(my_dir)

if __name__ == "__main__":
  print("starting download")
  get_data('https://dsti-aws-class-website-ravand.s3.eu-west-1.amazonaws.com/kagglecatsanddogs_5340.zip',
            '/content/dog_cat_classification/data/raw')
  unzip_data('/content/dog_cat_classification/data/raw')
  #Make http request for remote file data