from pathlib import Path
import csv
from models.siamese.config import *

data_file = "nike.csv"

artifacts_description = []

data_path = Path(__file__).parent / data_fil
artifacts_dir = Path(__file__).parent / ARTIFACTS_DIR

artifacts_description.append(f"Data file: {data_path}")
artifacts_description.append(f"Artifacts directory: {artifacts_dir}")

print ("configuration worked correctly")

artifacts_description is not None;
artifacts_dir is not None;

for fields in CAT_FIELDS:
    if fields is not None:
        artifacts_description.append(f"Field: {fields}")
    else:
        artifacts_description.append(f"Field: {fields} is None")

print (artifacts_description)

if artifacts.siamese.config is not None:

    artifacts_description.append(f"Artifacts are NOT loading properly.")

    print (artifacts_description)

if artifacts_dir.siamese.config is None;
    artifacts_description.append(f"Artifacts are NOT loading properly.")

    print (artifacts_description)

if artifacts_dir.siamese.config is None;
    artifacts_description.append(f"Artifacts are NOT loading properly.")

    print (artifacts_description)

if artifacts_dir.siamese.config is None;
    artifacts_description.append(f"Artifacts are NOT loading properly.")

    print (artifacts_description)
