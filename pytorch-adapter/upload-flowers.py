#!/usr/bin/env python

# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
This script downloads the Flowers dataset from <https://doi.org/10.7910/DVN/1ECTVN>
and uploads it to CVAT.
"""

import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from cvat_sdk import make_client, Client
from cvat_sdk.core.proxies.tasks import ResourceType
import cvat_sdk.models as models


DATASET_URL = "https://dataverse.harvard.edu/api/access/datafile/4105627"
SUBSETS = ("test", "train", "validation")


def create_tasks(ds_root: Path, client: Client) -> None:
    label_names = [dir.name for dir in (ds_root / SUBSETS[0]).iterdir()]

    print("Creating project...")

    project = client.projects.create(
        models.ProjectWriteRequest(
            "Flowers",
            labels=[models.PatchedLabelRequest(name=name) for name in label_names],
        )
    )

    print("Project created:", project.id)

    label_name_to_id = {label.name: label.id for label in project.labels}

    for subset in SUBSETS:
        image_paths = list((ds_root / subset).glob("*/*.jpg"))
        image_name_to_id = {
            image_path.name: label_name_to_id[image_path.parent.name]
            for image_path in image_paths
        }

        print(f"Creating task for the {subset} subset...")

        task = client.tasks.create_from_data(
            models.TaskWriteRequest(
                f"Flowers-{subset}",
                project_id=project.id,
                subset=subset,
            ),
            resource_type=ResourceType.LOCAL,
            resources=image_paths,
        )

        print("Task created:", task.id)
        print("Uploading annotations...")

        task.update_annotations(
            models.PatchedLabeledDataRequest(
                tags=[
                    models.LabeledImageRequest(
                        frame=frame_index,
                        label_id=image_name_to_id[frame.name],
                    )
                    for frame_index, frame in enumerate(task.get_frames_info())
                ]
            )
        )

        print("Annotations uploaded")


def main():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        flowers_zip_path = tmp_dir / "flowers.zip"

        print("Downloading dataset...")
        request = urllib.request.Request(DATASET_URL,
            # dataverse.harvard.edu blocks requests with the default urllib User-Agent
            headers={'User-Agent': 'upload-flowers'})
        with urllib.request.urlopen(request) as response, \
                open(flowers_zip_path, "wb") as flowers_zip_file:
            shutil.copyfileobj(response, flowers_zip_file)

        print("Unpacking dataset...")
        shutil.unpack_archive(flowers_zip_path, tmp_dir)
        ds_root = tmp_dir / "flowers/flower_photos"

        with make_client(
            os.getenv("CVAT_HOST", "app.cvat.ai"),
            credentials=(os.getenv("CVAT_USER"), os.getenv("CVAT_PASS")),
        ) as client:
            create_tasks(ds_root, client)


if __name__ == "__main__":
    main()
