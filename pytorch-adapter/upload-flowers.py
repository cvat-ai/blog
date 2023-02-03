#!/usr/bin/env python

# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
This script downloads the Flowers dataset from <https://doi.org/10.7910/DVN/1ECTVN>
and uploads a fraction of it to CVAT.
"""

import argparse
import contextlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterator

from cvat_sdk import make_client, Client
from cvat_sdk.core.proxies.tasks import ResourceType
import cvat_sdk.models as models


DATASET_URL = "https://dataverse.harvard.edu/api/access/datafile/4105627"
SUBSETS = ("test", "train", "validation")

@contextlib.contextmanager
def long_action(description: str) -> Iterator[None]:
    """Helper function to print "Doing something... done" messages."""
    print(description + "...", end='', flush=True)
    try:
        yield
    except:
        print() # the error should begin on the next line
        raise
    print(" done")


def create_tasks(ds_root: Path, client: Client, fraction: float) -> None:
    # First we'll create a project to hold the common set of labels for
    # our dataset. To determine the label names, just get the subdirectory
    # names in one of the subset directories.
    label_names = [dir.name for dir in (ds_root / SUBSETS[0]).iterdir()]

    with long_action("Creating project"):
        project = client.projects.create(
            models.ProjectWriteRequest(
                "Flowers",
                labels=[models.PatchedLabelRequest(name=name) for name in label_names],
            )
        )

    print("  project ID:", project.id)

    # Now that the project has been created, determine the internal ID
    # for each label. We will need this information to create annotations.
    label_name_to_id = {label.name: label.id for label in project.labels}

    # Now we'll create one task for each subset.
    for subset in SUBSETS:
        # Get the images that belong to this subset.
        image_paths = []
        for label_name in label_names:
            # Get all images with this label
            images_for_label = list((ds_root / subset / label_name).glob("*.jpg"))

            # Keep only a fraction of these images
            num_to_keep = round(len(images_for_label) * fraction)
            images_for_label = sorted(images_for_label)[:num_to_keep]

            image_paths += images_for_label

        with long_action(f"Creating task for the {subset} subset"):
            task = client.tasks.create_from_data(
                models.TaskWriteRequest(
                    f"Flowers-{subset}",
                    project_id=project.id,
                    subset=subset,
                ),
                resource_type=ResourceType.LOCAL,
                resources=image_paths,
            )

        print("  task ID:", task.id)

        # Now determine the label ID corresponding to each image.
        # The parent directory of each image is named after the label,
        # so look that name up in `label_name_to_id`.
        image_name_to_id = {
            image_path.name: label_name_to_id[image_path.parent.name]
            for image_path in image_paths
        }

        with long_action(f"Uploading annotations for the {subset} subset"):
            task.update_annotations(
                models.PatchedLabeledDataRequest(
                    tags=[
                        # Each annotation must be associated with a frame index.
                        # We don't know in advance which frame will have which index,
                        # so we query `task.get_frames_info()` to find out.
                        models.LabeledImageRequest(
                            frame=frame_index,
                            label_id=image_name_to_id[frame.name],
                        )
                        for frame_index, frame in enumerate(task.get_frames_info())
                    ]
                )
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction', type=float, default=0.1,
        help="Fraction of the dataset to upload")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        flowers_zip_path = tmp_dir / "flowers.zip"

        with long_action("Downloading dataset"):
            request = urllib.request.Request(DATASET_URL,
                # dataverse.harvard.edu blocks requests with the default urllib User-Agent
                headers={'User-Agent': 'upload-flowers'})
            with urllib.request.urlopen(request) as response, \
                    open(flowers_zip_path, "wb") as flowers_zip_file:
                shutil.copyfileobj(response, flowers_zip_file)

        # The SDK can only upload images from files on the system,
        # so we'll have to unpack the archive.
        with long_action("Unpacking dataset"):
            shutil.unpack_archive(flowers_zip_path, tmp_dir)

        # The structure of the archive is as follows:
        # flowers/
        #   flower_photos/
        #     test/
        #       daisy/
        #         134409839_71069a95d1_m.jpg
        #         ...
        #       dandelion/
        #         ...
        #       ...
        #     train/
        #       ...
        #     validation/
        #       ...
        #
        # Since everything is inside `flowers/flower_photos`, use
        # that directory as the root.
        ds_root = tmp_dir / "flowers/flower_photos"

        # `make_client`` establishes a connection to the server
        # with the given credentials.
        with make_client(
            os.getenv("CVAT_HOST", "app.cvat.ai"),
            credentials=(os.getenv("CVAT_USER"), os.getenv("CVAT_PASS")),
        ) as client:
            create_tasks(ds_root, client, args.fraction)


if __name__ == "__main__":
    main()
