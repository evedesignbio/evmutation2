"""
Parts of code below based on OpenFold code according to following license:

# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

import argparse
import json
from pathlib import Path
from loguru import logger


# copied from OpenFold implementation in create_alignment_db_sharded.py to make script independent of rest of package
def split_file_list(file_list: list[Path], n_shards: int):
    """
    Split up the total file list into n_shards sublists.
    """
    split_list = []

    for i in range(n_shards):
        split_list.append(file_list[i::n_shards])

    assert len([f for sublist in split_list for f in sublist]) == len(file_list)

    return split_list


def create_database(
    input_file_dir: Path,
    output_db_prefix: Path,
    n_shards: int = 1,
):
    # retrieve all files
    all_files = [
        path for path in input_file_dir.rglob("*") if path.is_file()
    ]

    # make sure all names without any extensions are unique (e.g. if subdirectories or different extensions)
    unique_filenames = {
        path.name.split(".")[0] for path in all_files
    }
    if len(unique_filenames) != len(all_files):
        raise ValueError("Input directory contains duplicated filename")

    logger.info(f"Number of files to be included in database: {len(all_files)}")

    # split files into shards
    split_files = split_file_list(all_files, n_shards=n_shards)

    # initialize database index
    db_index = {}

    files_written = 0

    for shard_index, shard_files in enumerate(split_files):
        logger.info(f"Creating shard {shard_index}")
        db_offset = 0
        shard_name = output_db_prefix.stem + f"_{shard_index}.db"
        shard_path = output_db_prefix.with_name(shard_name)

        # iterate through assigned files and write to shard file
        with open(shard_path, "wb") as db:
            for input_file_path in shard_files:
                with open(input_file_path, "rb") as f:
                    # read contents and write to db file
                    file_bytes = f.read()
                    file_size = len(file_bytes)
                    db.write(file_bytes)

                    # add to index
                    filename_no_ext = input_file_path.name.split(".")[0]
                    db_index[filename_no_ext] = {
                        "db": shard_name,
                        "files": [[
                            input_file_path.name,
                            db_offset,
                            file_size
                        ]]
                    }

                    # compute starting offset of next file
                    db_offset += file_size

                files_written += 1

                if files_written % 100 == 0:
                    logger.info(f"... {files_written} total files written")

    assert files_written == len(all_files)
    logger.info(f"Finished writing a total of {files_written} files to database shards")

    index_path = output_db_prefix.with_suffix(".index")
    with open(index_path, "w") as f:
        json.dump(db_index, f, indent=4)

    logger.info("Saved index file")

    return db_index


if __name__ == "__main__":
    # arguments follow OpenProteinSet params in create_alignment_db.shared.py
    parser = argparse.ArgumentParser(
        description="""
        Merge individual files into shared database file in same format
        as OpenProteinSet alignment DB (note that unlike create_alignment_db_sharded.py,
        for simplicity this script only uses a single process and does not group multiple files
        in a single entry).
        """
    )
    parser.add_argument(
        "input_file_dir",
        type=Path,
        help="Directory containing all individual files to be included in database",
    )
    parser.add_argument("output_db_prefix", type=Path)
    parser.add_argument(
        "--n_shards",
        type=int,
        help="Number of shards to split the database into",
        default=1,
    )

    args = parser.parse_args()
    create_database(
        input_file_dir=args.input_file_dir,
        output_db_prefix=args.output_db_prefix,
        n_shards=args.n_shards,
    )
