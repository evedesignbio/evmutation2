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
import gzip
import zipfile
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


def get_first_sequence(fasta_contents: str):
    header = None
    target = None

    for line in fasta_contents.split("\n", maxsplit=100):
        if line.startswith(">"):
            # start reading
            if target is None:
                target = ""
                header = line.strip()
                continue
            # reached next sequence, stop
            else:
                break
        elif line.startswith("#"):
            # skip comment lines
            continue
        else:
            # accumulate current sequence
            target += line.strip()

    return header, target


def load_cluster_file(cluster_file_path: Path):
    with open(cluster_file_path) as f:
        member_to_rep = {
            (s := line.strip().split("\t"))[1]: s[0] for line in f
        }

    return member_to_rep


def count_clusters(a3m: str, member_to_rep: dict[str, str]):
    # extract identifiers
    ids = [
        line.split("|")[1] for line in a3m.split("\n") if line.startswith(">")
    ]

    # map to unique cluster representatives
    clusters = {
        member_to_rep.get(id_) for id_ in ids
    }

    return clusters, len(clusters), len(ids)


def create_database(
    input_file_dir: Path,
    output_db_prefix: Path,
    n_shards: int = 1,
    target_shard: int | None = None,
    member_to_rep: dict[str, str] | None = None,
    filter_singletons: bool = True,
):
    # retrieve all files; sort to allow parallelized creation of different shards
    all_files = sorted([
        path for path in input_file_dir.rglob("*.zip") if path.is_file()
    ])

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
        # only create target shard if specified, skip all others
        if target_shard is not None and shard_index != target_shard:
            logger.info(f"Skipping shard {shard_index} as target_shard is {target_shard}")
            continue

        logger.info(f"Creating shard {shard_index}")
        db_offset = 0
        shard_name = output_db_prefix.stem + f"_{shard_index}.db"
        shard_path = output_db_prefix.with_name(shard_name)

        # iterate through assigned files and write to shard file
        with open(shard_path, "wb") as db:
            # iterate through archive files (zip)
            for input_file_path in shard_files:
                with zipfile.ZipFile(input_file_path, "r") as zip:
                    # iterate files inside zip archive
                    for i, name in enumerate(zip.namelist()):
                        name_path = Path(name)
                        # skip any non-alignment files
                        if not name_path.suffix == ".a3m":
                            continue

                        # read file from zip archive
                        data = zip.read(name)

                        # decode to string for checking alignment properties
                        fasta_decoded = data.decode("utf-8")

                        # determine number of sequences and length of query

                        header, target = get_first_sequence(fasta_decoded)
                        target_len = len(target)

                        if member_to_rep is not None:
                            cur_clusters, num_clusters, num_seqs = count_clusters(
                                fasta_decoded, member_to_rep
                            )

                            # -1: remove target sequence from count
                            num_seqs -= 1

                            if None in cur_clusters:
                                logger.warning(f"Unmappable sequence cluster for {input_file_path} -> {name}")
                        else:
                            # -1: remove target sequence from count
                            num_seqs = fasta_decoded.count(">") - 1
                            num_clusters = None

                        # skip current alignment if singletons should be filtered
                        if filter_singletons and num_seqs <= 1:
                            continue

                        # create unique entry ID
                        run_number = name_path.parent.stem
                        sequence_id = header.split("|")[1]
                        entry_id = f"{sequence_id}_{run_number}"

                        # compress data again (this time as individual gzip that can be seeked in DB)
                        file_bytes = gzip.compress(data)
                        file_size = len(file_bytes)

                        # write gzipped data into database
                        db.write(file_bytes)

                        # add entry to index
                        db_index[entry_id] = {
                            "db": shard_name,
                            "files": [[
                                name_path.name + ".gz",
                                db_offset,
                                file_size
                            ]],
                            "meta": {
                                "num_seqs": num_seqs,
                                "target_len": target_len,
                                "num_clusters": num_clusters,
                            }
                        }

                        # compute starting offset of next file
                        db_offset += file_size

                        files_written += 1

                        if files_written % 100 == 0:
                            logger.info(f"... {files_written} total files written")

        # save shard sub-index
        if target_shard is not None:
            index_path = shard_path.with_suffix(".index")
            with open(index_path, "w") as f:
                json.dump(db_index, f, indent=4)

            logger.info("Saved shard-specific index file")

    logger.info(f"Finished writing a total of {files_written} files to database shards")

    # save index file for entire database, but only if no target shard is specified (this case requires
    # merging at a later point when all parallel jobs are finished)
    if target_shard is None:
        index_path = output_db_prefix.with_suffix(".index")
        with open(index_path, "w") as f:
            json.dump(db_index, f, indent=4)

        logger.info("Saved index file")
    else:
        logger.info("Not saving global index file due to specified target_shard")

    return db_index


if __name__ == "__main__":
    # arguments follow OpenProteinSet params in create_alignment_db.shared.py
    parser = argparse.ArgumentParser(
        description="""
        Merge individual zip files into shared database file in same format
        as OpenProteinSet alignment DB (note that unlike create_alignment_db_sharded.py,
        for simplicity this script only uses a single process and does not group multiple files
        in a single entry).
        """
    )
    parser.add_argument(
        "input_file_dir",
        type=Path,
        help="Directory containing all zip archive files to be included in database",
    )
    parser.add_argument("output_db_prefix", type=Path)
    parser.add_argument(
        "--n_shards",
        type=int,
        help="Number of shards to split the database into",
        default=1,
    )
    parser.add_argument(
        "--targetshard",
        type=int,
        help="Create shard with this index only (for parallelization)",
        default=None,
    )
    parser.add_argument(
        "--cluster_file_path",
        type=Path,
        help="Tab-separated file from uniclust mapping with representative and cluster members in 1st/2nd columns",
        default=None,
    )

    args = parser.parse_args()

    if args.cluster_file_path is not None:
        cluster_mapping = load_cluster_file(args.cluster_file_path)
    else:
        cluster_mapping = None

    create_database(
        input_file_dir=args.input_file_dir,
        output_db_prefix=args.output_db_prefix,
        n_shards=args.n_shards,
        target_shard=args.targetshard,
        member_to_rep=cluster_mapping,
        filter_singletons=True,  # not exposed as command-line parameter
    )
