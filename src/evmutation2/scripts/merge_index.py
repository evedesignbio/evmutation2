import argparse
import json
from pathlib import Path
from loguru import logger


def merge_index(output_db_prefix: Path):
    # find all shard index files
    index_files = sorted(
        output_db_prefix.parent.glob(
            output_db_prefix.stem + "_*.index"
        )
    )

    db_files = list(
        output_db_prefix.parent.glob(
            output_db_prefix.stem + "_*.db"
        )
    )

    assert len(index_files) == len(db_files), "Number of index and db files does not match"

    db_index = {}
    expected_total_len = 0

    for index_file in index_files:
        with open(index_file) as f:
            cur_index = json.load(f)
            expected_total_len += len(cur_index)
            db_index.update(cur_index)
            logger.info(f"Added {index_file}")

    assert (
        len(db_index) == expected_total_len
    ), f"Wrong number of entries in index: {len(db_index)} vs {expected_total_len} expected"

    # write to joint index
    index_path = output_db_prefix.with_suffix(".index")
    with open(index_path, "w") as f:
        json.dump(db_index, f, indent=4)

    logger.info("Saved index file")


if __name__ == "__main__":
    # arguments follow OpenProteinSet params in create_alignment_db.shared.py
    parser = argparse.ArgumentParser(
        description="""
        Merge index files from parallelized execution of create_alignment_db_from_zip.py
        (using --target_shard parameter)
        """
    )
    parser.add_argument("output_db_prefix", type=Path)

    args = parser.parse_args()
    merge_index(args.output_db_prefix)
