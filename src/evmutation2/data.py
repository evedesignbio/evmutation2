"""
Anything dataset related (except feature extraction)

    # # use 20% of training data for validation
    # train_set_size = int(len(train_set) * 0.8)
    # valid_set_size = len(train_set) - train_set_size
    #
    # # split the train set into two
    # seed = torch.Generator().manual_seed(42)
    # train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)
    # train_loader = DataLoader(train_set)
    # valid_loader = DataLoader(valid_set)
"""
from functools import partial
from io import BytesIO
import json
from pathlib import Path
from os import PathLike
import gzip

import numpy as np
import pandas as pd
import pyarrow
import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from loguru import logger

from evmutation2.features import (
    batch_features, extract_msa_feature_data, prepare_msa_features, add_seqs_to_decode, filter_template_hits,
    add_template_features, prepare_template_features
)
from evmutation2.structures import PDB, ResourceError
from evmutation2.parsers import parse_a3m, parse_hhr
from evmutation2.utils import GAP, AA_TO_INDEX


def transform_index(index: dict) -> pd.DataFrame:
    """
    Map old index (JSON format) to new Dataframe-based format
    """
    suffixes = ["name", "start", "size"]

    def _map(k, v):
        filemap = {}

        for file_info in v["files"]:
            file_name = file_info[0]
            if ".a3m" in file_name:
                prefix = "alignment"
            elif ".hhr" in file_name:
                prefix = "template"
            else:
                # unknown file type
                continue

            filemap = {
                **filemap,
                **{f"{prefix}_{suffix}": val for suffix, val in zip(suffixes, file_info)}
            }

        if "meta" in v:
            metamap = v["meta"]
        else:
            metamap = {}

        return {
            "id": k,
            "db": v["db"],
            **filemap,
            **metamap,
        }

    # only map entries where at least one file is specified (some are empty in filtered OpenProteinSet)
    return pd.DataFrame(
        (_map(k, v) for k, v in index.items() if len(v["files"]) > 0)
    )


class SeqStructDataset(Dataset):
    """
    MSA/template hit dataset, inspired by OpenFold
    """
    def __init__(
        self,
        index_file_path: str | PathLike,
        crop_size: int = 256,
        max_seqs_per_msa: int = 16384,
        truncate_encoder_msa: bool = True,
        decoder_seqs_from_full_msa: bool = True,
        decoder_sample_size: int = 64,
        decoder_weight_by_gaps: bool = True,
        decoder_sample_with_replacement: bool = False,
        index_table_query: str | None = None,
        use_templates: bool = False,
        pdb_index_file_path: str | PathLike | None = None,
        max_templates: int = 2,
        template_max_evalue: float | None = 1e-05,
        template_min_crop_overlap: int | None = 64,
        template_limit_hits: int | None = 20,
    ):
        super().__init__()
        self.index_file_path = index_file_path
        self.crop_size = crop_size
        self.max_seqs_per_msa = max_seqs_per_msa
        self.truncate_encoder_msa = truncate_encoder_msa
        self.decoder_seqs_from_full_msa = decoder_seqs_from_full_msa
        self.decoder_sample_size = decoder_sample_size
        self.decoder_weight_by_gaps = decoder_weight_by_gaps
        self.decoder_sample_with_replacement = decoder_sample_with_replacement

        self.use_templates = use_templates
        self.max_templates = max_templates
        self.pdb_index_file_path = pdb_index_file_path
        self.template_max_evalue = template_max_evalue
        self.template_min_crop_overlap = template_min_crop_overlap
        self.template_limit_hits = template_limit_hits

        # load MSA/structure hit database index
        # try to read new format (pyarrow dataframe) first
        try:
            self.index = pd.read_feather(self.index_file_path)
        except pyarrow.ArrowInvalid:
            # legacy mode - transform old index into new format
            with open(self.index_file_path) as f:
                index_raw = json.load(f)

            # remove potential invalid entries without MSA, turn into list for item-based access
            self.index = transform_index({
                id_: file_info
                for id_, file_info in index_raw.items()
                if any([file_name.endswith(".a3m") or file_name.endswith(".a3m.gz")
                        for file_name, _, _ in file_info.get("files", [])])
            })

        # add weights from cluster size count, otherwise set all samples to same weight
        if "num_clusters" in self.index.columns:
            self.index.loc[:, "weight"] = 1.0 / self.index.loc[:, "num_clusters"]
        else:
            self.index.loc[:, "weight"] = 1.0

        # allow to dynamically filter dataset
        if index_table_query is not None:
            self.index = self.index.query(index_table_query).reset_index(drop=True)

        # extract unique database files for opening
        base_path = Path(index_file_path).parent
        self.unique_dbs = {
            (db, base_path) for db in self.index.db.unique()
        }

        # if available, also load PDB structure database
        if self.pdb_index_file_path is not None:
            with open(self.pdb_index_file_path) as f:
                # keep as dictionary (unlike MSA/HHR where we need index-based access)
                self.index_pdb = json.load(f)

            base_path_pdb = Path(self.pdb_index_file_path).parent
            unique_pdb_dbs = {
                (file_info.get("db"), base_path_pdb) for id_, file_info in self.index_pdb.items()
            }

            # make sure database names do not overlap
            assert len(
                {name for name, _ in unique_pdb_dbs} & {name for name, _ in self.unique_dbs}
            ) == 0, "Database names of MSA/HHR and PDB database overlap"

            # merge into overall database list
            self.unique_dbs |= unique_pdb_dbs
        else:
            if self.use_templates:
                logger.info(
                    "Warning: use_templates=True without defined pdb_index_file_path will fetch structures from RCSB"
                )
            self.index_pdb = None

        # do not initialize filehandles here, as object will be copied via pickling for num_workers > 0
        self.db_to_filehandle = None

    def __del__(self):
        # close all open database files
        if self.db_to_filehandle is not None:
            for db_name, filehandle in self.db_to_filehandle.items():
                filehandle.close()

    def __len__(self):
        return len(self.index)

    def open_files(self):
        if self.db_to_filehandle is None:
            self.db_to_filehandle = {
                db: open(
                    base_path.joinpath(db), "rb"
                ) for db, base_path in self.unique_dbs
            }

    def read_from_db(self, db_name, file_start, file_size, decompress=False, decode=True):
        # open files persistently, if not already done
        if self.db_to_filehandle is None:
            self.open_files()

        fh = self.db_to_filehandle[db_name]
        fh.seek(file_start)
        contents = fh.read(file_size)

        if decompress:
            contents = gzip.decompress(contents)

        if decode:
            contents = contents.decode("utf-8")

        return contents

    def __getitem__(self, idx):
        """
        Retrieve datapoint from merged database files
        """
        # verify access is valid
        if idx < 0 or idx >= len(self.index):
            raise IndexError("Dataset index out of bounds")

        # retrieve file info from index
        row = self.index.iloc[idx]
        id_ = row["id"]
        file_db = row["db"]

        # break out into MSA and structure hit info
        msa_info, hhr_info = None, None

        alignment_name = row.get("alignment_name")
        if alignment_name is not None:
            msa_info = (
                row["alignment_start"], row["alignment_size"], alignment_name.endswith(".gz")
            )

        template_name = row.get("template_name")
        if template_name is not None:
            hhr_info = (
                row["template_start"], row["template_size"], template_name.endswith(".gz")
            )

        msa = parse_a3m(
            self.read_from_db(file_db, *msa_info, decode=True)
        )

        target_seq = msa.sequences[0]
        msa_matrix, deletion_matrix = extract_msa_feature_data(msa)
        num_seqs, num_pos = msa_matrix.shape

        # apply random crop in primary sequence if longer than crop size
        if num_pos > self.crop_size:
            crop_start = int(torch.randint(0, num_pos - self.crop_size + 1, size=(1,))[0])
        else:
            crop_start = 0

        # first, crop MSA but do not remove sequences yet - this full alignment may be used to sample
        # sequences for decoding
        crop_end = crop_start + self.crop_size
        msa_matrix_cropped = msa_matrix[:, crop_start:crop_end]
        deletion_matrix_cropped = deletion_matrix[:, crop_start:crop_end]

        # sample sequences from MSA but do not do this at random (as in. AF3 supplement Section 2.2) as this might just
        # leave us with MSA crops full of gaps and nothing to learn/reconstruct; instead, rather truncate the MSA
        # to keep the sequences that are the most similar to the target (assuming ordered MSA),
        # which helps us to learn low-variation settings
        if self.truncate_encoder_msa:
            k = torch.randint(
                1, min(len(msa_matrix_cropped), self.max_seqs_per_msa) + 1, size=(1,)
            )
            k = int(k[0])

            msa_matrix_encoder = msa_matrix_cropped[:k]
            deletion_matrix_encoder = deletion_matrix_cropped[:k]
        else:
            # otherwise simply keep fixed upper number of sequences; most similar sequences to target assumed at top
            msa_matrix_encoder = msa_matrix_cropped[:self.max_seqs_per_msa]
            deletion_matrix_encoder = deletion_matrix_cropped[:self.max_seqs_per_msa]

        # featurize MSA
        f = prepare_msa_features(msa_matrix_encoder, deletion_matrix_encoder)

        # select source MSA for sampling decoder sequences, either from untruncated or truncated alignment
        if self.decoder_seqs_from_full_msa:
            source_msa_matrix_decoder = msa_matrix_cropped
        else:
            source_msa_matrix_decoder = msa_matrix_encoder

        # derive weights for sequence sampling based on number of non-gap positions per sequence
        if self.decoder_weight_by_gaps:
            # count how many actual aa residue positions we have for each sequence in the current crop;
            # use raw counts as weights, WeightedRandomSampler not expecting that weights sum up to 1.
            weights = (
                source_msa_matrix_decoder != AA_TO_INDEX[GAP]
            ).sum(axis=1)

            # do not sample more sequences than valid ones (especially relevant if sampling without replacement,
            # but also with replacement, this may lead to artifacts where single query is sampled over and over
            max_samples = min(
                int((weights > 0).sum()), self.decoder_sample_size
            )
        else:
            # sample from sequences uniformly
            weights = np.ones(
                len(source_msa_matrix_decoder)
            )

            # do not select more samples than there are in MSA
            max_samples = min(
                len(source_msa_matrix_decoder), self.decoder_sample_size
            )

        # sample sequences to decode at random, based on weights from above
        decoder_seqs_idx = list(
            WeightedRandomSampler(
                weights=weights, num_samples=max_samples, replacement=self.decoder_sample_with_replacement
            )
        )

        # extract sampled sequences from MSA and attach to input features
        seqs_to_decode = torch.from_numpy(
            source_msa_matrix_decoder[decoder_seqs_idx]
        )
        f = add_seqs_to_decode(f, seqs_to_decode)

        try:
            _worker_info = get_worker_info()
            _indices = _worker_info.dataset.indices[:5]
        except (AttributeError, KeyError):
            _worker_info = ""
            _indices = ""

        logger.info(
            f"dataset: get idx {idx} ({id_}); "
            f"start position={crop_start}, "
            f"truncate_encoder_msa={self.truncate_encoder_msa}, "
            f"#seqs_encoder={len(msa_matrix_encoder)} out of {len(msa_matrix_cropped)}, "
            f"decoder_seqs_from_full_msa={self.decoder_seqs_from_full_msa}, "
            f"decoder_weight_by_gaps={self.decoder_weight_by_gaps}, "
            f"#seqs_decoder={max_samples}, "
            f"worker: {_worker_info} {_indices}"
        )

        # then, handle 3D structure templates (or return right away if using MSA only, or if no template info
        # is available)
        if not self.use_templates or hhr_info is None:
            return f

        # extract structure hits
        template_hits = parse_hhr(
            self.read_from_db(file_db, *hhr_info, decode=True)
        )

        # filter to hits overlapping with current crop; this will overlap any hits that do not match 100% in sequence
        # or are too short hits too
        if self.index_pdb is not None:
            valid_pdb_ids = set(self.index_pdb)
        else:
            valid_pdb_ids = None

        filtered_template_hits = filter_template_hits(
            template_hits, target_seq, crop_start, crop_end,
            max_evalue=self.template_max_evalue,
            min_crop_coverage=self.template_min_crop_overlap,
            max_hits=self.template_limit_hits,
            valid_pdb_ids=valid_pdb_ids,
        )

        # AF3 supplement: "At training time we choose k random templates out of the available n,
        # k ∼ min(Uniform[0, n], 4)"
        # crop_start = int(torch.randint(0, num_pos - self.crop_size + 1, size=(1,))[0])
        num_template_to_sample = min(
            int(torch.randint(0, len(filtered_template_hits) + 1, size=(1,))[0]),
            self.max_templates
        )

        # print("# templates sampled", len(filtered_template_hits), "->", num_template_to_sample)  # TODO: remove
        if num_template_to_sample == 0:
            return f

        sampled_template_idx = list(
            WeightedRandomSampler(
                weights=np.ones(len(filtered_template_hits)),
                num_samples=num_template_to_sample,
                replacement=False
            )
        )

        # assemble template features
        templates = []
        for idx in sampled_template_idx:
            posmap, pdb_id, pdb_chain, original_idx, _ = filtered_template_hits[idx]

            try:
                if self.index_pdb is not None:
                    # access by key is safe as we filter hits by valid PDB ID
                    file_info = self.index_pdb[pdb_id]
                    _, file_start, file_size = file_info["files"][0]
                    pdb_contents = self.read_from_db(
                        file_info["db"], file_start, file_size, decompress=False, decode=False
                    )

                    with gzip.GzipFile(fileobj=BytesIO(pdb_contents), mode="r") as gzf:
                        chain = PDB(gzf).get_chain(pdb_chain)
                else:
                    # as fallback, load over internet from RCSB
                    try:
                        chain = PDB.from_id(pdb_id).get_chain(pdb_chain)
                    except ResourceError:
                        logger.error(f"Could not load PDB {pdb_id} ({id_} idx={idx})")
                        continue
            except ValueError as e:
                logger.error(
                    f"Error loading PDB {pdb_id} {pdb_chain} ({id_} idx={idx}): {str(e)}"
                )
                continue
            except Exception as e:
                logger.error(
                    f"Unexpected Error: {type(e).__name__} loading PDB {pdb_id} {pdb_chain} ({id_} idx={idx}): {str(e)}"
                )
                continue

            try:
                templates.append(
                    prepare_template_features(chain, posmap, True)
                )
            except ValueError as e:
                logger.error(
                    f"Error processing template {pdb_id} {pdb_chain} ({id_} idx={idx}): {str(e)}"
                )
                continue
            except Exception as e:
                logger.error(
                    f"Unexpected Error: {type(e).__name__} processing template {pdb_id} {pdb_chain} ({id_} idx={idx}): {str(e)}"
                )
                continue

        # check if there are any templates left to add, otherwise return without templates
        if len(templates) > 0:
            return add_template_features(f, templates)
        else:
            return f


class SeqStructDataModule(L.LightningDataModule):
    """
    cf. https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    # TODO: do we need to save dataset state or shuffling + reproducible manual seed enough to avoid train/text mix?
    """
    def __init__(
        self,
        # arguments for SeqStructDataset
        index_file_path: str | PathLike,
        crop_size: int = 256,
        max_seqs_per_msa: int = 16384,
        truncate_encoder_msa: bool = True,
        decoder_seqs_from_full_msa: bool = True,
        decoder_sample_size: int = 64,
        decoder_weight_by_gaps: bool = True,
        decoder_sample_with_replacement: bool = False,
        index_table_query: str | None = None,
        # template-related arguments
        use_templates: bool = False,
        pdb_index_file_path: str | PathLike | None = None,
        max_templates: int = 2,
        template_max_evalue: float | None = 1e-05,
        template_min_crop_overlap: int | None = 64,
        template_limit_hits: int | None = 20,

        # arguments for SeqStructDataModule
        batch_size: int = 2,
        train_data_split: float = 0.9,
        shuffle_training: bool | None = True,
        weight_sequence_families: bool = False,
        data_split_random_seed: int = 42,
        num_workers: int = 4,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        use_stateful_loader: bool = False,
    ):
        """
        # TODO: document params
        """
        super().__init__()

        # parameters for passing into SeqStructDataset
        self.index_file_path = index_file_path
        self.crop_size = crop_size
        self.max_seqs_per_msa = max_seqs_per_msa

        self.truncate_encoder_msa = truncate_encoder_msa
        self.decoder_seqs_from_full_msa = decoder_seqs_from_full_msa
        self.decoder_sample_size = decoder_sample_size
        self.decoder_weight_by_gaps = decoder_weight_by_gaps
        self.decoder_sample_with_replacement = decoder_sample_with_replacement
        self.index_table_query = index_table_query

        self.use_templates = use_templates
        self.pdb_index_file_path = pdb_index_file_path
        self.max_templates = max_templates
        self.template_max_evalue = template_max_evalue
        self.template_min_crop_overlap = template_min_crop_overlap
        self.template_limit_hits = template_limit_hits

        # parameters for this class
        self.batch_size = batch_size
        self.train_data_split = train_data_split
        self.data_split_random_seed = data_split_random_seed
        self.shuffle = shuffle_training
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        if use_stateful_loader:
            self.loader_class = StatefulDataLoader
        else:
            self.loader_class = DataLoader

        self.weight_sequence_families = weight_sequence_families

        if self.weight_sequence_families and self.shuffle is not None:
            raise ValueError("shuffle must be None for weight_sequence_families=True")

        self.full_data = None
        self.train_data = None
        self.val_data = None

        # prepare closure for data collation into batches;
        self.collate_fn = partial(
            batch_features,
            fixed_seq_len=self.crop_size,
            fixed_msa_len=self.max_seqs_per_msa,
            fixed_decoder_msa_len=self.decoder_sample_size,
            fixed_template_num=self.max_templates,
        )

    def setup(self, stage: str):
        """
        # TODO: document params
        """
        # create train/test split
        if stage == "fit":
            self.full_data = SeqStructDataset(
                index_file_path=self.index_file_path,
                crop_size=self.crop_size,
                max_seqs_per_msa=self.max_seqs_per_msa,
                truncate_encoder_msa=self.truncate_encoder_msa,
                decoder_seqs_from_full_msa=self.decoder_seqs_from_full_msa,
                decoder_sample_size=self.decoder_sample_size,
                decoder_weight_by_gaps=self.decoder_weight_by_gaps,
                decoder_sample_with_replacement=self.decoder_sample_with_replacement,
                index_table_query=self.index_table_query,
                use_templates=self.use_templates,
                pdb_index_file_path=self.pdb_index_file_path,
                max_templates=self.max_templates,
                template_max_evalue=self.template_max_evalue,
                template_min_crop_overlap=self.template_min_crop_overlap,
                template_limit_hits=self.template_limit_hits,
            )

            self.train_data, self.val_data = random_split(
                self.full_data,
                [self.train_data_split, 1.0 - self.train_data_split],
                generator=torch.Generator().manual_seed(self.data_split_random_seed)
            )
        else:
            raise NotImplementedError("Only fitting stage supported by DataModule for now")

    def train_dataloader(self):
        if self.weight_sequence_families:
            training_samples = self.train_data.dataset.index.iloc[self.train_data.indices]
            weights = training_samples.weight.values
            assert len(weights) == len(self.train_data)
            sampler = WeightedRandomSampler(weights, len(weights))
        else:
            sampler = None

        return self.loader_class(
            self.train_data,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=self.shuffle,
            sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return self.loader_class(
            self.val_data,
            batch_size=self.batch_size,
            drop_last=True,
            # shuffle=self.shuffle,  # discouraged to shuffle in validation set, use default/False
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        raise NotImplementedError("Test data handling not implemented")

    def predict_dataloader(self):
        raise NotImplementedError("Prediction data handling not implemented")

    # reenable this if lightning cannot handle NamedTuple as tuple subclass
    # def transfer_batch_to_device(self, batch, device, dataloader_idx):
    #     return InputFeatureBatch(
    #         # target sequence features
    #         pos_mask=batch.pos_mask.to(device),
    #         token_index=batch.token_index.to(device),
    #         restype=batch.restype.to(device),
    #
    #         # alignment features
    #         msa=batch.msa.to(device),
    #         msa_mask=batch.msa_mask.to(device),
    #         has_deletion=batch.has_deletion.to(device),
    #         deletion_value=batch.deletion_value.to(device),
    #         deletion_mean=batch.deletion_mean.to(device),
    #         profile=batch.profile.to(device),
    #     )

    # # from https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    # def state_dict(self):
    #     # track whatever you want here
    #     state = {"current_train_batch_index": self.current_train_batch_index}
    #     return state
    #
    # # from https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    # def load_state_dict(self, state_dict):
    #     # restore the state based on what you tracked in (def state_dict)
    #     self.current_train_batch_index = state_dict["current_train_batch_index"]

