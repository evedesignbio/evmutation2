"""
PDB structure handling from EVcouplings package
"""

from collections import OrderedDict
from collections.abc import Iterable
import gzip
from io import BytesIO
from os import path
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import requests
import msgpack
from Bio.PDB.binary_cif import _decode

PDB_BCIF_DOWNLOAD_URL = "https://models.rcsb.org/{pdb_id}.bcif.gz"


# amino acid one-letter code to three-letter code
AA1_to_AA3 = {
    "A": "ALA",
    "B": "ASX",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "X": "XAA",
    "Y": "TYR",
    "Z": "GLX",
}

# amino acid three-letter code to one-letter code
AA3_to_AA1 = {
    v: k for k, v in AA1_to_AA3.items()
}

class ResourceError(Exception):
    """
    Exception for missing resources (files, URLs, ...)
    """

# Mapping from MMTF secondary structure codes to DSSP symbols
MMTF_DSSP_CODE_MAP = {
    0: "I",  # pi helix
    1: "S",  # bend
    2: "H",  # alpha helix
    3: "E",  # extended
    4: "G",  # 3-10 helix
    5: "B",  # bridge
    6: "T",  # turn
    7: "C",  # coil
    -1: "",  # undefined
}

# Reduction to 3-state secondary structure annotation
DSSP_3_STATE_MAP = {
    "H": "H",
    "G": "H",
    "I": "H",
    "E": "E",
    "B": "E",
    "C": "C",
    "T": "C",
    "S": "C",
}

# format string for PDB ATOM records
PDB_FORMAT = (
    "{atom:<6s}{atom_id:>5} "
    "{atom_name:4s}{alt_loc_ind:1s}{residue_name:<3s} "
    "{chain_id:1s}{residue_id:>4}{ins_code:1}   "
    "{x_coord:>8.3f}{y_coord:>8.3f}{z_coord:>8.3f}"
    "{occupancy:>6.2f}{temp_factor:>6.2f}          "
    "{element_symbol:>2}{charge:>2}"
)


class Chain:
    """
    Container for PDB chain residue and coordinate information
    """

    def __init__(self, residues, coords):
        """
        Create new PDB chain, indexed by residue coordinate
        indeces

        Parameters
        ----------
        residues : pandas.DataFrame
            List of residues (as computed by PDB.get_chain())
        coords : pandas.DataFrame
            List of atom coordinates (as computed by
            PDB.get_chain())
        """
        self.residues = residues
        self.coords = coords

    def _update_ids(self, ids):
        """
        Update residue identifiers, and remove any
        residue that does not have new id. Also
        removes corresponding atom coordinates.

        Parameters
        ----------
        ids : pandas.Series (or list-like)
            New identifiers to be assigned to
            residue table. Has to be of same length
            in same order as residue table.

        Returns
        -------
        Chain
            Chain with new residue identifiers
        """
        residues = self.residues.copy()
        residues.loc[:, "id"] = ids.copy()
        residues = residues.dropna(subset=["id"])

        # drop coordinates of residues that were not kept
        coords = self.coords.loc[
            self.coords.residue_index.isin(residues.index)
        ]

        # reset atom index to consecutive numbers from 0
        coords = coords.reset_index(drop=True)

        return Chain(residues, coords)

    def to_seqres(self):
        """
        Return copy of chain with main index set to
        SEQRES numbering. Residues that do not have
        a SEQRES id will be dropped.

        Returns
        -------
        Chain
            Chain with seqres IDs as main index
        """
        return self._update_ids(
            self.residues.loc[:, "seqres_id"]
        )

    def filter_atoms(self, atom_name="CA"):
        """
        Filter coordinates of chain, e.g. to
        compute C_alpha-C_alpha distances

        Parameters
        ----------
        atom_name : str or list-like, optional (default: "CA")
            Name(s) of atoms to keep

        Returns
        -------
        Chain
            Chain containing only filtered atoms (and those
            residues that have such an atom)
        """
        if isinstance(atom_name, str):
            sel = self.coords.atom_name == atom_name
        else:
            sel = self.coords.atom_name.isin(atom_name)

        # update dataframe to rows having the right atom(s)
        coords = self.coords.loc[sel].copy()

        # reset atom index to consecutive numbers from 0
        coords = coords.reset_index(drop=True)

        # if there are residues without any atoms, remove
        # the entire residue
        residues = self.residues.loc[
            self.residues.index.isin(coords.residue_index)
        ].copy()

        return Chain(residues, coords)

    def filter_positions(self, positions):
        """
        Select a subset of positions from the chain

        Parameters
        ----------
        positions : list-like
            Set of residues that will be kept

        Returns
        -------
        Chain
            Chain containing only the selected residues
        """
        # map all positions to be strings
        positions = [str(p) for p in positions]

        residues = self.residues.loc[
            self.residues.id.isin(positions)
        ].copy()

        # drop coordinates of residues that were not kept
        coords = self.coords.loc[
            self.coords.residue_index.isin(residues.index)
        ]

        # reset atom index to consecutive numbers from 0
        coords = coords.reset_index(drop=True)

        return Chain(residues, coords)

    def remap(self, mapping, source_id="seqres_id"):
        """
        Remap chain into different numbering scheme
        (e.g. from seqres to uniprot numbering)

        Parameters
        ----------
        mapping : dict
            Mapping of residue identifiers from
            source_id (current main index of PDB chain)
            to new identifiers.

            mapping may either be:

            1. dict(str -> str) to map individual residue
               IDs. Keys and values of dictionary will be
               typecast to string before the mapping, so it
               is possible to pass in integer values too
               (if the source or target IDs are numbers)

            2. dict((int, int) -> (int, int)) to map ranges
               of numbers to ranges of numbers. This should
               typically be only used with RESSEQ or UniProt
               numbering. End index or range is \*inclusive*\
               Note that residue IDs in the end will still
               be handled as strings when mapping.

        source_id: {"seqres_id", "coord_id", "id"}, optional (default: "seqres_id")
            Residue identifier in chain to map \*from*\
            (will be used as key to access mapping)

        Returns
        -------
        Chain
            Chain with remapped numbering ("id" column
            in residues DataFrame)
        """
        # get one key to test which type of mapping we have
        # (range-based, or individual residues)
        test_key = next(iter(mapping.keys()))

        # test for range-based mapping
        if isinstance(test_key, Iterable) and not isinstance(test_key, str):
            # build up inidividual residue mapping
            final_mapping = {}
            for (source_start, source_end), (target_start, target_end) in mapping.items():
                source = map(
                    str, range(source_start, source_end + 1)
                )

                target = map(
                    str, range(target_start, target_end + 1)
                )

                final_mapping.update(
                    dict(zip(source, target))
                )
        else:
            # individual residue mapping, make sure all strings
            final_mapping = {
                str(s): str(t) for (s, t) in mapping.items()
            }

        # remap identifiers using mapping
        ids = self.residues.loc[:, source_id].map(
            final_mapping, na_action="ignore"
        )

        # create remapped chain
        return self._update_ids(ids)

    def to_file(self, fileobj, chain_id="A", end=True, first_atom_id=1):
        """
        Write chain to a file in PDB format (mmCIF not yet
        supported).

        Note that PDB files written this function may not
        be 100% compliant with the PDB format standards,
        in particular:

        * some HETATM records may turn into ATOM records
          when starting from an mmtf file, if the record
          has a one-letter code (such as MSE / M).

        * code does not print TER record at the end of
          a peptide chain

        Parameters
        ----------
        fileobj : file-like object
            Write to this file handle
        chain_id : str, optional (default: "A")
            Assign this chain name in file (allows to redefine
            chain name from whatever chain was originally)
        end : bool, optional (default: True)
            Print "END" record after chain (signals end of PDB file)
        first_atom_id : int, optional (default: 1)
            Renumber atoms to start with this index
            (set to None to keep default indices)

        Raises
        ------
        ValueError
            If atom or residue numbers are too wide and cannot
            be written to old fixed-column PDB file format
        """
        # maximum number of atoms and residues than can be written to
        # old PDB file format
        OLD_PDB_MAX_ATOM_NUM = 99999
        OLD_PDB_MAX_RESIDUE_NUM = 9999

        # merge residue-level information and atom-level information
        # in one joint table (i.e. the way data is presented in a
        # PDB/mmCIF file)
        x = self.coords.merge(
            self.residues, left_on="residue_index", right_index=True
        )

        # renumber atoms if requested (this helps to be able to
        # write chains from very large structures to old PDB
        # format that wouldn't fit into fixed columns otherwise)
        if first_atom_id is not None:
            if first_atom_id < 1:
                raise ValueError(
                    "First atom index must be > 0"
                )

            # renumber to start at first_atom_id
            x.loc[:, "atom_id"] = np.arange(
                first_atom_id, first_atom_id + len(x)
            ).astype(int)

        # write one atom at a time
        for idx, r in x.iterrows():
            # split residue ID into position and insertion code
            cid = str(r["id"])
            if cid[-1].isalpha():
                coord_id = cid[:-1]
                ins_code = cid[-1]
            else:
                coord_id = cid
                ins_code = ""

            if int(coord_id) > OLD_PDB_MAX_RESIDUE_NUM:
                raise ValueError(
                    "Residue index is too wide for old PDB format: "
                    "{} (maximum is {})".format(coord_id, OLD_PDB_MAX_RESIDUE_NUM)
                )

            if int(r["atom_id"]) > OLD_PDB_MAX_ATOM_NUM:
                raise ValueError(
                    "Atom index is too wide for old PDB format: "
                    "{} (maximum is {})".format(r["atom_id"], OLD_PDB_MAX_ATOM_NUM)
                )

            # atom element
            element = r["element"].upper()

            # need to split atom name into element and specifier
            # (e.g. beta carbon element:C, specifier:B) so we
            # can correctly justify in the 4-column atom
            # name field: first 2 (right-justified) are
            # element, second 2 (left-justified) are specifier
            src_atom_name = r["atom_name"]

            # to make things more complicated, there are cases like
            # HE21 (CNS) or 1HE2 (PDB) which break if assuming
            # that atom_element == element. In these cases, we
            # just use the full raw string
            if len(src_atom_name) == 4:
                atom_name = src_atom_name
            else:
                atom_element = src_atom_name[0:len(element)]
                atom_spec = src_atom_name[len(element):]
                atom_name = "{:>2s}{:<2s}".format(atom_element, atom_spec)

            # print charge if we have one (optional)
            charge = r["charge"]
            # test < and > to exclude nan values
            if isinstance(charge, int) and (charge < 0 or charge > 0):
                charge_sign = "-" if charge < 0 else "+"
                charge_value = abs(charge)
                charge_str = "{}{}".format(charge_value, charge_sign)
            else:
                charge_str = ""

            # format line and write
            s = PDB_FORMAT.format(
                atom="HETATM" if r["hetatm"] else "ATOM",
                atom_id=r["atom_id"],
                atom_name=atom_name,
                alt_loc_ind=r["alt_loc"],
                residue_name=r["three_letter_code"],
                chain_id=chain_id,
                residue_id=coord_id,
                ins_code=ins_code,
                x_coord=r["x"],
                y_coord=r["y"],
                z_coord=r["z"],
                occupancy=r["occupancy"],
                temp_factor=r["b_factor"],
                element_symbol=element,
                charge=charge_str,
            )
            fileobj.write(s + "\n")

        if end:
            fileobj.write("END" + 77 * " " + "\n")


class PDB:
    """
    Holds PDB structure from binaryCIF format; supersedes original PDB class based
    on MMTF format (renamed to MmtfPDB, cf. below) due to MMTF retirement in 2024
    """

    def __init__(self, filehandle, keep_full_data=False):
        """
        Initialize by parsing binaryCIF from open filehandle.
        Recommended to use from_file() and from_id() class methods to create object.

        Column extraction and decoding based on https://github.com/biopython/biopython/blob/master/Bio/PDB/binary_cif.py

        Parameters
        ----------
        filehandle: file-like object
            Open filehandle (binary) from which to read binaryCIF data
        keep_full_data: bool (default: False)
            Associate raw extracted data with object
        """
        # unpack information in bCIF file
        raw_data = msgpack.unpack(
            filehandle, use_list=True
        )

        data = {
            f"{category['name']}.{column['name']}": column
            for block in raw_data["dataBlocks"] for category in block["categories"] for column in category["columns"]
        }

        ATOM_TARGET_COLS = {
            "_atom_site.pdbx_PDB_model_num": "model_number",
            "_atom_site.group_PDB": "record_type",  # ATOM, HETATM etc.

            # atom IDs and types
            "_atom_site.id": "id",  # x
            "_atom_site.type_symbol": "type_symbol",  # x
            "_atom_site.label_atom_id": "label_atom_id",  # x
            "_atom_site.auth_atom_id": "auth_atom_id",
            "_atom_site.label_alt_id": "label_alt_id",

            # residue/molecule types (three-letter code)
            "_atom_site.label_comp_id": "label_comp_id",  # x
            "_atom_site.auth_comp_id": "auth_comp_id",

            # chain IDs (official, author) and entity IDs
            "_atom_site.label_asym_id": "label_asym_id",  # x
            "_atom_site.auth_asym_id": "auth_asym_id",
            "_atom_site.label_entity_id": "label_entity_id",

            # residue IDs (official and author)
            "_atom_site.label_seq_id": "label_seq_id",
            "_atom_site.auth_seq_id": "auth_seq_id",  # x
            "_atom_site.pdbx_PDB_ins_code": "insertion_code",

            # atom properties
            "_atom_site.Cartn_x": "x",  # x
            "_atom_site.Cartn_y": "y",  # x
            "_atom_site.Cartn_z": "z",  # x
            "_atom_site.occupancy": "occupancy",  # x
            "_atom_site.B_iso_or_equiv": "b_factor",  # x
            "_atom_site.pdbx_formal_charge": "charge",
        }

        # full list of conf types: https://mmcif.wwpdb.org/dictionaries/mmcif_ma.dic/Items/_struct_conf_type.id.html;
        # mapping between file types: https://manpages.debian.org/unstable/dssp/mkdssp.1.en.html
        CONF_TARGET_COLS = {
            "_struct_conf.conf_type_id": "conformation_type",
            "_struct_conf.id": "id",
            # label_asym_id and label_seq_id are sufficient for merging to atom table;
            # do not bother with author IDs here
            "_struct_conf.beg_label_asym_id": "beg_label_asym_id",
            "_struct_conf.beg_label_seq_id": "beg_label_seq_id",
            "_struct_conf.end_label_asym_id": "end_label_asym_id",
            "_struct_conf.end_label_seq_id": "end_label_seq_id",
        }

        SHEET_TARGET_COLS = {
            "_struct_sheet_range.sheet_id": "sheet_id",
            "_struct_sheet_range.id": "id",
            "_struct_sheet_range.beg_label_asym_id": "beg_label_asym_id",
            "_struct_sheet_range.beg_label_seq_id": "beg_label_seq_id",
            "_struct_sheet_range.end_label_asym_id": "end_label_asym_id",
            "_struct_sheet_range.end_label_seq_id": "end_label_seq_id",
        }

        if keep_full_data:
            self.data = data
        else:
            self.data = None

        # decode information into dataframe with BioPython helper method
        self.atom_table = pd.DataFrame({
            name: _decode(data[source_column]) for source_column, name in ATOM_TARGET_COLS.items()
        }).assign(
            # make sure chain identifiers are strings, in some pathologic cases, these come out as numbers
            # (e.g. entry 6swy)
            auth_asym_id=lambda df: df.auth_asym_id.astype(str),
            label_asym_id=lambda df: df.label_asym_id.astype(str),
        )

        # decode information into dataframe with BioPython helper method; note this section may not be
        # present if no helices exist in the structure
        try:
            self.conf_table = pd.DataFrame({
                name: _decode(data[source_column]) for source_column, name in CONF_TARGET_COLS.items()
            }).query(
                # there are a handful of PDB entries that have (probably wrong) secondary structure assignments
                # extending over more than one segment (e.g. 2bp7, 2wjv), drop these rather than raising an error
                "beg_label_asym_id == end_label_asym_id"
            )
        except KeyError:
            self.conf_table = None

        # decode information into dataframe with BioPython helper method; note this section may not be
        # present if no sheets exist in the structure
        try:
            self.sheet_table = pd.DataFrame({
                name: _decode(data[source_column]) for source_column, name in SHEET_TARGET_COLS.items()
            })
        except KeyError:
            self.sheet_table = None

        # create secondary structure table for merging to chain tables
        # (will only contain helix/H and strand/E, coil/C will need to be filled in)
        sse_raw = []
        for sse_type, sse_table, sse_filter in [
            ("H", self.conf_table, "HELX"),
            ("E", self.sheet_table, None),
            # also retrieve beta strands/bridges from conf_table if available
            ("E", self.conf_table, "STRN"),
        ]:
            # skip if secondary structure element not present in PDB file at all
            if sse_table is None:
                continue

            # filter table down to relevant entries for current secondary structure type
            if sse_filter is not None:
                sse_table = sse_table.query(
                    f"conformation_type.str.startswith('{sse_filter}')"
                )

            for _, row in sse_table.iterrows():
                for seq_id in range(row.beg_label_seq_id, row.end_label_seq_id + 1):
                    sse_raw.append({
                        "label_asym_id": row.beg_label_asym_id,
                        "label_seq_id": seq_id,
                        "sec_struct_3state": sse_type,
                    })

        # drop duplicates, there are overlapping helix segment annotations e.g. for PDB 6cup:A:Asp92
        if len(sse_raw) > 0:
            self.secondary_structure = pd.DataFrame(
                sse_raw
            ).drop_duplicates(
                subset=["label_asym_id", "label_seq_id"]
            )
        else:
            self.secondary_structure = None

        # store information about models/chains for quick retrieval and verification;
        # subtract 0 to start numbering consistently to how this was handled with MMTF
        self.models = list(
            sorted(self.atom_table.model_number.unique())
        )

        # model number to auth ID mapping
        self.model_to_chains = self.atom_table[
            ["model_number", "auth_asym_id"]
        ].drop_duplicates().groupby(
            "model_number"
        ).agg(
            lambda s: list(s)
        )["auth_asym_id"].to_dict()

        # model number to asym ID mapping
        self.model_to_asym_ids = self.atom_table[
            ["model_number", "label_asym_id"]
        ].drop_duplicates().groupby(
            "model_number"
        ).agg(
            lambda s: list(s)
        )["label_asym_id"].to_dict()

    @classmethod
    def from_file(cls, filename, keep_full_data=False):
        """
        Initialize structure from binaryCIF file

        inspired by https://github.com/biopython/biopython/blob/master/Bio/PDB/binary_cif.py

        Parameters
        ----------
        filename : str
            Path of MMTF file
        keep_full_data: bool (default: False)
            Associate raw extracted data with object

        Returns
        -------
        PDB
            initialized PDB structure
        """
        try:
            with (
                    gzip.open(filename, mode="rb")
                    if filename.lower().endswith(".gz") else open(filename, mode="rb")
            ) as f:
                return cls(f, keep_full_data=keep_full_data)
        except IOError as e:
            raise ResourceError(
                "Could not open file {}".format(filename)
            ) from e

    @classmethod
    def from_id(cls, pdb_id, keep_full_data=False):
        """
        Initialize structure by PDB ID (fetches structure from RCSB servers)

        Parameters
        ----------
        pdb_id : str
            PDB identifier (e.g. 1hzx)
        keep_full_data: bool (default: False)
            Associate raw extracted data with object

        Returns
        -------
        PDB
            initialized PDB structure
        """
        # TODO: add proper retry logic and timeouts
        # TODO: add better exception handling
        try:
            r = requests.get(
                PDB_BCIF_DOWNLOAD_URL.format(pdb_id=pdb_id.lower())
            )
        except requests.exceptions.RequestException as e:
            raise ResourceError(
                "Error fetching bCIF data for {}".format(pdb_id)
            ) from e

        if not r.ok:
            raise ResourceError(
                "Did not receive valid response fetching {}".format(pdb_id)
            )

        with gzip.GzipFile(fileobj=BytesIO(r.content), mode="r") as f:
            return cls(f, keep_full_data=keep_full_data)

    def get_chain(self, chain, model=0, is_author_id=True):
        """
        Extract residue information and atom coordinates
        for a given chain in PDB structure

        Parameters
        ----------
        chain : str
            ID of chain to be extracted (e.g. "A")
        model : int, optional (default: 0)
            *Index* of model to be extracted, starting counting at 0. Note that for backwards
            compatibility, this is *not* the actual PDB model identifier but indexes the model
            identifiers in self.models, i.e. model must be >= 0 and < len(self.models)
        is_author_id : bool (default: True)
            If true, interpret chain parameter as author chain identifier;
            if false, interpret as label_asym_id

        Returns
        -------
        Chain
            Chain object containing DataFrames listing residues
            and atom coordinates
        """
        # check if valid model was requested
        if not 0 <= model < len(self.models):
            raise ValueError(
                f"Invalid model index, valid options: {','.join(map(str, range(len(self.models))))}"
            )

        # map model index to model number/identifier
        model_number = self.models[model]

        # check if valid chain was requested
        if ((is_author_id and chain not in self.model_to_chains[model_number]) or
                (not is_author_id and chain not in self.model_to_asym_ids[model_number])):
            raise ValueError(
                "Invalid chain selection, check self.model_to_chains / self.model_to_asym_ids for options"
            )

        if is_author_id:
            chain_field = "auth_asym_id"
        else:
            chain_field = "label_asym_id"

        # filter atom table to model + chain selection
        atoms = self.atom_table.query(
            f"model_number == @model_number and {chain_field} == @chain"
        ).assign(
            # create coordinate ID from author residue ID + insertion code
            # (this should be unique and circumvents issues from 0 seqres values if selecting based on author chain ID)
            coord_id=lambda df: df.auth_seq_id.astype(str) + df.insertion_code,
            seqres_id=lambda df: df.label_seq_id.astype(str).replace("0", pd.NA).replace("", pd.NA),
            one_letter_code=lambda df: df.label_comp_id.map(AA3_to_AA1, na_action="ignore"),
            # id=lambda df: df.coord_id,  # note this was wrong - only should do this on residue level
            # note that MSE will now be labeled as HETATM, which was not the case with MMTF
            hetatm=lambda df: df.record_type == "HETATM",
        ).reset_index(
            drop=True
        )

        # create residue table by de-duplicating atoms
        res = atoms.drop_duplicates(
            subset=["coord_id"]
        ).assign(
            id=lambda df: df.coord_id
        ).reset_index(
            drop=True
        )
        res.index.name = "residue_index"

        # merge secondary structure information (left outer join as coil is missing from table)
        if self.secondary_structure is not None:
            res_sse = res.merge(
                self.secondary_structure,
                on=("label_seq_id", "label_asym_id"),
                how="left"
            )
        else:
            # initialize to pd.NA instead of np.nan or warning about assigning str to float64 column appears
            res_sse = res.assign(
                sec_struct_3state=pd.NA
            )

        res_sse.loc[
            res_sse.sec_struct_3state.isnull() & res_sse.seqres_id.notnull(),
            "sec_struct_3state"
        ] = "C"

        RES_RENAME_MAP = {
            "id": "id",
            "seqres_id": "seqres_id",
            "coord_id": "coord_id",
            "one_letter_code": "one_letter_code",
            "label_comp_id": "three_letter_code",
            "auth_asym_id": "chain_id",
            "label_asym_id": "asym_id",  # new
            "label_entity_id": "entity_id",
            "sec_struct_3state": "sec_struct_3state",
            "hetatm": "hetatm",
        }

        res_final = res_sse.loc[
                    :, list(RES_RENAME_MAP)
                    ].rename(
            columns=RES_RENAME_MAP
        )

        # not included in new version: alt_loc
        ATOM_RENAME_MAP = {
            "residue_index": "residue_index",
            "id": "atom_id",
            "label_atom_id": "atom_name",
            "type_symbol": "element",
            "charge": "charge",
            "x": "x",
            "y": "y",
            "z": "z",
            "occupancy": "occupancy",
            "b_factor": "b_factor",
            "label_alt_id": "alt_loc",
        }

        # add information about residue index to atoms
        atoms_with_residue_idx = atoms.merge(
            res.reset_index()[["coord_id", "residue_index"]],
            on="coord_id"
        ).loc[:, list(ATOM_RENAME_MAP)].rename(
            columns=ATOM_RENAME_MAP
        )
        assert len(atoms_with_residue_idx) == len(atoms)

        return Chain(res_final, atoms_with_residue_idx)
