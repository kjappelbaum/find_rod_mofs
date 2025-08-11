# -*- coding: utf-8 -*-

from __future__ import absolute_import

import concurrent.futures
import logging
from functools import partial

import pandas as pd
from pymatgen import Structure
from tqdm.autonotebook import tqdm

from .filter_metal_channels import (get_channel_dimensionality, get_channels, get_metal_metadata,
                                    get_metallic_substructure, get_structure_properties)
from .utils import get_bibliometric_information, get_structure_list, isnotebook
from six.moves import zip

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ScreenMetalChannel:
    # ToDo: maybe change the list default here to something less dangerous
    def __init__(  # pylint:disable=dangerous-default-value
        self,
        structure_list: list,
        njobs: int = 2,
        feature_mode: str = 'cheap',
        other_metals: list = ['Al', 'In', 'Ga'],
    ):
        self.structure_list = structure_list
        self.njobs = njobs
        self.feature_mode = feature_mode
        self.other_metals = other_metals

        if isnotebook():
            logger.warning('the code is untested in notebook enviornments and is likely to not work there')

    @classmethod
    def from_folders(cls, folder_1: str, extension: str = 'cif', njobs=2, feature_mode: str = 'cheap'):
        """Constructor method for a ScreenMetalChannel object"""
        sl_1 = get_structure_list(folder_1, extension)
        return cls(sl_1, njobs=njobs, feature_mode=feature_mode)

    @staticmethod
    def _check_structure(  # pylint:disable=dangerous-default-value
        structure_path: str,
        feature_mode: str = 'cheap',
        other_metals: list = ['Al', 'In', 'Ga'],
    ) -> int:
        """
        Gets number of channels for one cif file.

        Args:
            structure_path (str): path to structure file
            feature_mode (str): mode for feature calculation
            other_metals (list): list of metals that should be considered as metals in addition to
                transition metals and lanthanides

        Returns:

        """
        try:
            s = Structure.from_file(structure_path)
            s1_metall1, s1_nonmetall = get_metallic_substructure(s,
                                                                 return_non_metal_structure=True,
                                                                 other_metals=other_metals)

            num_channels = get_channels(s1_metall1, s1_nonmetall)
            if len(num_channels) > 0:
                # Now also calculate some features that might be interesting for selection or later analysis
                metadata = get_metal_metadata(s1_metall1, num_channels[0][0])
                structure_features = get_structure_properties(s, mode=feature_mode)

                structure_features['num_channels'] = len(num_channels)
                structure_features['channel_indices'] = num_channels
                structure_features.update(metadata)
                planes = get_channel_dimensionality(num_channels, s1_metall1)
                structure_features['planes'] = planes
                # ToDo: make this optional, as this only will be of use if the filenames are CSD refcodes
                # bib = get_bibliometric_information(structure_path)
                # structure_features.update(bib)
                return structure_features
        except Exception:
            return None

        return None

    def run(self):
        channel_list = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.njobs) as executor:

            partial_structure_check = partial(
                ScreenMetalChannel._check_structure,
                feature_mode=self.feature_mode,
                other_metals=self.other_metals,
            )

            i = 0
            for structure, structure_features in tqdm(
                    list(zip(
                        self.structure_list,
                        executor.map(partial_structure_check, self.structure_list),
                    )),
                    total=len(self.structure_list),
            ):
                if structure_features:
                    structure_features['name'] = structure
                    channel_list.append(structure_features)
                    i += 1

                if i % 100 == 0:
                    df = pd.DataFrame(channel_list)
                    df.to_csv('backup_{}.csv'.format(i))

        return pd.DataFrame(channel_list)
