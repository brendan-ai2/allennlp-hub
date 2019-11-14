import warnings

from allennlp import predictors
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from allennlp_semparse import predictors as semparse_predictors

# Ensure models are registered.
import allennlp_semparse.models


def _load_predictor(archive_file: str, predictor_name: str) -> Predictor:
    """
    Helper to load the desired predictor from the given archive.
    """
    archive = load_archive(archive_file)
    return Predictor.from_archive(archive, predictor_name)


# AllenNLP Semparse models


def wikitables_parser_dasigi_2019() -> semparse_predictors.WikiTablesParserPredictor:
    predictor = _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/wikitables-model-2019.07.29.tar.gz",
        "wikitables-parser",
    )
    return predictor


def nlvr_parser_dasigi_2019() -> semparse_predictors.NlvrParserPredictor:
    predictor = _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/nlvr-erm-model-2018-12-18-rule-vocabulary-updated.tar.gz",
        "nlvr-parser",
    )
    return predictor


def atis_parser_lin_2019() -> semparse_predictors.AtisParserPredictor:
    predictor = _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/atis-parser-2018.11.10.tar.gz",
        "atis-parser",
    )
    return predictor


def quarel_parser_tafjord_2019() -> semparse_predictors.QuarelParserPredictor:
    predictor = _load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/quarel-parser-zero-2018.12.20.tar.gz",
        "quarel-parser",
    )
    return predictor
