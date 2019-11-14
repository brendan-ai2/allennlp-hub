import warnings

from allennlp import predictors
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from allennlp_semparse import predictors as semparse_predictors

# Ensure models are registered.
import allennlp.models
import allennlp_semparse.models


def load_predictor(archive_file: str, predictor_name: str) -> Predictor:
    """
    Helper to load the desired predictor from the given archive.
    """
    archive = load_archive(archive_file)
    return Predictor.from_archive(archive, predictor_name)


# Core AllenNLP models


def srl_with_elmo_luheng_2018() -> predictors.SemanticRoleLabelerPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        predictor = load_predictor(
            "https://allennlp.s3.amazonaws.com/models/srl-model-2018.05.25.tar.gz",
            "semantic-role-labeling",
        )
        return predictor


def bert_srl_shi_2019() -> predictors.SemanticRoleLabelerPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        predictor = load_predictor(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz",
            "semantic-role-labeling",
        )
        return predictor


def bidirectional_attention_flow_seo_2017() -> predictors.BidafPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        predictor = load_predictor(
            "https://allennlp.s3.amazonaws.com/models/bidaf-model-2017.09.15-charpad.tar.gz",
            "machine-comprehension",
        )
        return predictor


def naqanet_dua_2019() -> predictors.BidafPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        predictor = load_predictor(
            "https://allennlp.s3.amazonaws.com/models/naqanet-2019.04.29-fixed-weight-names.tar.gz",
            "machine-comprehension",
        )
        return predictor


def open_information_extraction_stanovsky_2018() -> predictors.OpenIePredictor:
    predictor = load_predictor(
        "https://allennlp.s3.amazonaws.com/models/openie-model.2018-08-20.tar.gz",
        "open-information-extraction",
    )
    return predictor


def decomposable_attention_with_elmo_parikh_2017() -> predictors.DecomposableAttentionPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        predictor = load_predictor(
            "https://allennlp.s3.amazonaws.com/models/decomposable-attention-elmo-2018.02.19.tar.gz",
            "textual-entailment",
        )
        return predictor


def neural_coreference_resolution_lee_2017() -> predictors.CorefPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        predictor = load_predictor(
            "https://allennlp.s3.amazonaws.com/models/coref-model-2018.02.05.tar.gz",
            "coreference-resolution",
        )

        predictor._dataset_reader._token_indexers[
            "token_characters"
        ]._min_padding_length = 5
        return predictor


def named_entity_recognition_with_elmo_peters_2018() -> predictors.SentenceTaggerPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        predictor = load_predictor(
            "https://allennlp.s3.amazonaws.com/models/ner-model-2018.12.18.tar.gz",
            "sentence-tagger",
        )

        predictor._dataset_reader._token_indexers[
            "token_characters"
        ]._min_padding_length = 3
        return predictor


def fine_grained_named_entity_recognition_with_elmo_peters_2018() -> predictors.SentenceTaggerPredictor:
    predictor = load_predictor(
        "https://allennlp.s3.amazonaws.com/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz",
        "sentence-tagger",
    )

    predictor._dataset_reader._token_indexers[
        "token_characters"
    ]._min_padding_length = 3
    return predictor


def span_based_constituency_parsing_with_elmo_joshi_2018() -> predictors.ConstituencyParserPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        predictor = load_predictor(
            "https://allennlp.s3.amazonaws.com/models/elmo-constituency-parser-2018.03.14.tar.gz",
            "constituency-parser",
        )
        return predictor


def biaffine_parser_stanford_dependencies_todzat_2017() -> predictors.BiaffineDependencyParserPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        predictor = load_predictor(
            "https://allennlp.s3.amazonaws.com/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz",
            "biaffine-dependency-parser",
        )
        return predictor


def biaffine_parser_universal_dependencies_todzat_2017() -> predictors.BiaffineDependencyParserPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        predictor = load_predictor(
            "https://allennlp.s3.amazonaws.com/models/biaffine-dependency-parser-ud-2018.08.23.tar.gz",
            "biaffine-dependency-parser",
        )
        return predictor


def esim_nli_with_elmo_chen_2017() -> predictors.DecomposableAttentionPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        predictor = load_predictor(
            "https://allennlp.s3.amazonaws.com/models/esim-elmo-2018.05.17.tar.gz",
            "textual-entailment",
        )
        return predictor


# AllenNLP Semparse models


def wikitables_parser_dasigi_2019() -> semparse_predictors.WikiTablesParserPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        predictor = load_predictor(
            "https://storage.googleapis.com/allennlp-public-models/wikitables-model-2019.07.29.tar.gz",
            "wikitables-parser",
        )
        return predictor


def nlvr_parser_dasigi_2019() -> semparse_predictors.NlvrParserPredictor:
    predictor = load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/nlvr-erm-model-2018-12-18-rule-vocabulary-updated.tar.gz",
        "nlvr-parser",
    )
    return predictor


# TODO(brendanr): It's not clear from https://demo.allennlp.org/atis-parser
# whether there's a paper associated with this model. Verify this.
def atis_parser() -> semparse_predictors.AtisParserPredictor:
    predictor = load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/atis-parser-2018.11.10.tar.gz",
        "atis-parser",
    )
    return predictor


def quarel_parser_tafjord_2019() -> semparse_predictors.QuarelParserPredictor:
    predictor = load_predictor(
        "https://storage.googleapis.com/allennlp-public-models/quarel-parser-zero-2018.12.20.tar.gz",
        "quarel-parser",
    )
    return predictor
