import warnings

from allennlp import predictors
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from allennlp_semparse import predictors as semparse_predictors


class PretrainedModel:
    """
    A pretrained model is determined by both an archive file
    (representing the trained model)
    and a choice of predictor.
    """

    def __init__(self, archive_file: str, predictor_name: str) -> None:
        self.archive_file = archive_file
        self.predictor_name = predictor_name

    def predictor(self) -> Predictor:
        archive = load_archive(self.archive_file)
        return Predictor.from_archive(archive, self.predictor_name)


# TODO(Mark): Figure out a way to make PretrainedModel generic on Predictor, so we can remove these type ignores.

# Core AllenNLP models


def srl_with_elmo_luheng_2018() -> predictors.SemanticRoleLabelerPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        model = PretrainedModel(
            "https://allennlp.s3.amazonaws.com/models/srl-model-2018.05.25.tar.gz",
            "semantic-role-labeling",
        )
        return model.predictor()  # type: ignore


def bert_srl_shi_2019() -> predictors.SemanticRoleLabelerPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        model = PretrainedModel(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz",
            "semantic-role-labeling",
        )
        return model.predictor()  # type: ignore


def bidirectional_attention_flow_seo_2017() -> predictors.BidafPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        model = PretrainedModel(
            "https://allennlp.s3.amazonaws.com/models/bidaf-model-2017.09.15-charpad.tar.gz",
            "machine-comprehension",
        )
        return model.predictor()  # type: ignore


def naqanet_dua_2019() -> predictors.BidafPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        model = PretrainedModel(
            "https://allennlp.s3.amazonaws.com/models/naqanet-2019.04.29-fixed-weight-names.tar.gz",
            "machine-comprehension",
        )
        return model.predictor()  # type: ignore


def open_information_extraction_stanovsky_2018() -> predictors.OpenIePredictor:
    model = PretrainedModel(
        "https://allennlp.s3.amazonaws.com/models/openie-model.2018-08-20.tar.gz",
        "open-information-extraction",
    )
    return model.predictor()  # type: ignore


def decomposable_attention_with_elmo_parikh_2017() -> predictors.DecomposableAttentionPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        model = PretrainedModel(
            "https://allennlp.s3.amazonaws.com/models/decomposable-attention-elmo-2018.02.19.tar.gz",
            "textual-entailment",
        )
        return model.predictor()  # type: ignore


def neural_coreference_resolution_lee_2017() -> predictors.CorefPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        model = PretrainedModel(
            "https://allennlp.s3.amazonaws.com/models/coref-model-2018.02.05.tar.gz",
            "coreference-resolution",
        )
        predictor = model.predictor()

        predictor._dataset_reader._token_indexers[  # type: ignore
            "token_characters"
        ]._min_padding_length = 5
        return predictor  # type: ignore


def named_entity_recognition_with_elmo_peters_2018() -> predictors.SentenceTaggerPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        model = PretrainedModel(
            "https://allennlp.s3.amazonaws.com/models/ner-model-2018.12.18.tar.gz",
            "sentence-tagger",
        )
        predictor = model.predictor()

        predictor._dataset_reader._token_indexers[  # type: ignore
            "token_characters"
        ]._min_padding_length = 3
        return predictor  # type: ignore


def fine_grained_named_entity_recognition_with_elmo_peters_2018() -> predictors.SentenceTaggerPredictor:
    model = PretrainedModel(
        "https://allennlp.s3.amazonaws.com/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz",
        "sentence-tagger",
    )
    predictor = model.predictor()

    predictor._dataset_reader._token_indexers[  # type: ignore
        "token_characters"
    ]._min_padding_length = 3
    return predictor  # type: ignore


def span_based_constituency_parsing_with_elmo_joshi_2018() -> predictors.ConstituencyParserPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        model = PretrainedModel(
            "https://allennlp.s3.amazonaws.com/models/elmo-constituency-parser-2018.03.14.tar.gz",
            "constituency-parser",
        )
        return model.predictor()  # type: ignore


def biaffine_parser_stanford_dependencies_todzat_2017() -> predictors.BiaffineDependencyParserPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        model = PretrainedModel(
            "https://allennlp.s3.amazonaws.com/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz",
            "biaffine-dependency-parser",
        )
        return model.predictor()  # type: ignore


def biaffine_parser_universal_dependencies_todzat_2017() -> predictors.BiaffineDependencyParserPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        model = PretrainedModel(
            "https://allennlp.s3.amazonaws.com/models/biaffine-dependency-parser-ud-2018.08.23.tar.gz",
            "biaffine-dependency-parser",
        )
        return model.predictor()  # type: ignore


def esim_nli_with_elmo_chen_2017() -> predictors.DecomposableAttentionPredictor:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        model = PretrainedModel(
            "https://allennlp.s3.amazonaws.com/models/esim-elmo-2018.05.17.tar.gz",
            "textual-entailment",
        )
        return model.predictor()  # type: ignore


# AllenNLP Semparse models


def wikitables_parser_dasigi_2019() -> semparse_predictors.WikiTablesParserPredictor:
    model = PretrainedModel(
        "https://storage.googleapis.com/allennlp-public-models/wikitables-model-2019.07.29.tar.gz",
        "wikitables-parser",
    )
    return model.predictor()


def nlvr_parser_dasigi_2019() -> semparse_predictors.NlvrParserPredictor:
    model = PretrainedModel(
        "https://storage.googleapis.com/allennlp-public-models/nlvr-erm-model-2018-12-18-rule-vocabulary-updated.tar.gz",
        "nlvr-parser",
    )
    return model.predictor()


# TODO(brendanr): It's not clear from https://demo.allennlp.org/atis-parser
# whether there's a paper associated with this model. Verify this.
def atis_parser() -> semparse_predictors.AtisParserPredictor:
    model = PretrainedModel(
        "https://storage.googleapis.com/allennlp-public-models/atis-parser-2018.11.10.tar.gz",
        "atis-parser",
    )
    return model.predictor()


def quarel_parser_tafjord_2019() -> semparse_predictors.QuarelParserPredictor:
    model = PretrainedModel(
        "https://storage.googleapis.com/allennlp-public-models/quarel-parser-zero-2018.12.20.tar.gz",
        "quarel-parser",
    )
    return model.predictor()
