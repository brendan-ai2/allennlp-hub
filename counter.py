from collections import defaultdict

# Replaces every callable in the module with a wrapper that keeps track of how
# often it's called.
def install_hooks(module):
    module.__allennlp_call_counter = defaultdict(int)
    for attr in dir(module):
        original = getattr(module, attr)
        if not callable(original):
            continue
        # Introduce extra scope as Python is a joke of a language.
        def python_sucks(attr, original):
            def replacement(*args, **kwargs):
                module.__allennlp_call_counter[attr] += 1
                return original(*args, **kwargs)
            return replacement
        replacement = python_sucks(attr, original)
        setattr(module, attr, replacement)

# Install the hooks before we load the models on the off-chance somebody caches
# a function pointer during module initialization.
from torch.nn import functional
install_hooks(functional)


import spacy
from allennlp_hub import pretrained


class Counter:
    def test_machine_comprehension(self):
        predictor = pretrained.bidirectional_attention_flow_seo_2017()

        passage = """The Matrix is a 1999 science fiction action film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano. It depicts a dystopian future in which reality as perceived by most humans is actually a simulated reality called "the Matrix", created by sentient machines to subdue the human population, while their bodies' heat and electrical activity are used as an energy source. Computer programmer Neo" learns this truth and is drawn into a rebellion against the machines, which involves other people who have been freed from the "dream world". """
        question = "Who stars in The Matrix?"

        result = predictor.predict_json({"passage": passage, "question": question})

        correct = "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano"

        assert correct == result["best_span_str"]

    def test_semantic_role_labeling(self):
        predictor = pretrained.srl_with_elmo_luheng_2018()

        sentence = "If you liked the music we were playing last night, you will absolutely love what we're playing tomorrow!"

        result = predictor.predict_json({"sentence": sentence})

        assert result["words"] == [
            "If",
            "you",
            "liked",
            "the",
            "music",
            "we",
            "were",
            "playing",
            "last",
            "night",
            ",",
            "you",
            "will",
            "absolutely",
            "love",
            "what",
            "we",
            "'re",
            "playing",
            "tomorrow",
            "!",
        ]

        assert result["verbs"] == [
            {
                "verb": "liked",
                "description": "If [ARG0: you] [V: liked] [ARG1: the music we were playing last night] , you will absolutely love what we 're playing tomorrow !",
                "tags": [
                    "O",
                    "B-ARG0",
                    "B-V",
                    "B-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                ],
            },
            {
                "verb": "playing",
                "description": "If you liked [ARG1: the music] [ARG0: we] were [V: playing] [ARGM-TMP: last] night , you will absolutely love what we 're playing tomorrow !",
                "tags": [
                    "O",
                    "O",
                    "O",
                    "B-ARG1",
                    "I-ARG1",
                    "B-ARG0",
                    "O",
                    "B-V",
                    "B-ARGM-TMP",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                ],
            },
            {
                "verb": "will",
                "description": "If you liked the music we were playing last night , you [V: will] absolutely love what we 're playing tomorrow !",
                "tags": [
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-V",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                ],
            },
            {
                "verb": "love",
                "description": "[ARGM-ADV: If you liked the music we were playing last night] , [ARG0: you] [ARGM-MOD: will] [ARGM-ADV: absolutely] [V: love] [ARG1: what we 're playing tomorrow] !",
                "tags": [
                    "B-ARGM-ADV",
                    "I-ARGM-ADV",
                    "I-ARGM-ADV",
                    "I-ARGM-ADV",
                    "I-ARGM-ADV",
                    "I-ARGM-ADV",
                    "I-ARGM-ADV",
                    "I-ARGM-ADV",
                    "I-ARGM-ADV",
                    "I-ARGM-ADV",
                    "O",
                    "B-ARG0",
                    "B-ARGM-MOD",
                    "B-ARGM-ADV",
                    "B-V",
                    "B-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "O",
                ],
            },
            {
                "verb": "playing",
                "description": "If you liked the music we were playing last night , you will absolutely love [ARG1: what] [ARG0: we] 're [V: playing] [ARGM-TMP: tomorrow] !",
                "tags": [
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-ARG1",
                    "B-ARG0",
                    "O",
                    "B-V",
                    "B-ARGM-TMP",
                    "O",
                ],
            },
        ]

    def test_textual_entailment(self):
        predictor = pretrained.decomposable_attention_with_elmo_parikh_2017()

        result = predictor.predict_json(
            {
                "premise": "An interplanetary spacecraft is in orbit around a gas giant's icy moon.",
                "hypothesis": "The spacecraft has the ability to travel between planets.",
            }
        )

        assert result["label_probs"][0] > 0.7  # entailment

        result = predictor.predict_json(
            {
                "premise": "Two women are wandering along the shore drinking iced tea.",
                "hypothesis": "Two women are sitting on a blanket near some rocks talking about politics.",
            }
        )

        assert result["label_probs"][1] > 0.8  # contradiction

        result = predictor.predict_json(
            {
                "premise": "A large, gray elephant walked beside a herd of zebras.",
                "hypothesis": "The elephant was lost.",
            }
        )

        assert result["label_probs"][2] > 0.6  # neutral

    def test_coreference_resolution(self):
        predictor = pretrained.neural_coreference_resolution_lee_2017()

        document = "We 're not going to skimp on quality , but we are very focused to make next year . The only problem is that some of the fabrics are wearing out - since I was a newbie I skimped on some of the fabric and the poor quality ones are developing holes ."

        result = predictor.predict_json({"document": document})
        print(result)
        assert result["clusters"] == [
            [[0, 0], [10, 10]],
            [[33, 33], [37, 37]],
            [[26, 27], [42, 43]],
        ]
        assert result["document"] == [
            "We",
            "'re",
            "not",
            "going",
            "to",
            "skimp",
            "on",
            "quality",
            ",",
            "but",
            "we",
            "are",
            "very",
            "focused",
            "to",
            "make",
            "next",
            "year",
            ".",
            "The",
            "only",
            "problem",
            "is",
            "that",
            "some",
            "of",
            "the",
            "fabrics",
            "are",
            "wearing",
            "out",
            "-",
            "since",
            "I",
            "was",
            "a",
            "newbie",
            "I",
            "skimped",
            "on",
            "some",
            "of",
            "the",
            "fabric",
            "and",
            "the",
            "poor",
            "quality",
            "ones",
            "are",
            "developing",
            "holes",
            ".",
        ]

    def test_ner(self):
        predictor = pretrained.named_entity_recognition_with_elmo_peters_2018()

        sentence = """Michael Jordan is a professor at Berkeley."""

        result = predictor.predict_json({"sentence": sentence})

        assert result["words"] == [
            "Michael",
            "Jordan",
            "is",
            "a",
            "professor",
            "at",
            "Berkeley",
            ".",
        ]
        assert result["tags"] == ["B-PER", "L-PER", "O", "O", "O", "O", "U-LOC", "O"]

    def test_constituency_parsing(self):
        predictor = pretrained.span_based_constituency_parsing_with_elmo_joshi_2018()

        sentence = """Pierre Vinken died aged 81; immortalised aged 61."""

        result = predictor.predict_json({"sentence": sentence})

        assert result["tokens"] == [
            "Pierre",
            "Vinken",
            "died",
            "aged",
            "81",
            ";",
            "immortalised",
            "aged",
            "61",
            ".",
        ]
        assert (
            result["trees"]
            == "(S (NP (NNP Pierre) (NNP Vinken)) (VP (VP (VBD died) (NP (JJ aged) (CD 81))) (, ;) (VP (VBN immortalised) (S (ADJP (JJ aged) (CD 61))))) (. .))"
        )

    def test_dependency_parsing(self):
        predictor = pretrained.biaffine_parser_stanford_dependencies_todzat_2017()
        sentence = """He ate spaghetti with chopsticks."""
        result = predictor.predict_json({"sentence": sentence})
        # Note that this tree is incorrect. We are checking here that the decoded
        # tree is _actually a tree_ - in greedy decoding versions of the dependency
        # parser, this sentence has multiple heads. This test shouldn't really live here,
        # but it's very difficult to re-create a concrete example of this behaviour without
        # a trained dependency parser.
        assert result["words"] == ["He", "ate", "spaghetti", "with", "chopsticks", "."]
        assert result["pos"] == ["PRP", "VBD", "NNS", "IN", "NNS", "."]
        assert result["predicted_dependencies"] == [
            "nsubj",
            "root",
            "dobj",
            "prep",
            "pobj",
            "punct",
        ]
        assert result["predicted_heads"] == [2, 0, 2, 2, 4, 2]

    def test_wikitables_parser(self):
        predictor = pretrained.wikitables_parser_dasigi_2019()
        table = """#	Event Year	Season	Flag bearer
7	2012	Summer	Ele Opeloge
6	2008	Summer	Ele Opeloge
5	2004	Summer	Uati Maposua
4	2000	Summer	Pauga Lalau
3	1996	Summer	Bob Gasio
2	1988	Summer	Henry Smith
1	1984	Summer	Apelu Ioane"""
        question = "How many years were held in summer?"
        result = predictor.predict_json({"table": table, "question": question})
        # These seem busted... What's up?
        #assert result["answer"] == 7
        #assert (
        #    result["logical_form"][0]
        #    == "(count (filter_in all_rows string_column:season string:summer))"
        #)

    def test_nlvr_parser(self):
        predictor = pretrained.nlvr_parser_dasigi_2019()
        structured_rep = """[
            [
                {"y_loc":13,"type":"square","color":"Yellow","x_loc":13,"size":20},
                {"y_loc":20,"type":"triangle","color":"Yellow","x_loc":44,"size":30},
                {"y_loc":90,"type":"circle","color":"#0099ff","x_loc":52,"size":10}
            ],
            [
                {"y_loc":57,"type":"square","color":"Black","x_loc":17,"size":20},
                {"y_loc":30,"type":"circle","color":"#0099ff","x_loc":76,"size":10},
                {"y_loc":12,"type":"square","color":"Black","x_loc":35,"size":10}
            ],
            [
                {"y_loc":40,"type":"triangle","color":"#0099ff","x_loc":26,"size":20},
                {"y_loc":70,"type":"triangle","color":"Black","x_loc":70,"size":30},
                {"y_loc":19,"type":"square","color":"Black","x_loc":35,"size":10}
            ]
        ]"""
        sentence = "there is exactly one yellow object touching the edge"
        result = predictor.predict_json(
            {"structured_rep": structured_rep, "sentence": sentence}
        )
        assert result["denotations"][0] == ["False"]
        assert (
            result["logical_form"][0]
            == "(object_count_equals (yellow (touch_wall all_objects)) 1)"
        )

    def test_atis_parser(self):
        predictor = pretrained.atis_parser_lin_2019()
        utterance = "give me flights on american airlines from milwaukee to phoenix"
        result = predictor.predict_json({"utterance": utterance})
        predicted_sql_query = """
  (SELECT DISTINCT flight . flight_id
   FROM flight
   WHERE (flight . airline_code = 'AA'
          AND (flight . from_airport IN
                 (SELECT airport_service . airport_code
                  FROM airport_service
                  WHERE airport_service . city_code IN
                      (SELECT city . city_code
                       FROM city
                       WHERE city . city_name = 'MILWAUKEE' ) )
               AND flight . to_airport IN
                 (SELECT airport_service . airport_code
                  FROM airport_service
                  WHERE airport_service . city_code IN
                      (SELECT city . city_code
                       FROM city
                       WHERE city . city_name = 'PHOENIX' ) ))) ) ;"""
        assert result["predicted_sql_query"] == predicted_sql_query

    def test_quarel_parser(self):
        predictor = pretrained.quarel_parser_tafjord_2019()
        question = (
            "In his research, Joe is finding there is a lot more "
            "diabetes in the city than out in the countryside. He "
            "hypothesizes this is because people in _____ consume less "
            "sugar. (A) city (B) countryside"
        )
        qrspec = """[sugar, +diabetes]
[friction, -speed, -smoothness, -distance, +heat]
[speed, -time]
[speed, +distance]
[time, +distance]
[weight, -acceleration]
[strength, +distance]
[strength, +thickness]
[mass, +gravity]
[flexibility, -breakability]
[distance, -loudness, -brightness, -apparentSize]
[exerciseIntensity, +amountSweat]"""
        entitycues = """friction: resistance, traction
speed: velocity, pace, fast, slow, faster, slower, slowly, quickly, rapidly
distance: length, way, far, near, further, longer, shorter, long, short, farther, furthest
heat: temperature, warmth, smoke, hot, hotter, cold, colder
smoothness: slickness, roughness, rough, smooth, rougher, smoother, bumpy, slicker
acceleration:
amountSweat: sweat, sweaty
apparentSize: size, large, small, larger, smaller
breakability: brittleness, brittle, break, solid
brightness: bright, shiny, faint
exerciseIntensity: excercise, run, walk
flexibility: flexible, stiff, rigid
gravity:
loudness: loud, faint, louder, fainter
mass: weight, heavy, light, heavier, lighter, massive
strength: power, strong, weak, stronger, weaker
thickness: thick, thin, thicker, thinner, skinny
time: long, short
weight: mass, heavy, light, heavier, lighter"""
        result = predictor.predict_json(
            {"question": question, "qrspec": qrspec, "entitycues": entitycues}
        )
        assert result["answer"] == "B"
        assert result["explanation"] == [
            {
                "header": "Identified two worlds",
                "content": ['world1 = "city"', 'world2 = "countryside"'],
            },
            {
                "header": "The question is stating",
                "content": ['Diabetes is higher for "city"'],
            },
            {
                "header": "The answer options are stating",
                "content": [
                    'A: Sugar is lower for "city"',
                    'B: Sugar is lower for "countryside"',
                ],
            },
            {
                "header": "Theory used",
                "content": [
                    'When diabetes is higher then sugar is higher (for "city")',
                    'Therefore sugar is lower for "countryside"',
                    "Therefore B is the correct answer",
                ],
            },
        ]


def main():
    counter = Counter()
    for attr in dir(counter):
        if attr.startswith("test"):
            getattr(counter, attr)()

if __name__ == "__main__":
        main()
