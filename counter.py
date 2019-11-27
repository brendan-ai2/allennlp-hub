#from collections import defaultdict
#
#def install_hooks(module):
#    module.__allennlp_call_counter = defaultdict(int)
#    for attr in dir(module):
#        original = getattr(module, attr)
#        if not callable(original):
#            continue
#        # Introduce extra scope as Python is a joke of a language.
#        def python_sucks(attr, original):
#            def replacement(*args, **kwargs):
#                module.__allennlp_call_counter[attr] += 1
#                return original(*args, **kwargs)
#            return replacement
#        replacement = python_sucks(attr, original)
#        setattr(module, attr, replacement)
#
## Install the hooks before we load the models on the off-chance somebody caches
## a function pointer during module initialization.
#from torch.nn import functional
#install_hooks(functional)


import spacy
from allennlp_hub import pretrained

class Counter:
    pass

def main():
    counter = Counter()
    for attr in dir(counter):
        if attr.startswith("test"):
            getattr(counter, attr)()

if __name__ == "__main__":
        main()
