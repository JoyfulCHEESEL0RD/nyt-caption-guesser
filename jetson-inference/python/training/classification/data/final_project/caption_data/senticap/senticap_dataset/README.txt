Included Files
##############

senticap_reader.py: Contains classes for reading and using the senticap dataset
doc/ : documentation for senticap_reader.py
data/senticap_dataset.json: Contains the examples in `JSON file format` described below.

JSON file format:
#################

A list of `Image Object` each containing a list of corresponding `Sentence Object`.

Image Object
-------------
 - filename: Filename of the image in the MSCOCO dataset.
 - split: One of ["train", "test", "val"] indicating the set this example belongs to.
 - imgid: An unique numeric ID for this image. Assigned arbritrarily.
 - sentences: A list of sentence objects see: `Sentence Object` below 

Sentence Object
---------------
 - tokens: A tokenized version of the sentence with punctuation removed and words made lower case. Provided as a list
 - word_sentiment: Indicates which words are part of an ANP with sentiment. An element has a value 1 if the word is part of an ANP with sentiment, 0 otherwise.
 - sentiment: The sentiment polarity for the sentence. If 1 then the sentence expresses positive sentiment, if 0 then the sentence expresses negative sentiment
 - raw: The caption without any processing; taken directly from MTURK.
