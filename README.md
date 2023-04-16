# Analyzing Usefulness of Morpheme Analysis in NLP

by Kevin Chen

---

Most NLP approaches to language have the *word* as the atomic unit of meaning, which in languages with an alphabetic orthography is easy to determine being space separated. However, linguistically and psychologically, the *morpheme* is the atomic unit of meaning. This is extra-important for languages with rich morphology like Finnish, or languages with an unclear separation between words and characters like Japanese. 

By approaching NLP in a more human-like way, we can potentially create a model that is more likely to understand slang, misspellings, and other non-regular words. 

In this project, we explore this idea through a toy project where we train a model that takes a definition as input and generates a word (that may not be real) that has that meaning.

## Morphology Pipeline

To do this, the first component is the **morpheme segmenter/parser**, which is an RNN model that predicts where morpheme boundaries are within a word by taking in characters of a word.
* eg. "impression" --> "im-press-ion" 
* We use an RNN as it takes into consideration of location of a morpheme within a word as most morphemes are only attached to one side of a word (either a prefix or a suffix).

The second component is the **morpheme determiner**, which will split a word into substrings based on the morpheme segmenter's output AND label these substrings as one of the morphemes in $V$, our "vocabulary"/lexicon of English morphemes. In order to accurately determine the morpheme, it may need to take into consideration the surrounding morphemes and its relative position in the word. We will be using the MorphoLex-en project through the `morphemes` Python package to create $V$ and as a training set of where morpheme boundaries are.
* eg. "im-press-ion" --> ['in', 'press', 'ion']
* It might be nice to introduce arguments (nth_prefix, nth_suffix) into the determiner function where nth_prefix is its index from the front and nth_suffix is its index from the back. Or some other argument that is percent_of_morphemes.

The third component is the **morpheme embedder**, which is a function that takes in an input morpheme and outputs a vector-representation of that morpheme. Rather than reinventing the wheel, this function will be made by using Word2Vec to embed all the words in our training set. Then for each morpheme in $m \in V$, we find all the words in $training\_ words$ that include $m$. Then the vector-representation of $m$ will be some average of all of the embeddings of those words. Note that this average function should attempt to not be biased by the number of similar vectors (maybe clustering then average).
* eg. embedder('in') --> (#, ..., #)
* Alternatively, this embedder could be the result of a word2vec across the WordNet definition of the word

## Definition Pipeline to Generation

Now we can finally create a word based on its definition. This is also divided into subcomponents. 

The first component is the **definition encoder**. This is an RNN model that reads a definition sentence $d = [x_1, x_2, ..., x_n]$ where $n=|d|$ and $x_i$ is the $i$th word. This model will output a definition encoding vector that will be used by the decoder.

The second component is the **definition decoder**. This is another RNN model that reads in the definition encoding vector from the encoder and generates a sequence of morphemes that capture all the information in the defintion. 

These two components will have to be trained together. To train, we will get definitions of all the words in the MorphoLex set (from Princeton's WordNet, which tends to have shorter definitions (thus less noisy)). Put these definitions through the encoder and teacher-force the decoder with the actual morphemes described from MorphoLex.

Finally the last component (if we have time) is the **orthotactics concatenator**. This function will attempt to combine consecutive morphemes in the most likely way possible.
* eg. concat('in', 'possible') --> "i<u>m</u>possible"
* eg. concat('make', 'ing') --> "mak~~e~~ing" = "making"

The best way to implement this is likely to be rule-based rather than any trainable model.