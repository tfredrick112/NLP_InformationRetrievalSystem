# Add your import statements here

from nltk.corpus import wordnet
import nltk

# Add any utility functions here
def return_WordNet_POS_tag(word):
    """
    This function finds the POS tag returned by NLTK and then converts it to
    the WordNet POS tag format.
    """
    nltk_tag_first_char = nltk.pos_tag([word])[0][1][0].upper()
    tag_map_dict = {"N": wordnet.NOUN, "J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}
    if nltk_tag_first_char in tag_map_dict:
        return tag_map_dict[nltk_tag_first_char]
    else:
        return wordnet.NOUN # for the default case return Noun
