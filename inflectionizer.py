import csv
import nltk
from nltk.corpus import wordnet

lemmatizer = nltk.WordNetLemmatizer()

#  
# loading in data
#
WORD_LIST = []
WORD_2_MORPH = {}
with open('./morphemes_files/morphemes.csv') as file:
    reader = csv.reader(file)
    for word, morphemes_spaced in reader:
        WORD_LIST.append(word)
        WORD_2_MORPH[word] = morphemes_spaced.split()



#
# processing parts of speech
#

'''
It appears that the set of pos tags used by MorphoLex/ELP are:
- JJ: adjectives
- VB: verbs
- NN: nouns
- RB: adverb
- minor: onomatopoeia, interjections, foreign words/expressions (but not places), archaic words, 
- encl: words containing 's, n't, 'll, 'd, 're, 've, na (like wanna and gonna), ya (like whaddya, buncha), basically clitics
- '': just the word 'o so a mischaracterization I guess

I think I will just treat minor as nouns and I'll have to figure out what to do for the encl clitics ('s, n't, 'll, 'd, 're, 've).
However, it's looking like I will just ignore them since most of them are restricted to pronoun or auxiliary verbs rather than being
regular, EXCEPT for possessive 's 
'''

VALID_INFLECTIONS = {
    'es', 
    's', 
    'ed', 
    'd', 
    'ing', 
    "'s"
}
NORMAL_INFLECTIONS = {
    'es': 's',
    's': 's',
    'ed': 'ed',
    'd': 'ed',
    'ing': 'ing',
    "'s": "'s"
}
def normalize_inflection(inflection):
    if inflection[-3:] in VALID_INFLECTIONS:
        return NORMAL_INFLECTIONS[inflection[-3:]]
    elif inflection[-2:] in VALID_INFLECTIONS:
        return NORMAL_INFLECTIONS[inflection[-2:]]
    elif inflection[-1:] in VALID_INFLECTIONS:
        return NORMAL_INFLECTIONS[inflection[-1:]]
    else:
        return ''

def strDiff(longer, shorter):
    '''
    returns all the way till matching
    '''
    if len(shorter) > len(longer):
        print(longer, shorter)
        shorter, longer = longer, shorter
    i = 0
    while i < len(shorter):
        if longer[i] != shorter[i]:
            break
        i += 1
    return longer[i:]


'''
Returns the inflectional suffix for a word given that it exists, for the first part of speech in the poses list.
Will only return valid inflectional morphemes
Test on:
shitting --> shit has a +t (so should just be ing, not ting)
impieties --> impiety has a -y+i change (so impiety not in impieties)
'''
def find_inflection(word, poses = [wordnet.VERB, wordnet.ADJ, wordnet.NOUN, wordnet.ADV]):
    for pos in poses:
        lemma = lemmatizer.lemmatize(word, pos)
        remainder = strDiff(word, lemma)
        # print(f"{lemma} + {remainder} = {word} of {pos}")
        if not remainder: # lemmatized makes no difference
            continue
        if lemma in word:
            return remainder, lemma
        elif lemma[:-1] in word: # in case of letter change
            return remainder, lemma
    if word[-2:] == "'s":
        return ("'s", word[:-2])
    return '', word

'''
fixes MorphoLex's missing derivations, typically in the first 20k or so words
eg. tutored --> tutor, but MorphoLex has it as tutored
'''
def attach_inflection(morphemes, inflection):
    if inflection[-3:] == 'ing' and morphemes[-1][-3:] == 'ing':
        if morphemes[-1][-1] != 'e':
            print('occurred!!!', morphemes)
            morphemes[-1] = morphemes[-1][:-3]
        morphemes.append('ing') 
    elif inflection[-2:] == 'es' and morphemes[-1][-2:] == 'es':
        morphemes[-1] = morphemes[-1][:-2]
        morphemes.append('s') # plural -es
    elif inflection == "'s" and morphemes[-1][-2:] == "'s":
        print(morphemes)
        morphemes[-1] = morphemes[-1][:-2]
        morphemes.append("'s") # possessive -'s
    elif inflection[-2:] == 'ed' and morphemes[-1][-2:] == 'ed':
        morphemes[-1] = morphemes[-1][:-2]
        morphemes.append('ed') # past tense -ed
    elif inflection[-1:] == 's' and morphemes[-1][-1:] == 's':
        if morphemes[-1][-2:] != 'ss':
            morphemes[-1] = morphemes[-1][:-1]
            morphemes.append('s') # plural -s
    elif inflection[-1:] == 'd' and morphemes[-1][-1:] == 'd':
        morphemes[-1] = morphemes[-1][:-1]
        morphemes.append('ed') # past tense -d, (when word already ends in e)
    else:
        if n_inf := normalize_inflection(inflection): morphemes.append(n_inf)
        
def find_clitic(word):
    if word[-2:] == "'s":
        return ( "'s", word[:-2])
    return ('', word)

for full_word in WORD_LIST:
    clitic, word = find_clitic(full_word)
    inflection, lemma = find_inflection(word) # a colloquial "word" (full_word) = lemma + inflection + clitic
    if inflection: 
        # try to find if uninflected version exists:
        if lemma in WORD_2_MORPH:
            WORD_2_MORPH[full_word] = WORD_2_MORPH[lemma] + [normalize_inflection(inflection)]
        else:
            if full_word == 'indisposed':
                print('INDISPOSED', clitic, word, inflection, lemma)
            attach_inflection(WORD_2_MORPH[full_word], inflection)
    if clitic:
        if word in WORD_2_MORPH:
            WORD_2_MORPH[full_word] = WORD_2_MORPH[word] + [normalize_inflection(clitic)]
        else:
            print(full_word)
            attach_inflection(WORD_2_MORPH[full_word], clitic)
        

with open('./morphemes_files/morphemes_with_inflection.csv', 'w') as file:
    writer = csv.writer(file)
    for word in WORD_LIST:
        writer.writerow([word, ' '.join(WORD_2_MORPH[word])])

print(strDiff('shitting', 'shit'))
print(strDiff('impieties', 'impiety'))
print(strDiff('aliases', 'alias'))
print(strDiff('rambled', 'rambled'))
print(strDiff('tussling', 'tussling'))
print(strDiff('inspired', 'inspired'))
print('---')
print('shitting:', find_inflection('shitting'))
print('impieties:', find_inflection('impieties'))
print('aliases:', find_inflection('aliases'))
print('rambled:', find_inflection('rambled'))
print('tussling:', find_inflection('tussling'))
print('indisposed:', find_inflection('indisposed'))
print('california\'s:', find_clitic('california\'s'))
print("california" in WORD_2_MORPH)