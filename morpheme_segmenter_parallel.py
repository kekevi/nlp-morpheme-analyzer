'''
Parallel version. Uses multiprocessing module as it seems multithread doesn't work exactly.

This file uses the morphemes library and creates a csv of each word split up into its constituent
morphemes (space separated). 

The reason we do this is that the morphemes library is quite slow and we would like to be able to
cache the result and just have python read it in using its csv implementation (which is significantly
 faster).
'''
N_PARALLEL = 8

from morphemes import Morphemes
import csv
from multiprocessing import Process

MO = Morphemes('./morphemes_files')

def getMorphemes(word: str, lower: bool = False) -> list[str]:
    '''
    Returns a list of morphemes from the MorphoLex. Returns [word] if morpheme is not in MorphoLex.
    @param word - the word
    @param lower - whether the returned morphemes should be lower case only
    '''
    # get data from MorphoLex
    res = MO.parse(word)
    if res is None:
        raise 'Word ' + word + ' breaks the morphemes library.'
    
    if res['status'] == 'NOT_FOUND':
        return [word.lower() if lower else word]
    
    # linearize the morphology tree
    morphemes = []
    def dfs(curr):
        nonlocal morphemes
        if curr is None:
            print(word, 'has a None in child')
            return # skip due to bug in library
        
        if 'children' in curr:
            for node in curr['children']:
                dfs(node)
        elif 'text' in curr:
            if curr['text'] != '': # weird bug from the library
                morphemes.append(curr['text'])
        else:
            print('Not an actual node', curr)

    for node in res['tree']:
        dfs(node)
    
    return morphemes

def task(id, sublist):
    with open(f'./morphemes_files/morphemes-{id}.csv', 'w') as file:
        writer = csv.writer(file)
        for word in sublist:
            writer.writerow([word, ' '.join(getMorphemes(word, True))])

def main():
    VOCAB_LIST = [] # for tracking progress during debugging
    with open('./morphemes_files/morpholex_words.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            VOCAB_LIST.append(row[0].lower())

    # partition VOCAB_LIST (note, last object gets the remainder)
    sublists = []
    i = 0
    diff = len(VOCAB_LIST)//N_PARALLEL
    for _ in range(N_PARALLEL-1):
        sublists.append(VOCAB_LIST[i:i+diff])
        i += diff
    sublists.append(VOCAB_LIST[i:])

    print('sanity check', sum(len(sublist) for sublist in sublists))

    procs = []
    for i in range(N_PARALLEL):
        p = Process(target=task, args=(i, sublists[i]))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # combine constituent csvs
    with open(f'./morphemes_files/morphemes.csv', 'w') as combined_file:
        writer = csv.writer(combined_file)
        for i in range(N_PARALLEL):
            with open(f'./morphemes_files/morphemes-{i}.csv', 'r') as file:
                writer.writerows(list(csv.reader(file)))

if __name__ == "__main__":
    main()
