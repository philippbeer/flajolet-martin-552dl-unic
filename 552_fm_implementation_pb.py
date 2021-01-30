from typing import List, Set
import wikipedia
import re
import numpy as np
from typing import Union
import hashlib
import sympy
import math
import random
import pandas as pd
import time
import multiprocessing


class WikiText:
    """loads pages from wikipedia and preprocesses them"""

    def __init__(self, term: str) -> None:
        self.term = term
        self.wiki_page: wikipedia.wikipedia.WikipediaPage = self.get_wiki_page(
            self.term)
        #self.content: List[str] = self.preprocess_page_content(self.wiki_page)
        self.content: np.ndarray = np.array(
            self.preprocess_page_content(self.wiki_page))

    def get_wiki_page(self, search_term: str) -> wikipedia.wikipedia.WikipediaPage:
        """
        returns wikipedia page from wiki wrapper API

        params:
        :search_term: (string) search term for which page from
        wikipedia is to be returned

        returns: (WikipediaPage obj) page from wikipedia
        """
        try:
            return wikipedia.page(search_term)
        except PageError:
            print(f'Page for search term: {search_term} could not be found')

    def get_count_distinct_words(self) -> int:
        """
        expects a list of strings to be processed and returns number of unique words
        """
        assert type(self.content) == np.ndarray, 'requires a numpy array'
        return len(set(self.content))

    def preprocess_page_content(self,
                                page: wikipedia.wikipedia.WikipediaPage) -> List[str]:
        """
        preprocess page content (remove non-word characters)

        params:
        :content: (string) content to cleaned

        returns:
        :list_of_content_words: (list of strings) corpus split into list
        """
        assert type(
            page) is wikipedia.wikipedia.WikipediaPage, 'requires WikipediaPage obj'
        assert len(page.content) != 0, 'requires non-empty corpus'
        l = page.content.lower().split(" ")
        for i in range(len(l)):
            l[i] = re.sub("\W", " ", l[i])
            l[i] = re.sub("\d", " ", l[i])
            l[i] = re.sub("\s+", " ", l[i])
        return l


class FlajoletMartin:
    """
    FlajoletMartin algorithm to estimate length of passed stream
    """

    def __init__(self, data_stream: np.ndarray,
                 nmap: int = 1, L: int = 22,
                 optimization: str = 'reduce',
                 prime_hash: bool = True) -> None:
        """
        prepares Flajolet-Martin distinct count on provided data stream
        params:
        :data_stream: (numpy array) data stream for which to count unique elements
        :copies: (integer) number of copies of the hash function
        :L: (integer) power by which 2 will be raised to define length of bitmap
        """
        assert type(
            data_stream) == np.ndarray, 'data stream must be numpy array'
        assert data_stream.shape[0] > 0, 'data stream must be set'
        assert len(data_stream.shape) == 1, 'data stream must be flat'
        self.data_stream = data_stream
        assert type(nmap) == int, 'number of copies must be integer'
        assert nmap > 0, 'number of bitmaps must be larger than zero'
        self.nmap: int = nmap
        opt_options = ['reduce', 'mean_r']
        assert optimization in opt_options, 'valid optimization strategy must be chosen'
        self.optimization = optimization

        self.C: float = 1.3  # bias correction factor
        self.phi: float = 0.77351  # correction factor
        self.L = self.get_L(len(self.data_stream))

        # bit vector initialized to zeros
        self.bitmaps = np.zeros((self.nmap, 2**self.L), dtype=int)
        self.vs: list = self.generate_hash_factors(range_end=25,
                                                   number=self.nmap,
                                                   prime=prime_hash)  # initialize number of v-factors for hash
        self.ws: list = self.generate_hash_factors(range_end=25,
                                                   number=self.nmap,
                                                   prime=prime_hash)  # initialize number of w-factors for hash

        # PCSA Counting
        self.m = sympy.randprime(1, self.nmap)
        self.n = sympy.randprime(1, self.nmap)

    def get_L(self, n: int) -> int:
        """
        generates L based on Probabilistics Counting
        Algorithms for Data Base Applications by Philippe Flajolet
        params:
        :n: (integer) length of data stream for which unique count to be executed
        returns:
        :L: (integer) L
        """
        assert type(n) == int, 'length of data stream must be integer'
        assert n > 0, 'length of data stream must be positive'
        return math.floor(math.log2(n/self.nmap)+4)

    def generate_hash_factors(self, range_end: int = 10,
                              prime: bool = False,
                              number: int = 1) -> np.ndarray:
        """
        generates list of hash factors v, w and p based on set number of copies
        """
        if prime:
            l_prime = list(sympy.sieve.primerange(1, range_end))
            l_prime.sort()
            while len(l_prime) < number:
                l_prime.append(sympy.nextprime(l_prime[-1]))
            random.shuffle(l_prime)  # send random shuffle
            return l_prime
        else:
            val = 0
            vals = list()
            for i in range(self.nmap):
                val = 2
                while (val % 2 == 0):
                    val = random.randint(1, 2*self.nmap)
                vals.append(val)
            assert len(vals) == self.nmap, 'to few hash values generated'
            return vals

    def hash_val(self, word: str, v: int, w: int) -> int:
        """
        execute hashing of passed value via function h(a) = ((va+w) mod p)
        params:
        :a: (word) value to be hashed
        :v: (integer) multiplication factor
        :w: (integer) addend
        :p: (integer) modulo value (ideally prime)
        """
        # turn word into list of characters
        l = list(word)
        term1: int = 0
        for i in range(len(l)):
            term1 += ord(l[i])*128**i
        return int((v*term1 + w) % 2**self.L)

    def update_bitmap(self, word: str) -> None:
        """
        hash function that hashes the given value
        params:
        :e: (int) integer values to be hashed
        returns:
        :result: (int) hash result
        """
        # calculate hash value
        for i in range(self.nmap):
            # calculate hash with current set of values
            hash_val = self.hash_val(word=word,
                                     v=self.vs[i],
                                     w=self.ws[i])
            # find rightmost set bit in hash value
            r = self.rightmost_set_bit(hash_val)
            if r == None:  # cases need to be ignored as element value is 0
                continue
            assert type(r) == int, 'r must be int'
            if self.bitmaps[i, r] == 0:
                self.bitmaps[i, r] = 1

    def rightmost_set_bit(self, v: int) -> int:
        """
        calculates the position of the rightmost set bit
        params:
        :v: (integer) value for which to obtain rightmost set bit
        returns:
        :rightmost_position_set:
        """
        # using bit operations to identify position
        # of least significant set bit
        if v == 0:
            return None
        return int(math.log2(v & (~v + 1)))


    def leftmost_zero(self, bitmap: np.ndarray) -> int:
        """
        identifies position of leftmost bit set to zero
        params:
        :b: (Numpy Array) to be searched for rightmost zero
        returns
        :leftmost_zero: (integer) returns rightmost zero index position
                            counting from the right starting with index 0
        """
        res = np.where(bitmap == 0)[0]  # finds all zeros in bitmap
        return res[0]

    def reduce_bitmaps(self, bitmap: np.ndarray) -> np.ndarray:
        """
        reduces the bitmaps filled by random hash functions
        to single bitmap via element wise or
        returns:
        :reduced_bitmap: bitmap joined by component-wise OR on all bitmaps
        """
        # set inital bitmap for elementwise OR
        if bitmap.shape[0] == 1:
            return bitmap[0]
        else:
            reduced_bitmap = bitmap[0, :]
            for i in range(1, bitmap.shape[0]):
                assert bitmap[i, :].shape == reduced_bitmap.shape
                comp_bitmap = bitmap[i, :]
                reduced_bitmap = np.bitwise_or(reduced_bitmap, comp_bitmap)
            return reduced_bitmap

    def fm(self) -> int:
        """
        applies hash function to each value in stream
        params:
        :stream: (Numpy Array) to which hash function needs to be applied elementwise
        returns:
        :hashed_values: (Numpy Array) hashed values stream
        """
        # allowing for hashing of entire stream
        vbitmap_update = np.vectorize(self.update_bitmap)
        # contains hashed values for each element in stream
        vbitmap_update(self.data_stream)

        if self.optimization == 'reduce':
            # reduce bitmap
            red_bitmap = self.reduce_bitmaps(self.bitmaps)
            R = self.leftmost_zero(red_bitmap)
            return self.C*2**R
        elif self.optimization == 'mean_r':
            R = np.zeros((self.nmap,))
            for i in range(self.nmap):
                R[i] = self.leftmost_zero(self.bitmaps[i, :])
            mean_R = np.mean(R)
            return self.C*2**mean_R

    def pcsa_bitmap(self, word: str) -> None:
        """
        pcsa bitmap
        params:
        :e: (int) integer values to be hashed
        returns:
        :result: (int) hash result
        """
        hashedx = self.hash_val(word=word,
                                v=self.m,
                                w=self.n)
        alpha = hashedx % self.nmap
        beta = math.floor(hashedx/self.nmap)
        assert isinstance(beta, int), "index is integer"
        idx = self.rightmost_set_bit(beta)
        self.bitmaps[alpha, idx] = 1

    def fm_pcsa(self) -> int:
        # allowing for hashing of entire stream
        vbitmap_update = np.vectorize(self.pcsa_bitmap)
        # contains hashed values for each element in stream
        vbitmap_update(self.data_stream)
        S = 0
        for i in range(self.nmap):
            R = 0
            while (self.bitmaps[i, R] == 1) and (R < self.L):
                R += 1
            S += R
        return math.floor(self.nmap/self.phi*2**(S/self.nmap))


def calc_sample(term: int, rounds: int, ret_wiki_term: List[str],
                ret_true_cnt: List[int],
                ret_fm_cnt: list,
                ret_pcsa_cnt: list) -> None:
    assert isinstance(term, str), "term is a string"
    print(f'started processing of {term}')
    stream = WikiText(term)
    distinct_count = stream.get_count_distinct_words()
    # lists to capture results
    wiki_term = list()
    true_count = list()
    fm_count = list()
    fm_pcsa_count = list()
    for i in range(rounds):
        wiki_term.append(term)
        true_count.append(distinct_count)
        fma = FlajoletMartin(stream.content, nmap=64)
        fm_count.append(fma.fm())
        fm_pcsa_count.append(fma.fm_pcsa())

    ret_wiki_term.extend(wiki_term)
    ret_true_cnt.extend(true_count)
    ret_fm_cnt.extend(fm_count)
    ret_pcsa_cnt.extend(fm_pcsa_count)
    print(f'term {term} attached to mgr lists')


if __name__ == "__main__":
    search_terms = ['michael jordan',
                    'covid',
                    '2020 Nagorno-Karabakh war',
                    'List of association football families',
                    'Weisswurst',
                    'List of Crusades to Europe and the Holy Land',
                    'List_of_fatal_dog_attacks_in_the_United_States_(2010s)',
                    'Donald Trump',
                    'Timeline of the Israeli–Palestinian conflict 2015',
                    'List of University of Pennsylvania people',
                    'university of nicosia',
                    'data privacy'
                    ]

    procs = multiprocessing.cpu_count() - 1  # number of processes
    rounds = 2
    jobs = []
    df_all = pd.DataFrame()

    manager = multiprocessing.Manager()
    ret_wiki_term = manager.list()
    ret_true_cnt = manager.list()
    ret_fm_cnt = manager.list()
    ret_pcsa_cnt = manager.list()


    ########
    for elm in search_terms:
        p = multiprocessing.Process(target=calc_sample,
                                          args=(elm,
                                                rounds,
                                                ret_wiki_term,
                                                ret_true_cnt,
                                                ret_fm_cnt,
                                                ret_pcsa_cnt))
        jobs.append(p)
        p.start()

    # checking they are done
    for j in jobs:
        j.join()
    print('done')
    data = {'wiki_term': ret_wiki_term[:],
            'true_count': ret_true_cnt[:],
            'fm_count': ret_fm_cnt[:],
            'pcsa_count': ret_pcsa_cnt[:]}
    
    df = pd.DataFrame.from_dict(data)
    df.to_csv('./fm_analysis_'+str(rounds)+'.csv', index=False)
