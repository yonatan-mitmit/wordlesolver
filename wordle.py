"""
Fix unittests to focus on parsers and not solver

"""

import collections
import unittest
import argparse
import string
import itertools
import enum
import hashlib
import pickle
from math import log2
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from collections import namedtuple, defaultdict
import numpy as np
import functools


wordfile = "wordlist.txt"

class Color(enum.Enum):
    GRAY = 'b'
    YELLOW = 'y'
    GREEN = 'g'
    
    def __str__(self):
        return self.value

class LetterMatch:
    # Green tells me exact location, and sets lower bound on count (number of yellow and green)
    # Yellow forbids exact location, and sets lower bound on count (number of yellow and green)
    # Grey tells a +1 of exact count
    #
    def __init__(self, size=5):
        self.greens = {}
        self.yellows = defaultdict(set)
        self.lbs = {}
        self.counts = {}
        self.size = size

    def matches(self, word):
        for (i,l) in self.greens.items():
            if word[i] != l: return False
        for (i,ls) in self.yellows.items():
            if word[i] in ls: return False
        ctr = collections.Counter(word)
        for l, lb in self.lbs.items():
            if ctr[l]<lb:
                return False
        for l, c in self.counts.items():
            if ctr[l]!=c:
                return False
        return True

    def update(self, word, results):
        ctr = collections.Counter(word)
        exacts = set()
        lb = set()

        for i in range(len(word)):
            result = results[i]
            letter = word[i]
            match result:
                case Color.GRAY: 
                    exacts.add(letter)
                case Color.YELLOW:
                    self.yellows[i].add(letter)
                    lb.add(letter)
                case Color.GREEN:
                    self.greens[i] = letter
                    lb.add(letter)
        for l in exacts:
            self.counts[l] = ctr[l] - 1
        for l in (lb - exacts): # Letters for which we have yellow or green, but not gray
            self.lbs[l] = max(ctr[l], self.lbs.get(l,0))


wcGrammer = Grammar("""
    pair = word ":" colors
    colors = color*
    word = letter*
    color = green / yellow / gray
    green = "g"
    yellow = "y"
    gray = "b"
    letter = ~r"[a-z]"
""")

Pair = namedtuple("Pair", ['word','colors'])

class WordColorVisitor(NodeVisitor):
    def visit_pair(self, node, children):
        return Pair(children[0], children[2])
                
    def visit_word(self, node, children):
        return ''.join(children)

    def visit_colors(self, node, children):
        return children
    
    def visit_color(self, node, children):
        return children[0]

    def visit_green(self,node,children):
        return Color.GREEN

    def visit_yellow(self,node,children):
        return Color.YELLOW

    def visit_gray(self,node,children):
        return Color.GRAY

    def visit_letter(self, node, children):
        return node.text.lower()

    def generic_visit(self, node, children):
        return children or node

wcVisitor = WordColorVisitor()

def parse_word_color(wordcolor):
    tree = wcGrammer.parse(wordcolor)
    return wcVisitor.visit(tree)

word_grammer = Grammar("""
    word = atom*
    atom = match_letter / mismatch_letter / mismatch_range / wildcard
    match_letter = ~r"[A-Z]"
    mismatch_letter = ~r"[a-z]"
    mismatch_range = "[" mismatch_letter* "]"
    wildcard = "*"
""")

Match = namedtuple("Match", ['letter'])
Mismatch = namedtuple("Mismatch", ['letter'])
Range = namedtuple("Range", ['letters'])
Wildcard = "*"

class WordVisitor(NodeVisitor):
    def visit_word(self, node, children):
        return children

    def visit_atom(self, node, children):
        return children[0]

    def visit_mismatch_range(self, node, children):
        _, letters , _ = children
        return Range([x.letter for x in letters])

    def visit_match_letter(self, node, children):
        return Match(node.text.lower())
    
    def visit_mismatch_letter(self, node, children):
        return Mismatch(node.text)

    def visit_wildcard(self, node, children):
        return Wildcard

    def generic_visit(self, node, children):
        return children or node

GlobalWV = WordVisitor()

def parse_mask(word):
    tree = word_grammer.parse(word)
    return GlobalWV.visit(tree)



class BaseWordleSolver:
    @staticmethod
    def filterList(word_dict, letter_match):
        remaining = {}
        for word, idx in word_dict.items():
            #mat = BaseWordleSolver.word_matches(word, mask, incorrect)
            if letter_match.matches(word):
                remaining[word]=idx
        return remaining


    def __init__(self, solutions, candidates, hard_mode, letter_match):
        self.hard_mode = hard_mode
        self.all_solutions = {x[1]:x[0] for x in enumerate(solutions)}
        self.all_candidates = {x[1]:x[0] for x in enumerate(candidates)}
        self.letter_match = letter_match

        self.remaining_solutions = BaseWordleSolver.filterList(self.all_solutions, self.letter_match)

        if self.hard_mode:
            self.remaining_candidates = BaseWordleSolver.filterList(self.all_candidates, self.letter_match)
        else:
            self.candidates = self.all_candidates 
    

    @staticmethod
    def word_matches(word, mask, incorrect):
        wordSet = set(word)
        for i in range(len(word)):
            letter = word[i]
            if letter in incorrect: return False
            elif mask[i] == Wildcard: continue
            elif type(mask[i]) is Match:
                if mask[i].letter == letter: continue
                else: return False; 
            elif type(mask[i]) == Mismatch:
                if mask[i].letter == letter: return False; 
                if not mask[i].letter in wordSet: return False;
                continue
            elif type(mask[i]) == Range:
                for let in mask[i].letters:
                    if let == letter: return False; 
                    if not let in wordSet: return False;
                continue
        return True


    def word_score(self, word, word_idx):
        raise NotImplemented("Base Model")

    def best_matches(self, count, unique = False):
        best_candidates = []

        for word, idx in self.remaining_candidates.items():
            if unique and len(set(word)) != 5: continue 
            #if not self.word_matches(word): continue
            score = self.word_score(word, idx)
            best_candidates.append((word, word in self.remaining_solutions,  score))
            #for w in ret:
            #    self.debug_ent(w[0])

        best_candidates.sort(key = lambda x: (x[2], x[1], x[0]), reverse = True)
        best_candidates = best_candidates[:count]

        return best_candidates

    def all_scores(self):
        return { word : self.word_score(word, idx) for (word, idx) in self.all_candidates.items()} 


class HighestProbability(BaseWordleSolver):
    def __init__(self, solutions, candidates, hard_mode, letter_match):
        super().__init__(solutions, candidates, hard_mode, letter_match)
        self.hist = self.build_hist()

    def build_hist(self):
        ctr = collections.Counter()
        for word in self.all_solutions.keys():
            ctr.update(enumerate(word))
        return ctr

    def word_score(self, word, word_idx):
        return sum([log2(self.hist[(i[0],i[1])]) for i in enumerate(word) if self.hist[(i[0],i[1])] >0 ])


class LetterEntropy(BaseWordleSolver):

    def letter_score(self, letter):
        pair = self.map[letter]
        (lw, lwo) = [len(x) for x in pair]
        total = lw + lwo
        if total == 0:
            raise Exception("letter_score partitioned an empty set. Verify `mask` and `incorrect` are consistent")
        p = lw / total
        ent = -(p * log2(p + 1e-30) + (1-p) * log2(1-p + 1e-30))
        return ent

    def __init__(self, solutions, candidates, hard_mode, letter_match):
        super().__init__(solutions, candidates, hard_mode, letter_match)
        lowercase = set(string.ascii_lowercase)
        self.map = {x : (set(), set()) for x in string.ascii_lowercase}
        
        for word in self.all_candidates.keys():
            mat = self.word_matches(word, mask, incorrect)
            if mat:
                for letter in word:
                    self.map[letter][0].add(word)
                for letter in (lowercase - set(word)):
                    self.map[letter][1].add(word)

        self.letter_ent = {x : self.letter_score(x) for x in string.ascii_lowercase}


    def word_score(self, word, word_idx):
        return sum(self.letter_ent[x] for x in set(word))


class CachePolicy(enum.Enum):
    IGNORE = 'ignore'
    SAVE = 'save'
    LOAD = 'load'
    LOAD_AND_SAVE = 'load_and_save'

    def __str__(self):
        return self.value


class WordEntropy(BaseWordleSolver):

    def __init__(self, all_solutions, all_candidates, hard_mode, letter_match , cache_policy = CachePolicy.IGNORE):
        super().__init__(all_solutions, all_candidates, hard_mode, letter_match)
        self.cache_prefix = 'entmat'
        self.word_matrix = WordEntropy.get_entropy_matrix(cache_policy, self.cache_prefix, all_solutions, all_candidates)

    @staticmethod 
    def get_words_hash(*sets, letters=8):
        hs = hashlib.sha256()
        for word_set in sets:
            for word in word_set:
                hs.update(word.encode('utf-8'))
        digest = hs.hexdigest()[:letters]
        return digest

    @staticmethod
    def get_cache_name(prefix, *sets, letters=8):
        digest = WordEntropy.get_words_hash(*sets, letters=letters)
        fn = "{}-{}.npy".format(prefix, digest)
        return fn


    @staticmethod
    def load_match_matrix(prefix, all_solutions, all_candidates, letters = 8):
        fn = WordEntropy.get_cache_name(prefix, all_solutions, all_candidates, letters = letters)
        try:
            print("Loading entropy matrix from {}".format(fn));
            return np.load(fn)
        except FileNotFoundError as e:
            print("No match matrix found")
            return None

    @staticmethod
    def save_match_matrix(prefix, all_solutions, all_candidates, matrix, letters = 8):
        fn = WordEntropy.get_cache_name(prefix, all_solutions, all_candidates, letters = letters)
        print("Saving entropy matrix to {}".format(fn));
        np.save(fn, matrix)

    @staticmethod
    def build_match_matrix(solutions, candidates):
        acc = lambda x,y: x*3 + y
        def let(idx, letter, word_2):
            if letter == word_2[idx]: return 2;
            if letter in word_2: return 1;
            return 0

        dim_solutions = len(solutions)
        dim_candidates = len(candidates)
        ret = np.zeros((dim_candidates, dim_solutions), dtype = np.uint8)
        for i1, word_1 in enumerate(candidates):
            for i2, word_2 in enumerate(solutions):
                tup = map(lambda w: let(w[0], w[1], word_2), enumerate(word_1))
                ret[i1,i2] = functools.reduce(acc, tup, 0)
        return ret


    @staticmethod
    def get_entropy_matrix(cache_policy, prefix, all_solutions, all_candidates, letters = 8):
        matrix = None
        loaded = False

        if cache_policy in [CachePolicy.LOAD, CachePolicy.LOAD_AND_SAVE]:
            matrix = WordEntropy.load_match_matrix(prefix, all_solutions, all_candidates, letters)
            if matrix is not None: loaded = True

        if matrix is None:
            matrix = WordEntropy.build_match_matrix(all_solutions, all_candidates)

        if cache_policy == CachePolicy.SAVE or (cache_policy == CachePolicy.LOAD_AND_SAVE and loaded == False):
            WordEntropy.save_match_matrix(prefix, all_solutions, all_candidates, matrix, letters)

        return matrix

    def word_score_int(self, candidate, candidate_idx):
        groups = defaultdict(int)
        for _, sol_idx in self.remaining_solutions.items():
            key = self.word_matrix[candidate_idx, sol_idx]
            groups[key]+=1
        total = len(self.remaining_solutions)
        ent = 0
        for s in groups.values():
            p = s / total
            ent += p * log2(p + 1e-50)
        #if (-ent > 1): 
        #    print(-ent, word)
        #    print(total)
        #    for k,s in groups.items():
        #        print("{} => {}".format(k, s)) 
        return -ent, groups

    def word_score(self, candidate, candidate_idx):
        return self.word_score_int(candidate, candidate_idx)[0]

    def debug_ent(self, candidate, candidate_idx):
        ent, gr = self.word_score_int(candidate, candidate_idx)
        print(word)
        for k,s in gr.items():
            print("{} => {}".format(k, s)) 

def find_solvers():
    return dict(map(lambda x: (x.__name__ , x), BaseWordleSolver.__subclasses__()))

def relevantArgs(func, args, kwargs, ignore=['self']):
    import inspect
    spec = inspect.getfullargspec(func)
    func_args = list(filter(lambda x: x not in ignore, spec.args))
    unbound = func_args[len(args):]
    return { x: kwargs[x] for x in unbound if x in kwargs}

def readWordFile(fn):
    words = set()
    with open(fn,'r') as wf:
        for line in wf.readlines():
            line = line.strip()
            if(len(line)==5): words.add(line)
    return words

if __name__ == "__main__":
    solvers = find_solvers()

    parser = argparse.ArgumentParser(description='Best wordle match.')
    parser.add_argument('--solver', type=str, dest='solver', default='WordEntropy', choices=list(solvers.keys()),
                        help="Solver to use")
    #parser.add_argument('--mask', type=str, dest='mask', default="*****",
    #                    help="what's already known. Use * for unknown, Uppercase for match, and lowercase when location unknown")
    #parser.add_argument('--incorrect', dest='incorrect', default = "",
    #                    help='known letters that aren\'t matching')
    parser.add_argument('-m', '--matches', nargs="*", default=[],
                        help="pairs of <word:match> where match is a sequence of [g,y,b] for green, yellow, black (gray)" )
    parser.add_argument('--unique', dest='unique', default=False, action='store_true',
                        help='forbid letter repetition')
    parser.add_argument('--count', dest='count', default=10, 
                        help='number of results')
    parser.add_argument('--wordfile', dest='wordfile', default='./wordlelist.txt',
                        help='valid solutions for the puzzle')
    parser.add_argument('--wordfile_accepted', dest='wordfile_accepted', default='./wordle_accepted.txt',
                        help='additional legal words allowed. Use empty string to ignore')
    parser.add_argument('--no_hard_mode', dest='hard_mode', action='store_false', default=True,
                        help='use hard mode')
    parser.add_argument('--cache_policy', dest='cache_policy', default=CachePolicy.LOAD_AND_SAVE, type=CachePolicy, choices=list(CachePolicy),
                        help='Cache handling policy')
    parser.add_argument('-n','--dummy', dest='dummy', default=False, action='store_true', 
                        help="Don't actually run the solver, just do setup")
                        


    args = parser.parse_args()
    matches = [parse_word_color(x) for x in args.matches]
    solver = solvers[args.solver]
    #print ("mask {}, incorrect {}, unique {}".format(mask, args.incorrect, args.unique))
    #hist = build_hist(wordfile)
    #for w, s in (x for x in best_matches(wordfile=args.wordfile, hist = hist, count = args.count, mask = mask, incorrect = set(args.incorrect), unique = args.unique) if x[1] > 0):
    #    print("{} => {}".format(w, s))
    solutions = readWordFile(args.wordfile)
    if args.wordfile_accepted:
        candidates = readWordFile(args.wordfile_accepted)
    else:
        candidates = set()

    candidates.update(solutions)

    letter_match = LetterMatch()
    for mat in matches:
        letter_match.update(*mat)
    solver_args = (solutions, candidates, args.hard_mode, letter_match) 
    extraArgs = relevantArgs(solver, solver_args, vars(args))
    we = solver(*solver_args, **extraArgs)
    #we = solver(words, args.hard_mode, mask, set(args.incorrect), cache_policy = args.cache)
    if not args.dummy:
        res = we.best_matches(args.count, unique = args.unique)

        for word, sol, score in res:
            symbol = '*' if sol else ' ' 
            print("{} {} => {}".format(word, symbol, score))

#TESTS
class TestWordScore(unittest.TestCase):
    def setUp(self):
        self.hist = build_hist(wordfile)

    def testEmpty(self):
        words = ["hello", "ghost", "trial"]
        for word in words:
            expected = sum([log2(self.hist[(i,c)]) for (i,c) in enumerate(word)])
            self.assertEqual(word_score(word, self.hist), expected)

    def testWithIncorrect(self):
        words = ["hello", "ghost", "tribe"]
        incorrect = set("lr")
        scores = [0, word_score(words[1], self.hist), 0]
        for i, word in enumerate(words):
            self.assertEqual(word_score(word, self.hist, incorrect = incorrect), scores[i])

    def testWithUppercaseOnly(self):
        words = ["hello", #Matches
                "ghost",  #No match
                "trial",  #Letter found, but different place
                "boldl"   #Letter found and *also* in a wrong place
                ]
        mask = parse_mask("**L**")
        scores = [word_score(words[0], self.hist),  0, 0, word_score(words[3], self.hist)]
        for i, word in enumerate(words):
            self.assertEqual(word_score(word, self.hist, mask = mask), scores[i])

    def testWith2UppercaseOnly(self):
        words = ["hello", #Matches
                "ghost",  #No match
                "trial",  #Letter found, but different place
                "boldl"   #Letter found and *also* in a wrong place
                ]
        mask = parse_mask("**L*O")
        scores = [word_score(words[0], self.hist),  0, 0, 0]
        for i, word in enumerate(words):
            self.assertEqual(word_score(word, self.hist, mask = mask), scores[i])

    def testWithLowercaseOnly(self):
        words = [
                "trial",  #Letter found, in different place - matches
                "ghost",  #Letter doesn't appear
                "hello",  #Too close match - L in the right place 
                "boldl"   #Letter found and *also* in a wrong place
                ]
        mask = parse_mask("**l**")
        scores = [word_score(words[0], self.hist),  0, 0, 0]
        for i, word in enumerate(words):
            self.assertEqual(word_score(word, self.hist, mask = mask), scores[i])

    def testWithMultipleMismatches(self):
        words = [
                "trial",  #Letter found, in different place - matches
                "ghost",  #Letter doesn't appear
                "hello",  #Too close match - L in the right place 
                "herlo",  #r instead of l, but pattern shouldn't match
                "boldl"   #Letter found and *also* in a wrong place
                ]
        mask = parse_mask("**[lr]**")
        scores = [word_score(words[0], self.hist),  0, 0, 0, 0]
        for i, word in enumerate(words):
            self.assertEqual(word_score(word, self.hist, mask = mask), scores[i])

    def testMixLowercaseUppercase(self):
        words = [
                "trial",  #matches
                "ghost",  #no match
                "heilo",  #l in the wrong place, i matches
                "iobdl"   #i in wrong place
                ]
        mask = parse_mask("**Il*")
        scores = [word_score(words[0], self.hist),  0, 0, 0]
        for i, word in enumerate(words):
            self.assertEqual(word_score(word, self.hist, mask = mask), scores[i])

    def testMixLowercaseUppercaseAndExclusions(self):
        words = [
                "trial",  #matches
                "tribl",  #matches
                "ghost",  #no match
                "heilo",  #l in the wrong place, i matches
                "iobdl"   #i in wrong place
                ]
        mask = parse_mask("**Il*")
        incorrect = set("bqe")
        scores = [word_score(words[0], self.hist),  0, 0, 0, 0]
        for i, word in enumerate(words):
            self.assertEqual(word_score(word, self.hist, mask = mask, incorrect = incorrect), scores[i])



