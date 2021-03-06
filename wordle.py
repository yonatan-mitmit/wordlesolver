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
import random


wordfile = "wordlist.txt"
class Color(enum.Enum):
    GRAY = '⬛'
    YELLOW = '🟨'
    GREEN = '🟩'
    
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
        exacts = defaultdict(int)
        lb = set()

        for i in range(len(word)):
            result = results[i]
            letter = word[i]
            match result:
                case Color.GRAY: 
                    exacts[letter] += 1
                case Color.YELLOW:
                    self.yellows[i].add(letter)
                    lb.add(letter)
                case Color.GREEN:
                    self.greens[i] = letter
                    lb.add(letter)
        for l,c in exacts.items():
            self.counts[l] = ctr[l] - c
        for l in (lb - set(exacts.keys())): # Letters for which we have yellow or green, but not gray
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



class BaseWordleSolver:
    @staticmethod
    def filterList(word_dict, letter_match):
        remaining = {}
        for word, idx in word_dict.items():
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
            self.remaining_candidates = self.all_candidates.copy()
    

    def word_score(self, word, word_idx):
        raise NotImplemented("Base Model")

    def best_matches(self, count, unique = False):
        best_candidates = []

        for word, idx in self.remaining_candidates.items():
            if unique and len(set(word)) != 5: continue 
            score = self.word_score(word, idx)
            best_candidates.append((word, word in self.remaining_solutions,  score))

        best_candidates.sort(key = lambda x: (x[2], x[1], x[0]), reverse = True)
        best_candidates = best_candidates[:count]

        return best_candidates

    def entropy(self):
        l = len(self.remaining_solutions)
        return l, log2(l)

    def all_scores(self):
        return { word : self.word_score(word, idx) for (word, idx) in self.all_candidates.items()} 

    def all_scores_list(self):
        return sorted(self.all_score.items(), key = lambda x: (x[1], x[0])) 


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
            raise Exception("letter_score partitioned an empty set. Verify letter_match isn't empty, and answer is consistent")
        p = lw / total
        ent = -(p * log2(p + 1e-30) + (1-p) * log2(1-p + 1e-30))
        return ent

    def __init__(self, solutions, candidates, hard_mode, letter_match):
        super().__init__(solutions, candidates, hard_mode, letter_match)
        lowercase = set(string.ascii_lowercase)
        self.map = {x : (set(), set()) for x in string.ascii_lowercase}
        
        for word in self.remaining_candidates.keys():
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

    def __init__(self, all_solutions, all_candidates, hard_mode, letter_match , cache_policy = CachePolicy.LOAD):
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
            #print("Loading entropy matrix from {}".format(fn));
            return np.load(fn)
        except FileNotFoundError as e:
            print("Tried, but failed to load match_matrix from {} ")
            return None

    @staticmethod
    def save_match_matrix(prefix, all_solutions, all_candidates, matrix, letters = 8):
        fn = WordEntropy.get_cache_name(prefix, all_solutions, all_candidates, letters = letters)
        print("Saving entropy matrix to {}".format(fn));
        np.save(fn, matrix)

    @staticmethod
    def build_match_matrix(solutions, candidates):
        print("Building match matrix");

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

    def word_score(self, candidate, candidate_idx):
        rs = np.fromiter(self.remaining_solutions.values(), dtype=int)
        keys = self.word_matrix[candidate_idx, rs]
        unique, counts = np.unique(keys, return_counts=True)

        total = len(self.remaining_solutions)
        probs = counts / total
        ent = 0

        #ent = np.log2(probs) * probs
        #return -np.sum(ent)
        return -np.dot(np.log2(probs), probs)

class Board:
    def __init__(self, solution):
        self._solution = solution
    
    def score(self, word):
        ret = [Color.GRAY] * 5
        ctr = collections.Counter(self._solution)
        # mark greens first
        for i,l in enumerate(word):
            if l == self._solution[i]:
                ret[i] = Color.GREEN
                ctr[l] -= 1
        for i,l in enumerate(word):
            if l != self._solution[i] and ctr[l] > 0:
                ret[i] = Color.YELLOW
                ctr[l] -= 1
        return ret

    def solution(self):
        return self._solution


class Game:
    def __init__(self, board, solver_fact):
        self.board = board
        self.solver_fact = solver_fact
        self.letter_match = LetterMatch()

    def use_word(self, word):
        score = self.board.score(word)
        self.letter_match.update(word, score)
        return ''.join((str(x) for x in score))

    def next_guess(self):
        solver = self.solver_fact(self.letter_match)
        guess = solver.best_matches(1)
        if len(guess) == 0:
            raise Exception("Can't find solution")
        return guess[0][0]

    def run_game(self, first_word = None, rounds = 6, disp = True):
        total_rounds = rounds
        next_word = first_word

        for i in range(rounds):
            if next_word is None:
                next_word = self.next_guess()
                if first_word is None: first_word = next_word
            score = self.use_word(next_word)
            if disp: print ("{} => {} ".format(next_word, score))
            if next_word == self.board.solution():
                if disp:
                    print("'{}' found in {} steps starting from '{}'".format(self.board.solution(), i+1, first_word))
                return i
            next_word = None 
        print("Failed to find word in {} rounds".format(i+1))
        return -1 

def game(solution, first_word = None, solver_class = WordEntropy, solutions = None, candidates = None, hard_mode = True):
    solver_fact = lambda lm: solver_class(solutions, candidates, hard_mode, lm)
    g = Game(Board(solution), solver_fact)
    return g.run_game(first_word)

def test(first_word, games = 100, solutions = None, candidates = None, hard_mode=True):
    rets = {} 
    failed = set()
    for g in range(games):
        solution = random.choice(list(solutions))
        print("Round {}. Word {}".format(g, solution))
        rounds = game(solution, first_word, solutions = solutions, candidates = candidates, hard_mode = hard_mode) 
        if (rounds < 0):
            failed.add(solution)
        else:
            rets[solution] = rounds + 1

    print("Ran {} rounds. Starting at {}. Average rounds was {} and failed {} times".format(games, first_word, sum(rets.values())/len(rets), len(failed)))
    return rets, failed




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
    candidates = sorted(list(candidates))
    solutions = sorted(list(solutions))

    letter_match = LetterMatch()
    for mat in matches:
        letter_match.update(*mat)
    solver_args = (solutions, candidates, args.hard_mode, letter_match) 
    extraArgs = relevantArgs(solver, solver_args, vars(args))
    we = solver(*solver_args, **extraArgs)
    solver_fact = lambda lm, solver_class=WordEntropy, solutions=solutions, candidates=candidates : solver_class(solutions, candidates, True, lm)
    #we = solver(words, args.hard_mode, mask, set(args.incorrect), cache_policy = args.cache)
    if not args.dummy:
        res = we.best_matches(args.count, unique = args.unique)
        print("There are {} possible solutions ({} bits of info)".format(*we.entropy()))

        for word, sol, score in res:
            symbol = '*' if sol else ' ' 
            print("{} {} => {}".format(word, symbol, score))

#TESTS - Currently broken - use old functional interface and predate the inclusion. 
#Should be rewritten
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



