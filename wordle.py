import collections
import unittest
import argparse
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from collections import namedtuple

wordfile = "wordlist.txt"

def build_hist(wordfile):
    ctr = collections.Counter()
    with open(wordfile,'r') as wf:
        for line in wf.readlines():
            line = line.strip()
            if len(line) == 5: ctr.update(enumerate(line))
    return ctr

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


def word_score(word, hist, mask = '*****', incorrect = set()):
    wordSet = set(word)
    score = 0
    for i in range(len(word)):
        letter = word[i]
        letter_score = hist[(i,letter)]
        if letter in incorrect: return 0;
        elif mask[i] == Wildcard: score += letter_score
        elif type(mask[i]) is Match:
            if mask[i].letter == letter: score += letter_score
            else: return 0; 
        elif type(mask[i]) == Mismatch:
            if mask[i].letter == letter: return 0; 
            if not mask[i].letter in wordSet: return 0;
            score += letter_score
        elif type(mask[i]) == Range:
            for let in mask[i].letters:
                if let == letter: return 0; 
                if not let in wordSet: return 0;
            score += letter_score
    return score



def best_matches(wordfile, hist, count, mask = '*****', incorrect = set(), unique = False):
    ret = []
    with open(wordfile,'r') as wf:
        for line in wf.readlines():
            line = line.strip()
            if len(line) == 5:
                if unique and len(set(line)) != 5: continue 
                ret.append([line, word_score(line, hist, mask, incorrect)])
                ret.sort(key = lambda x: x[1], reverse = True)
                ret = ret[:count]
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Best wordle match.')
    parser.add_argument('--mask', type=str, dest='mask', default="*****",
                        help="what's already known. Use * for unknown, Uppercase for match, and lowercase when location unknown")
    parser.add_argument('--incorrect', dest='incorrect', default = "",
                        help='known letters that aren\'t matching')
    parser.add_argument('--unique', dest='unique', default=False, action='store_true',
                        help='forbid letter repetition')
    parser.add_argument('--count', dest='count', default=10, 
                        help='number of results')
    parser.add_argument('--wordfile', dest='wordfile', default='./wordlelist.txt',
                        help='word file to use')

    args = parser.parse_args()
    mask = parse_mask(args.mask)
    print ("mask {}, incorrect {}, unique {}".format(mask, args.incorrect, args.unique))
    hist = build_hist(wordfile)
    for w, s in (x for x in best_matches(wordfile=args.wordfile, hist = hist, count = args.count, mask = mask, incorrect = set(args.incorrect), unique = args.unique) if x[1] > 0):
        print("{} => {}".format(w, s))


#TESTS
class TestWordScore(unittest.TestCase):
    def setUp(self):
        self.hist = build_hist(wordfile)

    def testEmpty(self):
        words = ["hello", "ghost", "trial"]
        for word in words:
            expected = sum([self.hist[(i,c)] for (i,c) in enumerate(word)])
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



