import re
import nltk

def modernize(scene):
    output = re.sub(r'([yw])es\b',r'\1s',scene)
    output = re.sub(r'([aeiou]{2}[^aeiou])e\b',r'\1',output)
    output = re.sub(r'(ess)e\b',r'\1',output)
    output = re.sub(r'([a-z]*[aeiou]*[a-z]*)\'st\b',r'\1',output)
    output = re.sub(r'([aeiou]*)\'d\b',r'\1ed',output)
    return output
