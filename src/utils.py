import os
import re
import json

class QADataset:

    def __init__(self, data, dir="."):
        self.data = data.lower().split("_")[0]
        benchmark = json.load(open(os.path.join(dir, "benchmark.json")))
        if self.data not in benchmark:
            raise KeyError("{:s} not supported".format(data))
        self.dataset = benchmark[self.data]
        self.index = sorted(self.dataset.keys())

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.dataset[self.index[key]]
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")


def locate_answer(sentence:str):

    ans = re.findall("^\s*(A|B|C|D)$", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D) or", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D) and", sentence)
    if len(ans) > 0:
        return ans[0].upper()
        
    ans = re.findall("^\s*(A|B|C|D)/", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D),", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("[Oo]ption (A|B|C|D)", sentence)
    if len(ans) > 0:
        return ans[0]

    ans = re.findall(":\s*(A|B|C|D)", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall("^\s*(A|B|C|D)\.", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall("^\s*(A|B|C|D)\"", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D):", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    return "A"

def locate_answer4pub_llama(sentence:str):

    sentence = sentence.split("Answer:")[-1]

    ans = re.findall("[Oo]ption (A|B|C|D)", sentence)
    if len(ans) > 0:
        return ans[0]


    ans = re.findall("OPTION (A|B|C|D)", sentence)
    if len(ans) > 0:
        return ans[0]


    ans = re.findall("^\s*(A|B|C|D)\"", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D):", sentence)
    if len(ans) > 0:
        return ans[0].upper()    

    return "A"