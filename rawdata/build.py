import os
import json
import pandas as pd

def load_data(data):
    answer_list = ["A", "B", "C", "D"]
    if data == "medqa": # 1273
        dataset = [json.loads(line) for line in open("medqa/data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl").readlines()]
        dataset = {str(i).rjust(4, "0"):{
            "question": item["question"], 
            "options": item["options"], 
            "answer": item["answer_idx"]
            } for i,item in enumerate(dataset)}
    elif data == "medmcqa": # 4183
        dataset = [json.loads(line) for line in open("medmcqa/data/dev.json").readlines()]
        dataset = {item["id"]:{
            "question": item["question"], 
            "options": {"A": item["opa"], "B": item["opb"], "C": item["opc"], "D": item["opd"]}, 
            "answer": answer_list[item["cop"] - 1]
            } for i,item in enumerate(dataset)}
    elif data == "pubmedqa": # 500
        dataset = json.load(open("pubmedqa/data/test_set.json"))
        dataset = {key:{
            "question": value["QUESTION"], 
            "options": {"A": "yes", "B": "no", "C": "maybe"}, 
            "answer": answer_list[["yes","no","maybe"].index(value["final_decision"])],
            "PMID": [int(key)]
            } for key,value in dataset.items()}
    elif data == "bioasq": # 618
        dataset = [item for i in [11,10,9,8,7] for j in range(len([f for f in os.listdir("bioasq/Task{:d}BGoldenEnriched".format(i)) if f.endswith(".json")])) for item in json.load(open("bioasq/Task{:d}BGoldenEnriched/{:d}B{:d}_golden.json".format(i, i, j+1)))["questions"] if item["type"] == "yesno"]
        dataset = {item["id"]:{
            "question": item["body"], 
            "options": {"A": "yes", "B": "no"}, 
            "answer": answer_list[["yes","no"].index(item["exact_answer"].lower())],
            "PMID": [int(s.split('/')[-1]) for s in item["documents"]]
            } for item in dataset}
    elif data == "mmlu": # 1089
        dataset = {}
        for domain in ["anatomy", "clinical_knowledge", "college_biology", "college_medicine", "medical_genetics", "professional_medicine"]:
            dat = pd.read_csv("mmlu/data/test/{:s}_test.csv".format(domain), names=["question", "A", "B", "C", "D", "answer"])
            for i in range(len(dat)):
                dataset[domain + "-" + str(i).rjust(3, "0")] = {
                    "question": dat["question"][i],
                    "options": dict(dat.iloc[i,1:5]),
                    "answer": dat["answer"][i]
                }
    return dataset

if __name__ == "__main__":
    benchmark = {}
    datasets = ["medqa", "medmcqa", "pubmedqa", "bioasq", "mmlu"]
    for data in datasets:
        benchmark[data] = load_data(data)
    json.dump(benchmark, open("../benchmark.json", "w"), indent=4)