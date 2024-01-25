
from datasets import load_dataset

dataset = load_dataset("wikicorpus", "raw_en")
# you can use any of the following config names as a second argument:
# "raw_ca", "raw_en", "raw_es", "tagged_ca", 
# "tagged_en", "tagged_es"
split = dataset["train"].train_test_split(test_size=0.1)

train = split["train"]
test = split["test"]

print(train, test) # these are Dataset

# print first 10 
i = 0
for item in enumerate(train):
    print(item["text"])
    i += 1
    if i > 10:
        break