# Hugging Face: `tau/commonsense_qa`

```Python
from datasets import load_dataset

dataset = load_dataset("commonsense_qa", split="validation[:10]")
```

CommonsenseQA is a new multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers . It contains 12,102 questions with one correct answer and four distractor answers. The dataset is provided in two major training/validation/testing set splits: "Random split" which is the main evaluation split, and "Question token split", see paper for details: [URL](https://arxiv.org/pdf/1811.00937).

An example of 'train' looks as follows:
```
{'id': '075e483d21c29a511267ef62bedc0461',
 'question': 'The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?',
 'question_concept': 'punishing',
 'choices': {'label': ['A', 'B', 'C', 'D', 'E'],
  'text': ['ignore', 'enforce', 'authoritarian', 'yell at', 'avoid']},
 'answerKey': 'A'}
```

Samples: 
- train=9741  
- validation=1221
- test=1140

