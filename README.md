# Natural Language Processing with Hugging Face Transformers
Generative AI Guided Project on Cognitive Class by IBM

## Name : Jocelyn

## My todo :

### 1. Example 1 - Sentiment Analysis

```python
# TODO :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I am studying Artificial Intelligence at Infinite Learning, it is so exciting!")
```

Result :

```
[{'label': 'POSITIVE', 'score': 0.9998260140419006}]
```

Analysis on example 1 :

The sentiment analysis classifier correctly identifies the positive tone in the sentence. The confidence score of 99.98% shows that the model is highly reliable in detecting enthusiasm and excitement in straightforward English sentences. This result reflects how well the model handles informal, expressive language related to personal experiences.

---

### 2. Example 2 - Topic Classification

```python
# TODO :
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "Studying at Infinite Learning requires dedication, critical thinking, and the ability to manage time effectively. Students class, complete assignments, and collaborate with peers to gain knowledge and prepare for the team assignment.",
    candidate_labels=["education", "sports", "cooking"],
)
```

Result :

```
{'sequence': 'Studying at Infinite Learning requires dedication, critical thinking, and the ability to manage time effectively. Students class, complete assignments, and collaborate with peers to gain knowledge and prepare for the team assignment.',
 'labels': ['education', 'sports', 'cooking'],
 'scores': [0.9542193412780762, 0.026770269498229027, 0.0190103892236948]}
```

Analysis on example 2 :

The zero-shot classifier successfully identifies "education" as the most relevant label with a 95.42% confidence score. This demonstrates the model's strong ability to understand context and associate descriptive academic language with the correct category, even without any task-specific training on the input text.

---

### 3. Example 3 and 3.5 - Text Generator

```python
# TODO :
generator = pipeline("text-generation", model="distilgpt2")
generator(
    "Being an Artificial Intelligence student at Infinite Learning has taught me that",
    max_length=40,
    num_return_sequences=2,
)
```

Result :

```
[{'generated_text': 'Being an Artificial Intelligence student at Infinite Learning has taught me that in almost all cultures, the average student is not one to be intimidated off. For example, a student who likes to be alone or out'},
 {'generated_text': 'Being an Artificial Intelligence student at Infinite Learning has taught me that the word ‒ was just a bit more complicated.\n\n\n\nAs you can see, a lot of people want to create an'}]
```

Analysis on example 3 :

The text generation model produces creative and coherent continuations of the given prompt. Both outputs are contextually relevant and demonstrate the model's ability to generate fluent sentences. However, the content may vary in logic and relevance, which is expected behavior for generative language models.

```python
unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("Jocelyn is a student who <mask> very hard every day", top_k=4)
```

Result :

```
[{'score': 0.7397422790527344, 'token': 1364, 'token_str': ' works', 'sequence': 'Jocelyn is a student who works very hard every day'},
 {'score': 0.09090238064527512, 'token': 7717, 'token_str': ' trains', 'sequence': 'Jocelyn is a student who trains very hard every day'},
 {'score': 0.07413394004106522, 'token': 5741, 'token_str': ' tries', 'sequence': 'Jocelyn is a student who tries very hard every day'},
 {'score': 0.015245185233652592, 'token': 30761, 'token_str': ' strives', 'sequence': 'Jocelyn is a student who strives very hard every day'}]
```

Analysis on example 3.5 :

The fill-mask pipeline accurately predicts the most contextually appropriate word to replace the mask. The top prediction "works" achieves a high confidence score of 73.97%, and the remaining predictions such as "trains", "tries", and "strives" are all contextually suitable, showing the model's strong understanding of sentence structure and meaning.

---

### 4. Example 4 - Name Entity Recognition (NER)

```python
# TODO :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("My name is Jocelyn, I am an Artificial Intelligence student at Infinite Learning, Batam Island")
```

Result :

```
[{'entity_group': 'PER', 'score': np.float32(0.9991823), 'word': 'Jocelyn', 'start': 11, 'end': 18},
 {'entity_group': 'MISC', 'score': np.float32(0.6227044), 'word': '##ial Intelligence', 'start': 35, 'end': 51},
 {'entity_group': 'ORG', 'score': np.float32(0.97328424), 'word': 'Infinite Learning', 'start': 63, 'end': 80},
 {'entity_group': 'LOC', 'score': np.float32(0.9840598), 'word': 'Batam Island', 'start': 82, 'end': 94}]
```

Analysis on example 4 :

The NER model successfully identifies key entities from the sentence including a person (Jocelyn), an organization (Infinite Learning), and a location (Batam Island) with high confidence scores. The MISC label for "Artificial Intelligence" is expected as it is a general concept rather than a named entity, showing the model's nuanced understanding of entity classification.

---

### 5. Example 5 - Question Answering

```python
# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "Where does Jocelyn study?"
context = "Jocelyn is an Artificial Intelligence student who studies at Infinite Learning located in Batam Island, Indonesia."
qa_model(question = question, context = context)
```

Result :

```
{'score': 0.8200876116752625, 'start': 61, 'end': 78, 'answer': 'Infinite Learning'}
```

Analysis on example 5 :

The question-answering model correctly extracts "Infinite Learning" as the answer from the given context with an 82% confidence score. This demonstrates the model's strong capability in understanding natural language questions and accurately locating the most relevant answer span within a given passage.

---

### 6. Example 6 - Text Summarization

```python
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    """
Artificial Intelligence is one of the most transformative technologies of the 21st century. It enables computers and machines to simulate human intelligence, including the ability to learn from experience, recognize patterns, make decisions, and understand natural language. AI is now being applied across many industries, including healthcare, education, finance, transportation, and entertainment. In healthcare, AI helps doctors diagnose diseases more accurately and develop personalized treatment plans. In education, AI-powered tools can provide personalized learning experiences for students based on their individual needs and progress. In finance, AI is used to detect fraud, manage risks, and automate trading. In transportation, self-driving cars are becoming a reality thanks to advances in AI and machine learning. As a student of Infinite Learning Indonesia Artificial Intelligence Program, I believe that understanding AI is essential for preparing ourselves for the future in today's digital world.
"""
)
```

Result :

```
[{'summary_text': ' Artificial Intelligence is one of the most transformative technologies of the 21st century . It enables computers and machines to simulate human intelligence, including the ability to learn from experience, recognize patterns, make decisions, and understand natural language . AI is now being applied across many industries, including healthcare, education, finance, transportation, and entertainment .'}]
```

Analysis on example 6 :

The summarization model effectively condenses the paragraph into a shorter version while retaining the core ideas about AI's definition and broad industry applications. It demonstrates the model's ability to identify and preserve the most important information from a longer text without significant loss of meaning.

---

### 7. Example 7 - Translation

```python
# TODO :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Saya adalah mahasiswa program Kecerdasan Buatan di Infinite Learning")
```

Result :

```
[{'translation_text': "J'étudie l'intelligence artificielle à Infinite Learning."}]
```

Analysis on example 7 :

The translation model successfully translates the Indonesian sentence into natural and accurate French. It handles informal language smoothly and preserves the original meaning, demonstrating the model's effectiveness in multilingual tasks and its ability to understand context across different languages.

---

## Analysis on this project

This project provided a hands-on introduction to various NLP tasks using Hugging Face Transformers pipelines. Each example demonstrated how pre-trained models can be applied to real-world language tasks such as sentiment analysis, topic classification, text generation, named entity recognition, question answering, summarization, and translation. The variety of models and tasks highlighted the flexibility and power of transformer-based solutions. As an AI student at Infinite Learning, this project deepened my understanding of how modern NLP models work and how they can be applied efficiently using simple pipeline functions.
