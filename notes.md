# Trainings

## WordEmbeddings-glove

### 1

#### basic info

- sampling:     `1.0`
- fullpath:     `/content/gdrive/My Drive/SAKI_2019/data/resources/tagger/resume-ner-WordEmbeddings-glove`
- size on disk: `346MB`

```python
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('glove'),
    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings (needs a LONG time to train :-)
    #FlairEmbeddings('news-forward'),
    #FlairEmbeddings('news-backward'),
]
```


#### timings

- beginning: 2:30min / 1 epoch