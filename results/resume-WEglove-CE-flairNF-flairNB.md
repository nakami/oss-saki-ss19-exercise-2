## resume-WEglove-CE-flairNF-flairNB (1 run)

### basic info

- sampling:     `1.0`
- fullpath:     `/content/gdrive/My Drive/SAKI_2019/data/resources/tagger/resume-WEglove-CE-flairNF-flairNB`
- epochs:       `20`
- size on disk: `? MB`

```python
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('glove'),
    # comment in this line to use character embeddings
    CharacterEmbeddings(),

    # comment in these lines to use flair embeddings (needs a LONG time to train :-)
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]
```

### output

```
iae
```

### timings

- beginning: min / 1 epoch
- run totoal: h