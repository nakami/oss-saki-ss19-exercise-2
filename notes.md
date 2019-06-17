# Trainings

## Training call

```python
trainer.train(model_name,
              learning_rate=0.1,
              mini_batch_size=32,
              #anneal_with_restarts=True,
              max_epochs=42)
```

## WordEmbeddings-glove (1 run)

### basic info

- sampling:     `1.0`
- fullpath:     `/content/gdrive/My Drive/SAKI_2019/data/resources/tagger/resume-WEglove`
- epochs:       `42`
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

### output

```
2019-06-15 23:48:15,540 Testing using best model ...
2019-06-15 23:48:15,545 loading file resources/taggers/resume-ner-WordEmbeddings-glove/best-model.pt
2019-06-15 23:48:49,503 0.6888  0.5625  0.6193
2019-06-15 23:48:49,505 
MICRO_AVG: acc 0.4485 - f1-score 0.6193
MACRO_AVG: acc 0.4926 - f1-score 0.6186
-          tp: 7 - fp: 14 - fn: 469 - tn: 7 - precision: 0.3333 - recall: 0.0147 - accuracy: 0.0143 - f1-score: 0.0282
Degree     tp: 100 - fp: 72 - fn: 41 - tn: 100 - precision: 0.5814 - recall: 0.7092 - accuracy: 0.4695 - f1-score: 0.6390
Designation tp: 295 - fp: 179 - fn: 136 - tn: 295 - precision: 0.6224 - recall: 0.6845 - accuracy: 0.4836 - f1-score: 0.6520
L-Degree   tp: 95 - fp: 60 - fn: 42 - tn: 95 - precision: 0.6129 - recall: 0.6934 - accuracy: 0.4822 - f1-score: 0.6507
L-Designation tp: 314 - fp: 150 - fn: 98 - tn: 314 - precision: 0.6767 - recall: 0.7621 - accuracy: 0.5587 - f1-score: 0.7169
L-Name     tp: 133 - fp: 8 - fn: 13 - tn: 133 - precision: 0.9433 - recall: 0.9110 - accuracy: 0.8636 - f1-score: 0.9269
Name       tp: 133 - fp: 8 - fn: 13 - tn: 133 - precision: 0.9433 - recall: 0.9110 - accuracy: 0.8636 - f1-score: 0.9269
U-Degree   tp: 31 - fp: 9 - fn: 34 - tn: 31 - precision: 0.7750 - recall: 0.4769 - accuracy: 0.4189 - f1-score: 0.5905
U-Designation tp: 12 - fp: 6 - fn: 25 - tn: 12 - precision: 0.6667 - recall: 0.3243 - accuracy: 0.2791 - f1-score: 0.4363
2019-06-15 23:48:49,511 ----------------------------------------------------------------------------------------------------
{'dev_loss_history': [tensor(1.5428, device='cuda:0'),
  tensor(1.1334, device='cuda:0'),
  tensor(0.9405, device='cuda:0'),
  tensor(0.8493, device='cuda:0'),
  tensor(0.7548, device='cuda:0'),
  tensor(0.7112, device='cuda:0'),
  tensor(0.6841, device='cuda:0'),
  tensor(0.6598, device='cuda:0'),
  tensor(0.7170, device='cuda:0'),
  tensor(0.6312, device='cuda:0'),
  tensor(0.6189, device='cuda:0'),
  tensor(0.6253, device='cuda:0'),
  tensor(0.6076, device='cuda:0'),
  tensor(0.6262, device='cuda:0'),
  tensor(0.5838, device='cuda:0'),
  tensor(0.5864, device='cuda:0'),
  tensor(0.5865, device='cuda:0'),
  tensor(0.5927, device='cuda:0'),
  tensor(0.5681, device='cuda:0'),
  tensor(0.5694, device='cuda:0'),
  tensor(0.5655, device='cuda:0'),
  tensor(0.5638, device='cuda:0'),
  tensor(0.5658, device='cuda:0'),
  tensor(0.5476, device='cuda:0'),
  tensor(0.5525, device='cuda:0'),
  tensor(0.5481, device='cuda:0'),
  tensor(0.5441, device='cuda:0'),
  tensor(0.5460, device='cuda:0'),
  tensor(0.5467, device='cuda:0'),
  tensor(0.5485, device='cuda:0'),
  tensor(0.5474, device='cuda:0'),
  tensor(0.5471, device='cuda:0'),
  tensor(0.5466, device='cuda:0'),
  tensor(0.5451, device='cuda:0'),
  tensor(0.5429, device='cuda:0'),
  tensor(0.5430, device='cuda:0'),
  tensor(0.5427, device='cuda:0'),
  tensor(0.5423, device='cuda:0'),
  tensor(0.5419, device='cuda:0'),
  tensor(0.5421, device='cuda:0'),
  tensor(0.5422, device='cuda:0'),
  tensor(0.5421, device='cuda:0')],
 'dev_score_history': [0.3187,
  0.3573,
  0.5404,
  0.5424,
  0.561,
  0.6157,
  0.6174,
  0.6187,
  0.5924,
  0.6522,
  0.6398,
  0.6102,
  0.6358,
  0.6493,
  0.6436,
  0.6653,
  0.6456,
  0.6574,
  0.6584,
  0.6538,
  0.654,
  0.6638,
  0.6604,
  0.6583,
  0.6653,
  0.6494,
  0.6611,
  0.6549,
  0.6548,
  0.6541,
  0.6528,
  0.6555,
  0.6611,
  0.6604,
  0.6534,
  0.6646,
  0.6521,
  0.6618,
  0.6618,
  0.6646,
  0.6646,
  0.6632],
 'test_score': 0.6193,
 'train_loss_history': [3.255321723770122,
  1.7709290652858967,
  1.4236455041737783,
  1.2540150868649385,
  1.1619951228300731,
  1.0922388153619507,
  1.0411130577123084,
  1.0160721548053684,
  0.9830983853867264,
  0.9526460391729057,
  0.9443904631295983,
  0.9124478626818884,
  0.8975158451872618,
  0.8846787135211789,
  0.8375652415715918,
  0.8323883786797523,
  0.810026232077151,
  0.8117286653137531,
  0.8071514174950366,
  0.7868298142134738,
  0.7720122318588146,
  0.7725460075196766,
  0.7754052099423344,
  0.7734450001497658,
  0.7553771944577191,
  0.7538481478889784,
  0.7450204017717822,
  0.7480199431278267,
  0.7443713550867678,
  0.7366625928909195,
  0.748002025140386,
  0.7357378089103569,
  0.7506221494301647,
  0.7333263586573049,
  0.7418370548965169,
  0.7365644636000095,
  0.745652653885131,
  0.7420576044998202,
  0.7359075135722453,
  0.7475405704407465,
  0.7226628911535756,
  0.7301050403288433]}
  ```

### timings

- beginning: 2:30min / 1 epoch

## WordEmbeddings-glove-CharacterEmbeddings (1 run)

### basic info

- sampling:     `1.0`
- fullpath:     `/content/gdrive/My Drive/SAKI_2019/data/resources/tagger/resume-WEglove-CE`
- epochs:       `42`
- size on disk: `357 MB`

```python
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('glove'),
    # comment in this line to use character embeddings
    CharacterEmbeddings(),

    # comment in these lines to use flair embeddings (needs a LONG time to train :-)
    #FlairEmbeddings('news-forward'),
    #FlairEmbeddings('news-backward'),
]
```

### output

```
2019-06-16 03:48:48,898 ----------------------------------------------------------------------------------------------------
2019-06-16 03:48:48,901 Testing using best model ...
2019-06-16 03:48:48,910 loading file resources/taggers/resume-ner-WordEmbeddings-glove-CharacterEmbeddings/best-model.pt
2019-06-16 03:49:41,743 0.6963  0.5781  0.6317
2019-06-16 03:49:41,745 
MICRO_AVG: acc 0.4617 - f1-score 0.6317
MACRO_AVG: acc 0.5123 - f1-score 0.6342222222222222
-          tp: 12 - fp: 0 - fn: 464 - tn: 12 - precision: 1.0000 - recall: 0.0252 - accuracy: 0.0252 - f1-score: 0.0492
Degree     tp: 102 - fp: 71 - fn: 39 - tn: 102 - precision: 0.5896 - recall: 0.7234 - accuracy: 0.4811 - f1-score: 0.6497
Designation tp: 300 - fp: 184 - fn: 131 - tn: 300 - precision: 0.6198 - recall: 0.6961 - accuracy: 0.4878 - f1-score: 0.6557
L-Degree   tp: 98 - fp: 60 - fn: 39 - tn: 98 - precision: 0.6203 - recall: 0.7153 - accuracy: 0.4975 - f1-score: 0.6644
L-Designation tp: 317 - fp: 157 - fn: 95 - tn: 317 - precision: 0.6688 - recall: 0.7694 - accuracy: 0.5571 - f1-score: 0.7156
L-Name     tp: 136 - fp: 4 - fn: 10 - tn: 136 - precision: 0.9714 - recall: 0.9315 - accuracy: 0.9067 - f1-score: 0.9510
Name       tp: 136 - fp: 5 - fn: 10 - tn: 136 - precision: 0.9645 - recall: 0.9315 - accuracy: 0.9007 - f1-score: 0.9477
U-Degree   tp: 38 - fp: 12 - fn: 27 - tn: 38 - precision: 0.7600 - recall: 0.5846 - accuracy: 0.4935 - f1-score: 0.6609
U-Designation tp: 12 - fp: 9 - fn: 25 - tn: 12 - precision: 0.5714 - recall: 0.3243 - accuracy: 0.2609 - f1-score: 0.4138
2019-06-16 03:49:41,754 ----------------------------------------------------------------------------------------------------
{'dev_loss_history': [tensor(1.6044, device='cuda:0'),
  tensor(1.1632, device='cuda:0'),
  tensor(0.8955, device='cuda:0'),
  tensor(0.7346, device='cuda:0'),
  tensor(0.6852, device='cuda:0'),
  tensor(0.6602, device='cuda:0'),
  tensor(0.6755, device='cuda:0'),
  tensor(0.6163, device='cuda:0'),
  tensor(0.6243, device='cuda:0'),
  tensor(0.5889, device='cuda:0'),
  tensor(0.6167, device='cuda:0'),
  tensor(0.5941, device='cuda:0'),
  tensor(0.5837, device='cuda:0'),
  tensor(0.5541, device='cuda:0'),
  tensor(0.5650, device='cuda:0'),
  tensor(0.5572, device='cuda:0'),
  tensor(0.5561, device='cuda:0'),
  tensor(0.5323, device='cuda:0'),
  tensor(0.5243, device='cuda:0'),
  tensor(0.5329, device='cuda:0'),
  tensor(0.5386, device='cuda:0'),
  tensor(0.5106, device='cuda:0'),
  tensor(0.5123, device='cuda:0'),
  tensor(0.5187, device='cuda:0'),
  tensor(0.5125, device='cuda:0'),
  tensor(0.5064, device='cuda:0'),
  tensor(0.5037, device='cuda:0'),
  tensor(0.5050, device='cuda:0'),
  tensor(0.5064, device='cuda:0'),
  tensor(0.4925, device='cuda:0'),
  tensor(0.4936, device='cuda:0'),
  tensor(0.4863, device='cuda:0'),
  tensor(0.4874, device='cuda:0'),
  tensor(0.4858, device='cuda:0'),
  tensor(0.4821, device='cuda:0'),
  tensor(0.4906, device='cuda:0'),
  tensor(0.4815, device='cuda:0'),
  tensor(0.4873, device='cuda:0'),
  tensor(0.4821, device='cuda:0'),
  tensor(0.4786, device='cuda:0'),
  tensor(0.4773, device='cuda:0'),
  tensor(0.4812, device='cuda:0')],
 'dev_score_history': [0.1901,
  0.4909,
  0.5245,
  0.5969,
  0.6186,
  0.6162,
  0.5941,
  0.6375,
  0.6331,
  0.6415,
  0.6328,
  0.6389,
  0.6356,
  0.6694,
  0.6625,
  0.6625,
  0.6292,
  0.6639,
  0.6688,
  0.6721,
  0.6498,
  0.674,
  0.6777,
  0.687,
  0.6673,
  0.6855,
  0.6873,
  0.686,
  0.694,
  0.679,
  0.6727,
  0.666,
  0.6472,
  0.6949,
  0.6834,
  0.6804,
  0.6916,
  0.6885,
  0.69,
  0.6847,
  0.6889,
  0.6903],
 'test_score': 0.6317,
 'train_loss_history': [4.188379483969033,
  1.7424931075094507,
  1.4048353272433183,
  1.2122774146953408,
  1.0964671455222328,
  1.0215699490742618,
  1.0034325493334912,
  0.9497258414824804,
  0.926460910613845,
  0.8827051403267043,
  0.8706765283836799,
  0.8768464563046994,
  0.8143197548936825,
  0.8184385377736318,
  0.8084518358516856,
  0.7941822022402367,
  0.7700404757950582,
  0.7658949381336063,
  0.739802555719606,
  0.7086938733653146,
  0.7147943204661615,
  0.7072845356804984,
  0.7110646194746705,
  0.7189580633425389,
  0.687715104394624,
  0.6767995175557072,
  0.6902183733728467,
  0.6771903002039105,
  0.6706525424913484,
  0.6651259780842431,
  0.6704198183352444,
  0.6572568166722246,
  0.6587109565734863,
  0.6403535186838941,
  0.6279138584627586,
  0.6240889998723049,
  0.6350216556365799,
  0.6204545362567415,
  0.620329450790574,
  0.6077990245048691,
  0.6114903189191202,
  0.6084115952760184]}
```

### timings

- beginning: 5:30min / 1 epoch
- run totoal: 3:55h

## resume-WEglove-CE-flairNF (1 run)

### basic info

- sampling:     `1.0`
- fullpath:     `/content/gdrive/My Drive/SAKI_2019/data/resources/tagger/resume-WEglove-CE-flairNF`
- epochs:       `42`
- size on disk: `568 MB`

```python
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('glove'),
    # comment in this line to use character embeddings
    CharacterEmbeddings(),

    # comment in these lines to use flair embeddings (needs a LONG time to train :-)
    FlairEmbeddings('news-forward'),
    #FlairEmbeddings('news-backward'),
]
```

### output

```
2019-06-16 20:09:59,640 Testing using best model ...
2019-06-16 20:09:59,648 loading file resources/taggers/resume-WEglove-CE-flairNF/best-model.pt
2019-06-16 20:11:39,512 0.745   0.656   0.6977
2019-06-16 20:11:39,514 
MICRO_AVG: acc 0.5357 - f1-score 0.6977
MACRO_AVG: acc 0.5645 - f1-score 0.6921222222222222
-          tp: 150 - fp: 73 - fn: 326 - tn: 150 - precision: 0.6726 - recall: 0.3151 - accuracy: 0.2732 - f1-score: 0.4292
Degree     tp: 96 - fp: 54 - fn: 45 - tn: 96 - precision: 0.6400 - recall: 0.6809 - accuracy: 0.4923 - f1-score: 0.6598
Designation tp: 313 - fp: 136 - fn: 118 - tn: 313 - precision: 0.6971 - recall: 0.7262 - accuracy: 0.5520 - f1-score: 0.7114
L-Degree   tp: 92 - fp: 43 - fn: 45 - tn: 92 - precision: 0.6815 - recall: 0.6715 - accuracy: 0.5111 - f1-score: 0.6765
L-Designation tp: 318 - fp: 105 - fn: 94 - tn: 318 - precision: 0.7518 - recall: 0.7718 - accuracy: 0.6151 - f1-score: 0.7617
L-Name     tp: 142 - fp: 3 - fn: 4 - tn: 142 - precision: 0.9793 - recall: 0.9726 - accuracy: 0.9530 - f1-score: 0.9759
Name       tp: 142 - fp: 3 - fn: 4 - tn: 142 - precision: 0.9793 - recall: 0.9726 - accuracy: 0.9530 - f1-score: 0.9759
U-Degree   tp: 42 - fp: 18 - fn: 23 - tn: 42 - precision: 0.7000 - recall: 0.6462 - accuracy: 0.5060 - f1-score: 0.6720
U-Designation tp: 11 - fp: 12 - fn: 26 - tn: 11 - precision: 0.4783 - recall: 0.2973 - accuracy: 0.2245 - f1-score: 0.3667
2019-06-16 20:11:39,517 ----------------------------------------------------------------------------------------------------
{'dev_loss_history': [tensor(1.7569, device='cuda:0'),
  tensor(0.9845, device='cuda:0'),
  tensor(0.7844, device='cuda:0'),
  tensor(0.7614, device='cuda:0'),
  tensor(0.7171, device='cuda:0'),
  tensor(0.6607, device='cuda:0'),
  tensor(0.6322, device='cuda:0'),
  tensor(0.6225, device='cuda:0'),
  tensor(0.6085, device='cuda:0'),
  tensor(0.5988, device='cuda:0'),
  tensor(0.6253, device='cuda:0'),
  tensor(0.5757, device='cuda:0'),
  tensor(0.5674, device='cuda:0'),
  tensor(0.5638, device='cuda:0'),
  tensor(0.5307, device='cuda:0'),
  tensor(0.5489, device='cuda:0'),
  tensor(0.6114, device='cuda:0'),
  tensor(0.5197, device='cuda:0'),
  tensor(0.5526, device='cuda:0'),
  tensor(0.5487, device='cuda:0'),
  tensor(0.5356, device='cuda:0'),
  tensor(0.5428, device='cuda:0'),
  tensor(0.5273, device='cuda:0'),
  tensor(0.5242, device='cuda:0'),
  tensor(0.5380, device='cuda:0'),
  tensor(0.5530, device='cuda:0'),
  tensor(0.5191, device='cuda:0'),
  tensor(0.5229, device='cuda:0'),
  tensor(0.5226, device='cuda:0'),
  tensor(0.5036, device='cuda:0'),
  tensor(0.4972, device='cuda:0'),
  tensor(0.5023, device='cuda:0'),
  tensor(0.5050, device='cuda:0'),
  tensor(0.5154, device='cuda:0'),
  tensor(0.4958, device='cuda:0'),
  tensor(0.5078, device='cuda:0'),
  tensor(0.4998, device='cuda:0'),
  tensor(0.5210, device='cuda:0'),
  tensor(0.5411, device='cuda:0'),
  tensor(0.5145, device='cuda:0'),
  tensor(0.5017, device='cuda:0'),
  tensor(0.5067, device='cuda:0')],
 'dev_score_history': [0.4856,
  0.562,
  0.6299,
  0.6285,
  0.6156,
  0.6559,
  0.671,
  0.6723,
  0.6705,
  0.6735,
  0.663,
  0.6739,
  0.6749,
  0.6798,
  0.6735,
  0.6924,
  0.6725,
  0.6649,
  0.7021,
  0.6786,
  0.6835,
  0.7069,
  0.6901,
  0.7202,
  0.6973,
  0.6966,
  0.7189,
  0.7071,
  0.7196,
  0.7183,
  0.7172,
  0.7241,
  0.7321,
  0.7232,
  0.7141,
  0.7267,
  0.7415,
  0.7252,
  0.7182,
  0.7269,
  0.7022,
  0.7225],
 'test_score': 0.6977,
 'train_loss_history': [3.575308448603364,
  1.4825447388753599,
  1.1589316093191808,
  1.0416070507699942,
  0.9462972197200166,
  0.8748105055063354,
  0.8369528261475823,
  0.7998476308219287,
  0.7770417261184478,
  0.7516868025267205,
  0.7227398388239802,
  0.7090962926546732,
  0.7026245058799276,
  0.6584393509796688,
  0.6570246937830432,
  0.6454020078693118,
  0.648033370016789,
  0.6040513772948258,
  0.5991235014025856,
  0.592962888430576,
  0.589417573140592,
  0.5492035813477575,
  0.5731183659462702,
  0.5569822637199544,
  0.5273733180092306,
  0.5358109131049947,
  0.5208045761702823,
  0.5138587683033781,
  0.4717811816910497,
  0.46584890777764676,
  0.462260037255125,
  0.4413157703012836,
  0.447828268467569,
  0.44749523298878247,
  0.43085856854814253,
  0.4339737211968623,
  0.4350613414430294,
  0.4208426458292267,
  0.4234564659648201,
  0.412040372233407,
  0.4125245247340324,
  0.3977503267173864]}
```

### timings

- beginning: 9min / 1 epoch
- run totoal: 6:15h

## resume-WEglove-CE-flairNF-flairNB (1 run)

### basic info

- sampling:     `1.0`
- fullpath:     `/content/gdrive/My Drive/SAKI_2019/data/resources/tagger/resume-WEglove-CE-flairNF-flairNB`
- epochs:       `42`
- size on disk: `568 MB`

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