# Frasal
Il repo contiene il foglio eseguibile su colab per l'analisi del dataset e l'addestramento del modello per la classificazione automatica di eventi di malfunzionamento.

Per avviare il foglio è sufficiente cliccare sul [foglio colab](https://colab.research.google.com/github/leptoquark/frasal/blob/main/frasal.ipynb)

Una volta avviato, è possibile estrarre il modello in formato joblib, utilizzabile in codice custom.


```
from joblib import load

predict_clf_01 = load('frasal_model_D1.joblib')
predict_clf    = load('frasal_model_D2.joblib')

st = [7.5,12.5,16.5,26.5,26.5,4.2,105,115,3.5]

val = predict_clf_01.predict([st])[0]

if (val=='P'):
  print(predict_clf.predict([st]))
else:
  print(val);
```

in $main.py$ è present eil codice per esportare la funzionalità di addestramento e predizione in flask
