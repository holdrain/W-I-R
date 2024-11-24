# W-I-R
Official code of WIR

This project is made of three modules: attack,certify,and train

1. Attack: implement identity linking attack, identity forge attack, identity extract attack

2. Certify: certify watermark model

3. Train: implement watermark model(stegastamp and hidden) train.


Train watermark model or mutal information version watermark model

```
    bash scripts/stega/train.sh
```

Certity watermark model
```
    bash scripts/stega/certify.sh
```

Attack watermark model
```
You can implement identity linking attack by run attack/KNN.ipynb
You can implement identity forge attack by run attack/KNN_forge.ipynb
You can implement identity extract attack by run attack/extract_attack.ipynb
```



