# Évaluation du modèle EMOTYC

Évaluation du modèle [EMOTYC](https://huggingface.co/TextToKids/CamemBERT-base-EmoTextToKids) sur le corpus [CyberAgression-Large](https://github.com/aollagnier/CyberAgression-Large) qui contient des messages de cyberharcèlement en français, rédigés par des jeunes entre 11 et 18 ans.

## Exemple d'utilisation

```
pip install -r requirements.txt
```

```
cd EvalEMOTYC
```

```bash
python emotyc_predict.py --xlsx ./golds/CyberAdoAgg_gold_global_total.xlsx --out_dir ./results/CyberAggAdo/ContextTemplate --use-context
```
Ce qui lançe une inférence avec le templace BCA en ajoutant les phrases adjacentes. Ce script d'inférence a été conçu et testé sur GPU NVIDIA et testés sur Tesla T4 et Jetson Orin NX.





### Les 19 labels

Le modèle produit un vecteur de 19 logits, organisés en 4 groupes :

<br>
<p align="center">
  <img src="emotyc_output_vector.svg" width="700">
</p>

# Trois types d'émotions

Le schéma d'annotation distingue trois types d'émotions :

<br>
<p align="center">
  <img src="types_emotions.svg" width="700">
</p>



> **Note** : Les noms des colonnes dans les fichiers gold utilisent les accents français (ex. `Colère`, `Dégoût`, `Fierté`, `Désignée`, `Montrée`, `Suggérée`), tandis que les labels internes du modèle sont en ASCII. Les scripts gèrent automatiquement ce mapping.



## Remarques

On a testé différentes configurations, notamment en enlevant ou non l'espace après "current". Il semble que le dépôt huggingface documente :

```
before:{prev}</s>current: {s}</s>after:{next}</s>
```

Mais on teste aussi :

```
before:{prev}</s>current:{s}</s>after:{next}</s>
```
