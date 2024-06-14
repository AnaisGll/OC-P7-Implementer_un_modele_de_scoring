# OC-P7-Umplémentez-un-modèle-de-scoring
Implémentez un modèle de scoring

## Contexte

Pour ce projet, je suis Data Scientist au sein d'une société financière, nommée "Prêt à dépenser", qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt. L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).
De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’accord de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.
Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

## Données 

Les données sont issues de Kaggle et sont disponibles à l'adresse suivant : 
https://www.kaggle.com/c/home-credit-default-risk/data

## Missions 

1. Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
2. Analyser les features qui contribuent le plus au modèle, d’une manière générale (feature importance globale) et au niveau d’un client (feature importance locale), afin, dans un soucis de transparence, de permettre à un chargé d’études de mieux comprendre le score attribué par le modèle.
3. Mettre en production le modèle de scoring de prédiction à l’aide d’une API et réaliser une interface de test de cette API.
4. Mettre en œuvre une approche globale MLOps de bout en bout, du tracking des expérimentations à l’analyse en production du data drift.

## Construction 

1. Le notebook comportant l'analyse exploratoire des données, la création de features engineering et la sélection de features ainsi que l'entraînement, d'optimisation et de sélection de modèle : Guille_Anais_1_modelisation.ipynb
2. Le script python des fonctions utilisées dans le notebook : fonctions.py
3. Le script python de la configuration de l’API : api.py
4. Le script python de la configuration du dashboard pour Streamlit : dashboard.py
5. Les scripts python des tests unitaires réalisés avec Pytest pour l’API et le dashboard : test_api.py et test_dashboard.py
6. Le fichier listant les packages utilisés : requirements.txt

## API en production :

l'API en production est disponible à l’adresse suivante : 
[API déployée](https://pret-a-depenser.azurewebsites.net)

## Dashboard en production :

Le dashboard streamlit en production est disponible à l’adresse suivante :
[Dashboard Streamlit](https://pret-a-depenser.streamlit.app/)
