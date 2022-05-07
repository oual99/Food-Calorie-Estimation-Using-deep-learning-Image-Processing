
## Table of Contents
1. [A propos](#A-propos)
2. [Réalisateurs](#R%C3%A9alisateurs)
3. [Technologies](#technologies)
4. [Utilisation](#utilisation)
5. [Collaboration](#collaboration)

### Réalisateurs
***
* **BOULAK Ibtissam**
* **AIT OUALLANE Abderrahmane**


### A Propos
***
Ce projet est le résultat d’un travail que nous avons réalisé dans le cadre de notre projet de fin d’études au sein de la Faculté des sciences Semlalia Marrakech.
Dans ce projet, nous avons essayé de réaliser une application de l'apprentissage profond pour la classification et la reconnaissance des aliments, et une implémentation de quelques méthodes de la segmentation des images dont le but est de donner une estimation du poids d’un aliment dans une image prise par l’utilisateur.
## Technologies
***
Liste des technologies utilisée dans ce projet:
* Partie du traitement:
```
  * [Python](https://www.python.org/): Version 3.7.0
  * Les bibliothèques utilisées sont cités dans le fichier 'requiremente.txt'
```
Pour installer  toutes les bibliothèques mentionées dans le fichier 'requiremente.txt' vous pouver utiliser la commande suivante:
```
	pip install -r requiremente.txt
```
* Pour l'interface graphique:
```
  * Html
  * CSS
  * JavaScirpt
  * Flask(backend)
```
## Utilisation
***
Pour utiliser l'application vous devez juste installer le langage de programmation et les bibliothèques qui ont déjà citées, et lancer la commande ``` python test.py ``` , après vous ouvrez le fichier index.html dans un navigateur et vous pouvez tester l'outil par les images qui ont dans le dossier test. 

## Description
***
Le dossier du projet contient les elements suivants:
> foodArea.py : Contient la fonction qui calcule la surface couverte par l'aliment et le pouce.
> foodCal.py: Contient les fonctions de calculent de calories.
> prediction.py: Contient les fonctions qui retournent la prédiction.
> test.py: Contient le code principal.
> cnn_zero: Contient la creation d'un CNN de zero.
> transfer_fine_tuning.py: Contient la création d'un CNN en utilisant Transfer learning/Fine tuning.
> saved_model_fine_tuning: Ce dossier contient le modèle enregistré.
> test: Vous pouvez utiliser les images qui ont dans ce dossier pour tester l'application.
> static : Dossier contient les fichiers du CSS, JS,.. 
> templates: Dossier contient le fichier HTML.
> dataset.zip: est le jeu de données utilisé dans ce projet.
> seg_img: Dossier contient les différntes étapes de la segmentation appliquées sur l'image saisie.