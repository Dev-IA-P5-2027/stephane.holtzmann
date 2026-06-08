# Rapport de stage — BNDRS
## Plan détaillé (fin de 1ère année — DEV IA, Greta de Besançon)

> Calé sur la trame imposée par Bassam Kurdy (canal *Stages*, 13/04) :
> introduction + présentation entreprise → partie 1 (conception) →
> partie 2 (mise en œuvre) → partie 3 (bilan & perspectives) →
> conclusion → glossaire.
> **Présentation orale : 15 à 20 minutes.**
>
> Pour chaque section : **À écrire** (le contenu attendu) et **Preuve BNDRS**
> (le matériel concret — code, capture, fichier — à montrer). La vision long
> terme reste légère et placée en perspective, jamais au centre : le rapport
> prouve des compétences, il ne vend pas un rêve.

---

## 0. Introduction

**À écrire.** Qui tu es (reconversion, formation Développeur IA au Greta), le
cadre du stage (210 h, fin de 1ère année), et ta situation particulière —
assumée clairement : ton stage s'est déroulé **sur ton propre projet, BNDRS**.
Pose l'objet du rapport en une phrase, puis annonce le plan.

**Preuve BNDRS.** L'adresse live du projet : `https://bndrs.fr` (à citer dès
l'intro — un projet réellement déployé, pas une maquette).

---

## 1. Présentation de l'entreprise

**À écrire.** Comme tu es ton propre cadre de stage, présente **BNDRS en tant
que projet/structure** : sa nature (un moteur d'exploration cognitive et
narrative), sa mission, son positionnement (ce que c'est **et ce que ce n'est
pas** — *pas un fact-checker*), son stade actuel (alpha, démo publique en
ligne), et la façon dont le stage s'y est inséré. Reste honnête sur le statut
particulier : c'est une force (autonomie totale, bout-en-bout) à condition de
le cadrer.

**Preuve BNDRS.** `BNDRS_CHARTE_COGNITIVE.md`, `BNDRS_CONCEPT_AION_SPHERE.md`
(le positionnement « 3 niveaux, pas un fact-checker » est déjà figé) ; capture
de la page d'accueil bndrs.fr.

---

## 2. Première partie — Conception

### 2.1 Compréhension du besoin client
**À écrire.** Quel problème BNDRS adresse : aider à explorer un contenu sous
trois angles (faits / émotion / vision) et à en lire les tensions, là où les
outils existants tranchent (vrai/faux) au lieu d'explorer. Définis le
« client » (utilisateur cible) et le besoin auquel tu réponds.

**Preuve BNDRS.** La page *Projet* du site (qui explique la mécanique au
visiteur).

### 2.2 État de l'art
**À écrire.** Panorama des approches existantes : fact-checking, analyse de
sentiment (NLP), détection de désinformation, LLM génériques. Ce qu'ils font,
leurs limites, et le créneau que BNDRS occupe. *(C'est exactement la compétence
« veille / état de l'art » du référentiel — Module 9.)*

**Preuve BNDRS.** Ta veille (sources, comparatif) — à formaliser en 1 page.

### 2.3 Éléments de conception technique
**À écrire.** L'architecture d'ensemble : front public, back IA, base de
données, dashboard d'administration, infrastructure d'hébergement. Présente la
**charte cognitive** comme modèle conceptuel (3 entités — Lucida/faits,
Cordis/émotion, Auréa/vision — et le cercle chromatique).

**Preuve BNDRS.** Schéma d'architecture (je peux te le générer) ; le modèle de
données `AnalysisRecord` / `User` (`backend/app/db/models.py`).

### 2.4 Choix techniques
**À écrire.** Justifie chaque brique : **React + Vite** (front), **FastAPI**
(API Python), **SQLite** (base), **API OpenAI** (le service d'IA intégré),
**three.js** (la sphère 3D), **Caddy** (reverse proxy HTTPS), **auto-
hébergement** (Freebox/DNS). Pour chaque choix : pourquoi lui, quelle
alternative écartée.

**Preuve BNDRS.** `requirements`/`package.json`, le `Caddyfile`, les scripts
`.bat` de build/déploiement.

### 2.5 Réponse finale — ce qui a été réalisé
**À écrire.** Le livrable concret au terme du stage : un site en ligne, un
pipeline d'analyse fonctionnel, la visualisation AION, un dashboard
d'administration avec suivi des coûts, une gestion des accès par rôles.

**Preuve BNDRS.** Démo live + captures des pages Analyse et Résultat.

---

## 3. Seconde partie — Mise en œuvre du projet

### 3.1 Organisation technique & environnement de développement
**À écrire.** Structure du dépôt (`frontend`, `frontend-admin`, `backend`),
outils (Git, npm, Python/venv, l'éditeur), et le flux **build → déploiement**
(tes `.bat`, le passage en HTTPS via Caddy, la mise en ligne sur bndrs.fr).
Décris l'environnement « tout au long de la production », pas juste l'état
final.

**Preuve BNDRS.** Arborescence du projet ; `TOUT_VALIDER_ET_RELANCER.bat`
(build + déploiement + contrôle de santé).

### 3.2 Gestion de projet
**À écrire.** Comment tu as piloté : itérations successives, priorisation, et
surtout ton **tri permanent entre attendu pédagogique / bonus / R&D
personnelle** (ta façon de ne pas surdimensionner). C'est une compétence en soi
*(Module 19 — gestion de projet)*.

**Preuve BNDRS.** Ta liste de tâches / journal d'itérations (les roadmaps
chiffrées déjà produites).

### 3.3 Retours d'expérience — outils, techniques, compétences
**À écrire.** Ce que tu as appris **en faisant** : Python/async, conception
d'API, intégration d'un service d'IA (orchestration de plusieurs appels),
React, JWT et gestion des rôles, reverse proxy, prompt engineering, suivi des
coûts de tokens. Puis les **difficultés réelles et comment tu les as
résolues** — c'est ce qui impressionne un jury :
- délai d'analyse trop court → passage du timeout à 300 s ;
- plantage de la sphère 3D (variable non définie) → diagnostic via la console,
  correction ciblée ;
- modèles refusant un paramètre de température → adaptation + relances
  automatiques ;
- gestion des accès (admin / observateur / profs) → rôles et permissions.

**Preuve BNDRS.** Extraits de code commentés (orchestrateur d'analyse,
`model_router`, sécurité/rôles), captures console avant/après.

---

## 4. Troisième partie — Bilan & perspectives

**À écrire (bilan).** Ce qui fonctionne aujourd'hui, l'état alpha assumé, et —
surtout — la **liste des compétences que le projet démontre** (fais le lien
explicite avec ce que la formation visait cette année).

**À écrire (améliorations envisageables).** Hébergement 24/7 (Hetzner), tests
automatisés et chaîne de livraison continue, durcissement sécurité,
multimodal (image/vidéo/audio), modèle de participation payante.

**À écrire (perspectives — vision longue, légère).** Où tu vois BNDRS aller.
Une demi-page maximum : c'est l'ouverture, pas le cœur. Point central de
cette ouverture : **les agents**. Explique que l'étape suivante est de
développer des agents qui interviennent **en amont** de l'analyse, pour
**détecter des patterns** sur l'ensemble des sujets déjà traités et à venir
(c'est déjà commencé). Replace-les dans la structure d'ensemble : ce qui tourne
chez toi (PC, serveur NAS), ce qui est à l'extérieur (bndrs.fr, API OpenAI),
l'application en cours, et la collaboration visée avec MAPTYCS (Ernest Legrand).

**Preuve BNDRS.** La roadmap chiffrée en heures déjà rédigée + le **schéma
d'architecture** `SCHEMA_ARCHITECTURE_BNDRS.html` (actuel vs prévu) — à insérer
ici en image.

---

## 5. Conclusion

**À écrire.** Bilan personnel de la reconversion, ce que ce stage t'a apporté
concrètement, et l'ouverture vers la 2ème année (alternance). Ton, honnête et
mesuré.

---

## 6. Glossaire

**À écrire.** Définis les termes techniques cités : API, FastAPI, React, Vite,
LLM, OpenAI, JWT, reverse proxy, Caddy, SQLite, three.js, bloom, endpoint,
build, déploiement, CI/CD, MLOps. *(Je te le pré-remplis quand on rédige.)*

---

## Annexe — Présentation orale (15–20 min)

Mini-déroulé proposé, à valider plus tard :
1. **0–2 min** — qui tu es, le contexte, BNDRS en une phrase.
2. **2–5 min** — le besoin + l'état de l'art (le créneau).
3. **5–9 min** — l'architecture et les choix techniques.
4. **9–14 min** — **démo live** de bndrs.fr (analyse + sphère + admin).
5. **14–17 min** — difficultés rencontrées et résolues.
6. **17–20 min** — bilan, perspectives, questions.

> La démo en direct est ton meilleur atout : peu de stagiaires de 1ère année
> présentent un projet réellement déployé et accessible publiquement.

---

### Prochaine étape
Tu valides / ajustes ce plan (ordre, parties à renforcer ou alléger), puis on
rédige partie par partie. Dis-moi si une section doit être plus développée ou,
au contraire, raccourcie.
