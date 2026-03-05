# NextLoc-dashboard
Dashboard to inform next touring location for electronic music artists (from NextLoc GRU)

## Project overview
This dashboard is an early prototype of an ambitious project aimed at informing touring strategy for music artists and industry stakeholders. The current version predicts likely next locations based on past touring patterns, while the wider vision is to combine factors such as carbon footprint, gig attendance, and local fanbase. The goal is to create a dynamic tool that can support sustainable, scene-centric touring decisions for a variety of music actors.

![Dashboard Screenshot](screenshot_dashboard.png)

## Download
fork or dowload .zip

## Install
run : pip install -r requirements.txt

## Run
run : python NextLoc-dashboard.py

open in browser : http://127.0.0.1:8050

## Model documentation
Gated Reccurent Unit Neural Network trained on 2.5M+ clubbing events in 560+ cities, from Resident Advisor

Benchmark to Order-2 Markov chain:
- top-1 acc. 55% (GRU) ; 48% (Markov)
- top-10 acc. 80% (GRU) ; 70% (Markov)

Training data sampling strategy:
- select artists who performed at least once in 2025 in any of the following 10 cities: Berlin, Paris, Amsterdam, London, Barcelona, Lisbon, Mexico City, Los Angeles, Toronto.
- for these artists, include their full touring history in the training set.

## Implications & Biases
The sampling approach overrepresents two types of artists in the model:
- local artists who predominantly perform in one of these 10 cities, making the model more accurate for familiar, high-density western urban regions
- frequent international performers (“jet-setters”) who tour across multiple of these cities, potentially skewing predictions towards highly mobile artists and underrepresenting smaller or regional tours
- as a result, the model may be less accurate for artists whose touring patterns are concentrated outside these major cities or who follow atypical touring trajectories.

## Quirks
In the dropdown, selections can be cities, regions, or countries. A city corresponds exactly to that location. A region represents other cities within that region not individually listed, and a country represents other cities within that country. This reflects the spatial granularity of Resident Advisor data, where not every city has a dedicated page, so events are often aggregated at the regional or national level.
