# Human Detection for SUAS 2026

### Overview

Main goal of this repo is to store notes, documentation and scripts connected to training, testing and dateset preparation for SUAV 2026 competition. Here the main pivot is human detection task. The theme of the competition is SAR (Search and Rescue) after natural disaster (tornado). 

### Mission 

One mannequin and one tent will be scatered in the search boundary. Minimal altitude is 150 feet (45,72m). The mission is to detect this objects and deliver (drop) payloads (water bottle -> mannequin and beacon -> tent). The allowed radius is 50ft (15,24m). Precise scoring:
- Object Survives (Within Vicinity of Search Boundary) = 20 Points
- Object Lands within 50' of an Target = 50 Points
- Object Delivered to the Correct Target (Water Bottle to Mannequin + Beacon to Tent) = 30 Points

### Where the model will be deployed

__Jetson Orin Nano__

### Sketch of CONOPS (concept of operations)
```
while(time is not up)
    models analyzing frames from camera # question are wo going to do this one by one (first tent than mannequin or both at the same time)
    if model detect:
        calculate the cords of the objcect
        calcualte drop point
        modify the flight path
        drop payload when reaches drop point
```

Questions:
- Are we going to set a treshold like model needs to detect the object in the same place at least for 3 frames or sth like that?
- What the error margin of calculating object position from camera frame?
- Are we going to calculate object position from more than one frame (take average for example)?

## Human Detection

This part covers most of this repo is made for.

### Datasets

- NOMAD -> Natural, Occluded, Multi-scale, Aerial Dataset. Contains images from 5 different heights (10m, 30m ,50m, 70m, 90m) and 10 different levels of occlusion. Made by filming 100 different human actors from drone across 100 different sceneries. (42k images)
- C2A -> Combination to Application dataset, synthetic dataset made by pasting humans into disaster backgorunds (10k images)
- SARD -> images taken by dron of real people with different positions "the authors involved actors, who simulated exhausted and injured people" (5.5k images) (rather lower altitudes <50m)
- WISARD -> large dataset with SAR images (28k images), also contains termovision images
- HERIDAL -> humans from drone (1600 images) (3000x4000 res)

### CV models training platforms

- kaggle 30h/week, 2x15gb gpu, a bit slow cpu, datasets, notebooks
- roboflow, 15 credits a month on free plan (premium 99used), training times similar to kaggle, nice interface, low code, on free plan very limited access to models
- ultralytics hub, 5usd free credits, very fast, you can use 96gb GPU, access to their modesl (wide)
- modal, you can get 30usd of credits but you have to pay verification fee 0.5usd and give them you card data
- google colab, weaker than kaggle, rather need pro for sensible work, still not worth probably