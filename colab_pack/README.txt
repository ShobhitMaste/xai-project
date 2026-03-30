Colab upload pack (lightweight)
==============================

This folder holds everything needed for Google Colab training — no node_modules,
no exported Colab weights, and an empty backend/model/saved until you train.

Refresh contents from your full project
---------------------------------------
From XAI/project (parent of colab_pack):

  python colab_pack/build_colab_pack.py

Zip and upload
--------------
Zip THIS ENTIRE FOLDER (colab_pack) as e.g. colab_pack.zip, upload to the notebook
cell, then run Train_Stress_Detection_Colab.ipynb.

What is included
----------------
- backend/     Python API + training code; model/saved/.gitkeep only
- frontend/    package.json, public/, src/ (run npm install locally when you need UI)
- resources/   CSVs and reference material
- Train_Stress_Detection_Colab.ipynb
- colab_train_stress_all_in_one.py (optional single-file trainer)
