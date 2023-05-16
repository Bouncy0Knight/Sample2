# Sample2

This is one of my earlier examples of data-analysis and exploration. For this purpose, I used the nursery dataset and implemented Kmodes and Kprototypes-based clustering.

The Kmodes testing file reads in the Nursery data frame and clusters through KModes. The results of the clustering algorithm are analyzed through silhouette scoring.

The baseCode file reads in the Nursery data frame and clusters through Kprototypes. The accuracy of the clustering is tested internally, through the element class, which compares the results of the clustering algorithm with the predetermined answers. The clustering is further tested through silhouette scoring. 

NOTE: for the program to run, it may be necessary to pip install (via terminal) certain packages:

Package           Version
----------------- -------
contourpy         1.0.7
cycler            0.11.0
et-xmlfile        1.1.0
filelock          3.11.0
fonttools         4.39.3
Jinja2            3.1.2
joblib            1.2.0
kiwisolver        1.4.4
kmodes            0.12.2
MarkupSafe        2.1.2
matplotlib        3.7.1
mpmath            1.3.0
networkx          3.1
numpy             1.24.2
openpyxl          3.1.2
packaging         23.1
pandas            2.0.1
Pillow            9.5.0
pip               23.1.2
pyparsing         3.0.9
python-dateutil   2.8.2
pytz              2023.3
scikit-learn      1.2.2
scipy             1.10.1
setuptools        57.4.0
six               1.16.0
sympy             1.11.1
threadpoolctl     3.1.0
torch             2.0.0
typing_extensions 4.5.0
tzdata            2023.3
