Script MATLAB pentru analiza și clasificarea datelor de hepatită C folosind Random Forest și Multi-Layer Perceptron. Compară performanțele modelelor pe diferite split-uri train/test și evaluează impactul selecției de features.
Structura Codului
# 1. Încărcarea și Pregătirea Datelor
Z = readtable('HepatitisCdata.csv')

● Input: Fișier CSV cu date hepatită C
Procesare:

●	Extrage features (X) și target (Y)

●	Salvează numele features înainte de conversie

●	Convertește toate coloanele la numeric (categorice → indici)

●	Creează target binar: Donor (0) vs Disease (1)

●	Înlocuiește NaN cu 0
# 2. Analiza 
Dependențe între variabile:
●	Calculează matricea de corelație între features

●	Identifică dependențe liniare puternice (|corr| > 0.9)
Distribuția claselor:

●	Afișează tabele pentru clasificare multiclasă și binară

●	Vizualizează dezechilibrul claselor
Statistici descriptive:
●	Calculează: mean, variance, min, max, range pentru fiecare feature

●	Generează histograme pentru toate features

●	Afișează boxplot-uri pentru features cu variație mare
Detectarea outlierilor:
●	Z-score: Identifică valori cu |z| > 3

●	IQR: Detectează valori sub Q1-1.5×IQR sau peste Q3+1.5×IQR
# 3. Evaluarea Split-urilor (80/20, 70/30, 60/40, 50/50)
Pentru fiecare split:
Random Forest:
rf = TreeBagger(250, XTrain, YTrain, 'MinLeafSize', 5)
●	250 arbori

●	Out-of-Bag validation

●	Feature importance calculat automat
MLP:
net = patternnet(32, 'trainscg')
●	32 neuroni în layer ascuns

●	Antrenare cu Scaled Conjugate Gradient

●	Z-score standardizare

●	Oversampling clasa minoritară pentru echilibrare

●	Cross-entropy loss, max 300 epoci
Metrici evaluate:

●	Accuracy, Precision, Recall, F1-score, AUC

●	Matrice de confuzie (TN, FP, FN, TP)
# 4. Feature Selection (opțional)
ReliefF:
[idxRelief, wRelief] = relieff(X, Ybin, K=10)
●	Ranking features după importanță

●	Selecție top 6 features

●	Comparație RF/MLP: all features vs top features

Funcții Auxiliare
buildDatasets(X, Y, pTest, pVal)
Scop: Împarte dataset-ul în train/test(/validation)
Parametri:
●	X: matrice features (N × d)

●	Y: vector target (N × 1)

●	pTest: proporție test (ex: 0.2 = 20%)

●	pVal: proporție validare (opțional)

Return: XTrain, YTrain, XTest, YTest, XVal, YVal
evalBinary(yTrue, yPred, yScorePos)

Scop: Calculează metrici clasificare binară

Return: struct cu Confusion matrix, Accuracy, Precision, Recall, F1, AUC
oversampleMinority(Xtrain, Ytrain)

Scop: Echilibrează clasele prin random oversampling

Metodă: Duplică random sample-uri din clasa minoritară până se egalează cu clasa majoritară
showTableFig(T, figTitle)

Scop: Afișează table MATLAB într-o fereastră GUI
## Output-uri Principale
1.	Tabele comparative: Distribuții clase, statistici, outlieri
2.	Grafice: Histograme, boxplots, matrice corelație
3.	Rezultate performanță: Tabel cu Accuracy/Precision/Recall/F1/AUC pentru toate split-urile și modele
4.	Feature importance: Ranking ReliefF și comparație pre/post selecție
## Concluzii Integrate
Random Forest:

●	 Robust, nu necesită scalare, interpretabil

●	 Recall mai mic pe clase dezechilibrate
MLP:

●	 Recall mai bun cu oversampling, modelează relații complexe

●	 Necesită preprocesare, sensibil la hiperparametri

Feature Selection:

●	Reduce dimensionalitate, crește viteza

●	 Risc de pierdere informație

