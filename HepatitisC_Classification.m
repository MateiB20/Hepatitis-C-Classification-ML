clear; close all; clc;
%% parameters
fileNameRawData='HepatitisCdata.csv';

%% loaded the dataset
Z = readtable(fileNameRawData);

% extracted X Y
Y = Z.Category;   
X = Z(:, 3:end);
featureNames = X.Properties.VariableNames'; 

Ymulti = grp2idx(categorical(Y));

% tinta binara: Donor(0) vs Disease(1) 
Ycat = categorical(Y);
isDonor = (Ycat=="0=Blood Donor") | (Ycat=="0s=suspect Blood Donor");
Ybin = double(~isDonor);

% conversie X la numeric
Xnum = zeros(size(X,1), width(X));  
for i = 1:width(X)
    colName = X.Properties.VariableNames{i};
    col = X.(colName);  
    if iscell(col) || isstring(col) || ischar(col) || iscategorical(col)
        Xnum(:,i) = grp2idx(categorical(col));
    else
        Xnum(:,i) = double(col);
    end
end
X = Xnum;

%% fixed missing values 
X(isnan(X)) = 0;
Ymulti(isnan(Ymulti)) = 0;
Ybin(isnan(Ybin)) = 0;

%% dependencies between input variables 
aux=eye(size(X,2));
for i=1:size(X,2)
    for j=2:size(X,2)
        aux(i,j)=abs(corr(X(:,j),X(:,i))); aux(j,i)=aux(i,j);
    end
end


%% distribution per classes
disp('Distributie clase (multiclas):'); tabulate(Ymulti)
disp('Distributie clase (binar 0=donor, 1=disease):'); tabulate(Ybin)

Tmulti = array2table(tabulate(Ymulti), 'VariableNames', {'Value','Count','Percent'});
Tbin   = array2table(tabulate(Ybin),   'VariableNames', {'Value','Count','Percent'});
showTableFig(Tmulti, 'Distributie clase (multiclas)');
showTableFig(Tbin,   'Distributie clase (binar 0/1)');

%% ststistics 
mPlot=3; 
nPlot=4; 
M = zeros(1, size(X,2));
STD = zeros(1, size(X,2)); 
VAR = zeros(1, size(X,2));
MAX = zeros(1, size(X,2)); 
MIN = zeros(1, size(X,2));
for k=1:size(X,2)
    idx=rem(k,mPlot*nPlot);
    if idx==0, idx=mPlot*nPlot; end
    if idx==1, figure; end
    subplot(mPlot,nPlot,idx), plot(X(:,k))
    histogram(X(:,k));
    grid on;
    title(featureNames{k}, 'Interpreter','none');
    xlabel('Value'); ylabel('Count');
    M(k)=mean(X(:,k));     
    STD(k)=std(X(:,k));
    VAR(k)=var(X(:,k));
    MAX(k)=max(X(:,k));    
    MIN(k)=min(X(:,k));    
end

featureNames = Z(:,3:end).Properties.VariableNames';
Tstats = table(featureNames, M', VAR', MIN', MAX', (MAX'-MIN'), ...
    'VariableNames', {'Feature','Mean','Variance','Min','Max','Range'});
showTableFig(Tstats, 'Statistici (mean/var/min/max)');

%% outlieri/discrepante (Z-score si IQR)
Zsc = abs((X - mean(X,1)) ./ std(X,[],1));
outZ = sum(Zsc > 3, 1);

outIQR = zeros(1,size(X,2));
loIQR  = zeros(1,size(X,2));
hiIQR  = zeros(1,size(X,2));
for j=1:size(X,2)
    q1 = quantile(X(:,j),0.25);
    q3 = quantile(X(:,j),0.75);
    iqrVal = q3-q1;
    lo = q1 - 1.5*iqrVal;
    hi = q3 + 1.5*iqrVal;
    loIQR(j)=lo; hiIQR(j)=hi;
    outIQR(j) = sum(X(:,j)<lo | X(:,j)>hi);
end
Tout = table(featureNames, outZ', outIQR', loIQR', hiIQR', ...
    'VariableNames', {'Feature','Outliers_Zscore','Outliers_IQR','IQR_Low','IQR_High'});
showTableFig(Tout, 'Outlieri (Z-score si IQR)');

%% indicatori suplimentari
[~,idxRange] = sort((MAX-MIN),'descend');
topShow = idxRange(1:min(6,numel(idxRange)));
figure; boxplot(X(:,topShow),'Labels',featureNames(topShow));
xtickangle(45); title('variabile cu variatie mare');

%% split datasets – verificare dimensiuni
pTest = 0.2; pVal  = 0;
[XTrain, YTrain, XTest, YTest, XVal, YVal] = buildDatasets(X, Ybin, pTest, pVal);
disp('Split 80/20 sizes:'), size(XTrain), size(XVal), size(XTest)

pTest = 0.3; pVal  = 0;
[XTrain, YTrain, XTest, YTest, XVal, YVal] = buildDatasets(X, Ybin, pTest, pVal);
disp('Split 70/30 sizes:'), size(XTrain), size(XVal), size(XTest)

pTest = 0.4;
[XTrain, YTrain, XTest, YTest, XVal, YVal] = buildDatasets(X, Ybin, pTest, pVal);
disp('Split 60/40 sizes:'), size(XTrain), size(XVal), size(XTest)

pTest = 0.5;
[XTrain, YTrain, XTest, YTest, XVal, YVal] = buildDatasets(X, Ybin, pTest, pVal);
disp('Split 50/50 sizes:'), size(XTrain), size(XVal), size(XTest)

%% comparatie RF vs MLP pe toate split-urile
rng(42,'twister');
pTests = [0.2 0.3 0.4 0.5];
results = [];

for pTest = pTests
    fprintf('\n\n');
    fprintf('Evaluare pe split Train %.0f%% / Test %.0f%%\n',(1-pTest)*100,pTest*100);

    [XTrain, YTrain, XTest, YTest] = buildDatasets(X, Ybin, pTest, 0);

    tTr = tabulate(YTrain);
    tTe = tabulate(YTest);
    fprintf('Distributie train: Donor=%d, Disease=%d\n', tTr(1,2), tTr(2,2));
    fprintf('Distributie test : Donor=%d, Disease=%d\n', tTe(1,2), tTe(2,2));

    nTrees = 250;
    rf = TreeBagger(nTrees, XTrain, categorical(YTrain), ...
        'Method','classification', ...
        'OOBPrediction','on', ...
        'OOBPredictorImportance','on', ...
        'MinLeafSize', 5);

    [yhatRF, scoreRF] = predict(rf, XTest);
    yhatRF = str2double(cellstr(yhatRF));
    idxPos = find(strcmp(rf.ClassNames,'1')); if isempty(idxPos), idxPos=2; end
    scoreRFpos = scoreRF(:,idxPos);

    metRF = evalBinary(YTest, yhatRF, scoreRFpos);

    [XTrainZ, mu, sig] = zscore(XTrain);
    sig(sig==0)=1;
    XTestZ = (XTest - mu) ./ sig;

    [Xos, Yos] = oversampleMinority(XTrainZ, YTrain);

    metMLP = struct('Accuracy',NaN,'Precision',NaN,'Recall',NaN,'F1',NaN,'AUC',NaN,'Confusion',[]);
    scoreMLPpos = NaN(size(YTest));
    try
        hidden = 32;
        net = patternnet(hidden,'trainscg');
        net.performFcn = 'crossentropy';
        net.trainParam.showWindow = false;
        net.divideFcn = 'dividetrain';
        net.trainParam.epochs = 300;
        net.trainParam.max_fail = 10;

        Ttrain = full(ind2vec(Yos'+1));
        net = train(net, Xos', Ttrain);

        Yprob = net(XTestZ')';
        scoreMLPpos = Yprob(:,2);
        yhatMLP = double(scoreMLPpos >= 0.5);

        metMLP = evalBinary(YTest, yhatMLP, scoreMLPpos);
    catch ME
        warning('MLP nu a putut fi antrenat. %s', ME.message);
    end

    results = [results; struct('pTest',pTest,'RF',metRF,'MLP',metMLP,'RFimp',rf.OOBPermutedPredictorDeltaError)];
end

%% avantaje/dezavantaje 
disp(' Avantaje / Dezavantaje ');
disp('Random Forest (RF):');
disp('  + Robust la outlieri, nu necesita scalare, interpretabil prin importanta trasaturi, performanta stabila.');
disp('  - Pe clase dezechilibrate poate avea recall mai mic pe clasa minoritara.');
disp('MLP:');
disp('  + Poate obtine recall mai bun pe clasa minoritara.');
disp('  - Necesita standardizare, sensibil la hiperparametri.');
disp('Feature selection:');
disp('  + Reduce dimensionalitatea.');
disp('  - Poate pierde informatie.');

function met = evalBinary(yTrue, yPred, yScorePos)
    yTrue = double(yTrue(:));
    yPred = double(yPred(:));

    cm = confusionmat(yTrue, yPred, 'Order', [0 1]);
    TN = cm(1,1); FP = cm(1,2); FN = cm(2,1); TP = cm(2,2);

    acc  = (TP+TN) / max(sum(cm(:)),1);
    prec = TP / max(TP+FP,1);
    rec  = TP / max(TP+FN,1);
    f1   = 2*prec*rec / max(prec+rec, eps);

    try
        [~,~,~,AUC] = perfcurve(yTrue, yScorePos, 1);
    catch
        AUC = NaN;
    end

    met = struct('Confusion',cm,'TN',TN,'FP',FP,'FN',FN,'TP',TP, ...
                 'Accuracy',acc,'Precision',prec,'Recall',rec,'F1',f1,'AUC',AUC);
end

function [Xos, Yos] = oversampleMinority(Xtrain, Ytrain)
    y = double(Ytrain(:));
    X = Xtrain;

    idx0 = find(y==0);
    idx1 = find(y==1);
    n0 = numel(idx0); n1 = numel(idx1);

    if n0==0 || n1==0
        Xos = X; Yos = y; 
        return;
    end

    if n0>n1
        idxMinor = idx1; nMajor = n0;
    else
        idxMinor = idx0; nMajor = n1;
    end

    addN = nMajor - numel(idxMinor);
    addIdx = idxMinor(randi(numel(idxMinor), addN, 1));

    Xos = [X; X(addIdx,:)];
    Yos = [y; y(addIdx)];
end

function showTableFig(T, figTitle)
    f = figure('Name', figTitle, 'NumberTitle','off');
    uitable(f, 'Data', table2cell(T), ...
        'ColumnName', T.Properties.VariableNames, ...
        'RowName', [], ...
        'Units','normalized', 'Position',[0 0 1 1]);
end
