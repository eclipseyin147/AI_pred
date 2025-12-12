clear all
clc
%% 创建变量
load('Copy_of_select-model fault data');
XTrain=XTrain2;
YTrain=categorical(YTrain2);%将数值数组转化为类别数组 [XTrain1;XText1] [YTrain2;YText2]
XValidation =XText2;
TValidation =categorical(YText2);

%% 构建CNN网络
numClasses = 4;
numFeatures = 8;
filterSize = 2;  %指定两个一维卷积、ReLU 和层归一化层块，其中卷积层的过滤器大小为 3。分别为第一和第二卷积层指定 32 个和 64 个过滤器。对于两个卷积层，左垫输入，使输出具有相同的长度（因果填充）。
numFilters = 32;

layers = [ ...
    sequenceInputLayer(numFeatures,Name="input")%Normalization="rescale-symmetric",
    convolution1dLayer(filterSize,numFilters,'Padding' ,"same")%,'Padding' ,"causal"
    reluLayer
    layerNormalizationLayer
    convolution1dLayer(filterSize,numFilters,'Padding' ,"same")%,'Padding' ,"causal"
    reluLayer
    layerNormalizationLayer
    globalMaxPooling1dLayer
    fullyConnectedLayer(numClasses,Name="fc")
    softmaxLayer
    classificationLayer];


miniBatchSize = 26;%分块尺寸
maxEpochs = 200;%最大训练周期数
    %'GradientDecayFactor',0,...
    %'SquaredGradientDecayFactor',0.99,...
options = trainingOptions("adam", ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',maxEpochs, ...
    'SequencePaddingDirection',"left", ...
    'InitialLearnRate',0.001,...
    'ValidationFrequency',10, ...
    'Plots',"training-progress", ...
    'Verbose',1);%'SequencePaddingDirection',"left", ...    'ValidationData',{XValidation,TValidation}, ...

%% 训练
net = trainNetwork(XTrain,YTrain,layers,options);
%feat=activations(net,XTrain,"fc");  %中间特征提取
%% 预测
YPred = classify(net,XValidation, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequencePaddingDirection',"left");
%% 精确度检验
acc = mean(YPred == TValidation)
confusionchart(TValidation,YPred)