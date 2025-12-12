%% 每块之间残差连接，每个残差2层，残差之间膨胀，并采用dropout
clear all

load("ALL_Traindata1.mat");
XTrain = AXTrain3;
TTrain = categorical(AYTrain);


numObservations = numel(XTrain);
setdemorandstream(1);

% s = load("3S_Testdata.mat");
% XTest = s.TAXTest;
% TTest = s.TAYTest;
% TTest = categorical(TTest);
% 
% TXTrain = {XTrain
%     XTest};
% TTTrain = {TTrain
%     TTest};

classes = {'1','2','3','4','5'};
numClasses = numel(classes);

numFeatures = size(XTrain,1);

% figure
% for i = 1:3
%     X = s.XTrain{1}(i,:);
% 
%     subplot(4,1,i)
%     plot(X)
%     ylabel("Feature " + i + newline + "Acceleration")
% end
% 
% subplot(4,1,4)
% 
% hold on
% plot(s.YTrain{1})
% hold off
% 
% xlabel("Time Step")
% ylabel("Activity")
% 
% subplot(4,1,1)
% title("Training Sequence 1")
%% 结构
numFilters = 64;
filterSize = 3;
dropoutFactor = 0.005;
numBlocks = 4;

layer = sequenceInputLayer(numFeatures,Normalization="rescale-symmetric",Name="input"); %
lgraph = layerGraph(layer);

outputName = layer.Name;

for i = 1:numBlocks
    dilationFactor =2^(i-1) ;%2^(i-1)
    
    layers = [
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal",Name="conv1_"+i)
        layerNormalizationLayer
        reluLayer
        spatialDropoutLayer(dropoutFactor)
        additionLayer(2,Name="add_"+i)];
%     convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal")
%         layerNormalizationLayer
%         reluLayer
%         spatialDropoutLayer(dropoutFactor)


    % Add and connect layers.
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph,outputName,"conv1_"+i);

    % Skip connection.
    if (i == 1)
        % Include convolution in first skip connection.
        layer = convolution1dLayer(1,numFilters,Padding="same",Name="convSkip");
    
        lgraph = addLayers(lgraph,layer);
        lgraph = connectLayers(lgraph,"input","convSkip");
        lgraph = connectLayers(lgraph,"convSkip","add_" + i + "/in2");
    else
       
        lgraph = connectLayers(lgraph,outputName,"add_" + i + "/in2");
    end
    
    % Update layer output name.
    outputName = "add_" + i;
end

layers = [
    fullyConnectedLayer(48, Name="fc1")    
    fullyConnectedLayer(numClasses, Name="fc2")
    softmaxLayer
    classificationLayer];
lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,outputName,"fc1");

figure
plot(lgraph)
title("Temporal Convolutional Network")

%将最后两个卷积层作为目标域和源域
%mmd_XY=my_mmd(x, y, 4)，xy为目标和源的全连接层的值


options = trainingOptions("adam", ...
    MaxEpochs=100, ...
    miniBatchSize=1, ...
    Plots="training-progress", ...
    Verbose=1 ,...
    VerboseFrequency=2);

TCNnet = trainNetwork(TXTrain,TTTrain,lgraph,options);

%feat=activations(TCNnet,XTrain,"fc1");  %中间特征提取

%% 分类测试
s = load("3S_Testdata.mat");
XTest = s.AXTest;
TTest = s.AYTest;
TTest = categorical(TTest);
% TTest = {TTest};

[YPred, err] = classify(TCNnet,XTest);

figure
plot(YPred,".-")
hold on
plot(TTest)
hold off

xlabel("Time Step")
ylabel("Activity")
legend(["Predicted" "Test Data"],Location="northeast")
title("Test Sequence Predictions")

figure
confusionchart(TTest,YPred)

accuracy = mean(YPred == TTest)