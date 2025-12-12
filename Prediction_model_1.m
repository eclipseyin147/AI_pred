%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%        BP神经网络
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

%% 导入数据
load Data_V13_40kW.txt
data=Data_V13_40kW(1:900,:);
w = 5 ; % 窗口大小

Input  = [data(:,5:6),data(:,9),data(:,11)]; % 4变量
Output = [data(:,12)];   % 1输出
Dataset= [Input, Output]; % 5*5-1

dd = length(Output(:,1));
input_data =[];
output_data=[];
Input_pre  =[];
%% SW方法处理数据集
for i=1:dd-w
    for j=1:w
        Input_pre = [Input_pre, Dataset(i + j - 1,: )];
    end
    input_data = [input_data;  Input_pre(1:end-1)];
    output_data= [output_data; Input_pre(end) ];
    Input_pre =[];
end
input_data=input_data';
output_data=output_data';

%%
%序列的前 90% 用于训练，后 10% 用于测试
% numTimeStepsTrain = floor(0.7*numel(output_data));
numTimeStepsTrain=300;
XTrain = input_data(:,1:numTimeStepsTrain);
YTrain = output_data(1:numTimeStepsTrain);

XTest = input_data(:,numTimeStepsTrain+1:end);
YTest = output_data(numTimeStepsTrain+1:end);

input_train=XTrain;        % n组训练输入（6个元素）
output_train=YTrain;      % n组输出结果（1个元素）
input_test=XTest;   % m-n 组测试输入（6个元素）
output_test=YTest;

%% 输入输出数据归一化处理
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%% 循环训练神经网络并检测误差，一直到符合要求的预测精度中止
for pp=1:10
    pp
    %% 初始化网格结果
    net=newff(inputn,outputn,[50,50]);  % newff(a,b,c)  a 输入样本数  b 输出样本数  c 隐藏层节点数
    net.trainParam.epochs=20000;   % 设置迭代次数
    net.trainParam.lr  = 1e-9;     % 设置学习率  1e-6
    net.trainParam.goal= 1e-10;     % 设置目标值  1e-6
    net=train(net,inputn,outputn); % 网格训练
    
    %% Validation
    % 输入需要优化过程的初始状态
    kk = length(output_test(1,:));
    
    %% 开始预测下一个时刻的值
    input = input_test(1,1);
    for n=1:kk
        %         n=n;
        %% 测试模块 test
        % 输出结果反归一化处理
        inputn_new=mapminmax('apply',input,inputps);  % 预测数据归一化
        an=sim(net,inputn_new)                     ;  % 网格预测输出
        BPoutput=mapminmax('reverse',an,outputps)  ;  % 网格输出反归一化
        
        output=BPoutput';
        
        input = input_test(:,n);
        input(5*w-5) = output;
        
        
        %% 将当前时间层数据放入数据矩阵中
        data_pre(n,:)=output';
    end
    YPred = data_pre';
    
    %% 误差分析
    RMSE = (sum((YTest(1:end) - YPred).^2,2)./kk).^0.5
    YTest_ave = mean(YTest,2);
    RR_2 = 1 - sum((YTest - YPred).^2,2)./sum((YTest_ave - YTest).^2,2)
    
    if min(RR_2) > 0.8
        %     if RR_2(5) >= 0.8
        save net
        
        break
    end
end
%%
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Loads")
title("Forecast with Updates")
subplot(2,1,2)
stem(YPred - YTest)
xlabel("Hours")
ylabel("Error")
title("RMSE = " + RMSE)
%%
YTest=YTest';
YPred=YPred';


