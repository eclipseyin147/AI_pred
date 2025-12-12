%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%        混合模型20220309
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

%% 导入数据
load Data_V13_40kW.txt
data=Data_V13_40kW(1:900,:);
w = 5 ; % SW窗口大小
%% 导入实验数据
tt = data(:,1) ; % 累计时间 ，h
Pa = data(:,6) ; % 氢气路入口气压
Pc = data(:,5) ; % 空气路入口气压
T  = data(:,9) + 273.15; % 冷却液出口温度
I  = data(:,11) ; % 电流 ，A
V_cell_exp = data(:,12)./300 ; % 单池电压 ，V
% P_out_exp  = data(:,13)./300 ;  % 单池功率 ，kW

%%

Input  = [data(:,5:6),data(:,9),data(:,11)]; % 4变量
Output = [data(:,12)];   % 1输出
Dataset= [Input, Output]; %

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
% numTimeStepsTrain = floor(0.9*numel(output_data));
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

%% Validation
% 输入需要优化过程的初始状态
kk = length(output_test(1,:));

%% 开始预测下一个时刻的值
input = input_test(1,1);

V_Std = output_test(1);
aV_DDM=[];
aV_SEM=[];
load net1.mat
%%
for n=1:kk
    n
    %% 测试模块 test
    % 输出结果反归一化处理
    inputn_new=mapminmax('apply',input,inputps);  % 预测数据归一化
    an=sim(net,inputn_new)                     ;  % 网格预测输出
    BPoutput=mapminmax('reverse',an,outputps)  ;  % 网格输出反归一化
    
    output=BPoutput';    
    input = input_test(:,n);
    
    %% 数据驱动模型预测结果
    V_DDM = output;
    %% 将当前时间层数据放入数据矩阵中
    input1 = tt(300+w-1+n);
    input2 = Pc(300+w-1+n);
    input3 = Pa(300+w-1+n);
    input4 = T(300+w-1+n);
    input5 = I(300+w-1+n);
    
    %% 半经验模型预测结果
    [V_SEM]=SEDM(input1,input2,input3,input4,input5);
    
    %% 动态分配权重
    RR = 4   ;
    output =(RR*V_SEM + V_DDM)/(RR+1);
    
    input(5*w-5) = output; % 带回输入进行循环预测
    %% 更新下一次权重
    V_Std=output;
    aV_DDM(n,:) =  V_DDM ;
    aV_SEM(n,:) =  V_SEM ;
    aV_hybrid(n,:)=output';
    
    
end
YPred = aV_hybrid;



%% 误差分析

YTest_ave = mean(YTest,1);
RR_SEM    = 1 - sum((YTest - aV_SEM).^2,    1)./sum((YTest_ave - YTest).^2,1)
RR_DDM    = 1 - sum((YTest - aV_DDM).^2,    1)./sum((YTest_ave - YTest).^2,1)
RR_Hybird = 1 - sum((YTest - aV_hybrid).^2, 1)./sum((YTest_ave - YTest).^2,1)

RE_SEM    = 100*(aV_SEM    - YTest)./YTest;
RE_DDM    = 100*(aV_DDM    - YTest)./YTest;
RE_Hybrid = 100*(aV_hybrid - YTest)./YTest;

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

function [V_stack_sim]=SEDM(tt,Pc,Pa,T,I)
%% 定义常数
nn     = 300; % 电堆中单池数量
A_cell = 190e-4;  % m2
L_Pt = 4; % g m-2
F = 96487;         % Faraday's Constant (coulomb/mole)
R = 8.314472;      % Ideal gas constant (J/K/mol)
P0= 101325;
Alpha_c = 0.2;
Alpha_a = 0.8;
Gamma_a = 0.5;   % 气体浓度影响活化损失的程度
Gamma_c = 1.0;
c_o2_ref= 3.39;
t_MEM  = 15e-6;
t_CLc  = 15e-6;
t_MPLc = 30e-6;
t_GDLc = 180e-6;
t_CHc  = 440e-6;
POR_CLc  = 0.455;
POR_MPLc = 0.4;
POR_GDLc = 0.6;
j_ref_a = 10;   % 参考交换电流密度 A m-2    典型值 1~10 A m-2  (基于真实单位表面积)   相差6个数量级
j_ref_c = 1e-5;
%%   定义经验系数
b_leak = 1e-3; %  1e-5  修正泄露电密
b_ECSA = -2e-4; % -4e-4  修正有效反应面积
b_ion  = 2e-4;  %  1e-4  修正离子电导率
b_R    = 1e-8; % 修正接触电阻
b_D    = 1e-1; % 修正气体扩散系数
b_B    = 1e-5; % 修正浓差损失放大系数
%% 修正系数
r_leak = exp(b_leak.*tt);
r_ECSA = exp(b_ECSA.*tt);
r_ion  = exp(b_ion .*tt);
%% 初始化参数
i_leak_ini = 20*190e-4;  % 200 A m-2 190e-4 m2
A_ECSA_ini = 60.*(A_cell.*L_Pt);% 微观总反应面积 m2/g * g m-2 * m2
R_ion_ini  = 100e-7./A_cell;
R_ele_ini  = 20e-7./A_cell; % ohmic
D_o2_ini   = 2.652e-5 .* (T ./333.15).^1.5 .* (1./Pc) .* POR_GDLc.^1.5;
K_c_ini    = 100;
% 泄露电流变化，和膜厚度有关
i_leak = i_leak_ini .* r_leak;
% 有效反应面积变化 m2 g-1
A_ECSA = A_ECSA_ini .* r_ECSA;
% 总阻抗变化
R_total = R_ion_ini .* r_ion + (R_ele_ini + b_R.*tt);
% 气体传质阻力变化
D_o2 = D_o2_ini + b_D.*tt;
% 浓差损失修正系数变化
K_c = K_c_ini + b_B.*tt;
%% 能斯特电压 % 开路电压 0.968 V
E_nernst = 1.229 - 0.846e-3.*(T - 298.15) + R .* T ./ 2 ./F .* ((log( Pa ) + 0.5.*log( Pc.*0.21))) ;
%% 活化损失
% 参考体交换电流密度 A m-2 (单位有效反应面积)
b_a = R .* T ./ (2.*Alpha_a.* F) ;
theta_T_a = exp(-1400.*(1./T - 1/298.15));
c_h2_CLa = Pa.*P0./R./T;
k_ele_a = j_ref_a.* (c_h2_CLa ./ c_o2_ref).^Gamma_a.*theta_T_a;
V_act_a =  b_a.*(i_leak + I)./A_ECSA./ k_ele_a ;
b_c = R .* T ./ (4.*Alpha_c.* F) ;
theta_T_c = exp(-7900.*(1./T - 1/298.15));
c_o2_CLc = 0.21.*Pc.*P0./R./T;
k_ele_c = j_ref_c.* (c_o2_CLc ./ c_o2_ref).^Gamma_c.*theta_T_c;
V_act_c =  -b_c.*log( (i_leak + I)./A_ECSA./ k_ele_c );
%% 欧姆损失
V_ohm = -I.*R_total;
%% 浓差损失
D_o2_GDLc = 2.652e-5 .* (T ./333.15).^1.5 .* (1./Pc) .* POR_GDLc.^1.5;
P_o2 = Pc*0.21*P0;
I_lim = 4*F.*(D_o2_GDLc./t_GDLc).*(P_o2./R./T);% 极限电流密度估算
term_c = 1 - (I./A_ECSA) ./ I_lim;
V_conc_c =  K_c .* b_c.* log(term_c) ;
%% 单池输出电压，Vout V
V_cell_sim = E_nernst + V_act_a  + V_act_c + V_ohm + V_conc_c ;

%% 单池输出功率 kW
P_out_sim = V_cell_sim .* I./1000;
% Monitor_eta = [V_act_c, V_ohm, V_conc_c, V_cell_sim, V_cell_exp, P_out_sim, P_out_exp];
% Monitor = [V_act_a(end); V_act_c(end); V_ohm(end); V_conc_c(end)]
V_stack_sim = V_cell_sim.*300;
% V_stack_exp = V_cell_exp.*300;
end
