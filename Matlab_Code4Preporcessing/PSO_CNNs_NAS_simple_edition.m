
clc
clear all
%一、初始化部分
%1.1 预处理样本数据
% 选取训练样本(x,y)
for  i=1:126
    x=0+0.0251*(i-1);            
    y(i)=(sin(x)+(x.^2/9+x/3)*exp((-0.5)*(x.^2)))/2;  % 待逼近函数
end
AllSamIn=0:0.0251:pi;  %训练样本输入
AllSamOut=y;         %训练样本输出
%选取测试样本
for  i=1:125
    x=0.0125+0.0251*(i-1);  %测试样本输入
    ytest(i)=(sin(x)+(x.^2/9+x/3)*exp((-0.5)*(x.^2)))/2;  %测试样本输出
end
AlltestIn=0.0125:0.0251:(pi-0.0125);
AlltestOut=ytest;
%归一化训练样本，测试样本
[AlltestInn,minAlltestIn,maxAlltestIn,AlltestOutn,minAlltestOut,maxAlltestOut]= premnmx(AlltestIn,AlltestOut);  %测试样本
[AllSamInn,minAllSamIn,maxAllSamIn,AllSamOutn,minAllSamOut,maxAllSamOut]= premnmx(AllSamIn,AllSamOut); %训练样本
testIn=AlltestInn;
testOut=AlltestOutn;
global Ptrain;
Ptrain = AllSamInn;
global Ttrain;
Ttrain = AllSamOutn;

%1.2 设置神经网络参数
global indim;    %输入层神经元个数
indim=1;
global hiddennum; %隐藏层神经元个数
hiddennum=3;
global outdim;   %输出层神经元个数
outdim=1;
global Gpos;  

%1.3 设置微粒群参数
vmax=0.5;      % 速度上限
minerr=1e-7;    % 目标误差
wmax=0.95;
wmin=0.25;
global itmax;  % 最大迭代次数
itmax=200;  
c1=1.5;
c2=1.5;
%权值随迭代次数线性递减以保证收敛
for  iter=1:itmax
    W(iter)=wmax-((wmax-wmin)/itmax)*iter; 
end 
a=-1; 
b=1;
m=-1;
n=1;
global N;  % 微粒个数
N=30;
global D;  % 每个微粒的维数
D=(indim+1)*hiddennum+(hiddennum+1)*outdim;  %所有权值和阈值
% 初始化微粒位置
rand('state',sum(100*clock));  %产生和时间相关的随机数
global X;
X=a+(b-a)*rand(N,D,1);   %X的值在a 和b之间 
%初始化微粒速度
V=m+(n-m)*rand(N,D,1);  %V的值在m和n之间 


%二、微粒群更新迭代部分
%global net;
net=newff(minmax(Ptrain),[hiddennum,outdim],{'tansig','purelin'});
global gbest;  %全局最优位置
global pbest;  %局部最优位置  

%2.1第一次迭代
fitness=fitcal(X,indim,hiddennum,outdim,D,Ptrain,Ttrain); %计算适应值
[C,I]=min(fitness(:,1,1));  %第一代，返回微粒群中最小适应值给C，该微粒的序号给I
L(:,1,1)=fitness(:,1,1);    %第一代，每个微粒的适应值
B(1,1,1)=C;            %第一代，全局最优适应值（B存储当前代最优适应值）
bestminimum(1)=C;     % bestminimum存储所有代中的全局最小适应值  
gbest(1,:,1)=X(I,:,1);    %第一代，全局最优的微粒位置

for  p=1:N     
    G(p,:,1)=gbest(1,:,1);  %G便于速度更新运算（函数格式统一）
end
Gpos=gbest(1,:,1);  

for  i=1:N;
    pbest(i,:,1)=X(i,:,1);   %因为是第一代，当前位置即为历史最优位置 
end
V(:,:,2)=W(1)*V(:,:,1)+c1*rand*(pbest(:,:,1)-X(:,:,1))+c2*rand*(G(:,:,1)-X(:,:,1));  % 更新速度
% 判断速度是否越界
for ni=1:N
    for di=1:D
        if V(ni,di,2)>vmax
           V(ni,di,2)=vmax;
        else if V(ni,di,2)<-vmax
              V(ni,di,2)=-vmax;
            else
              V(ni,di,2)=V(ni,di,2);
        end
    end
end 
X(:,:,2)=X(:,:,1)+V(:,:,2);   %更新位置

%disp('执行到这里')
%2.2 第2次到最后一次迭代
for j=2:itmax 
h=j;
    disp('迭代次数，当前代全局最佳适应值，本代以前所有代中的全局最佳适应值')
    disp(j-1)
    disp(B(1,1,j-1))         %j-1代全局最优适应值
    disp(bestminimum(j-1))  %j-1代以前所有代中的全局最优适应值
    disp('******************************') 
    fitness=fitcal(X,indim,hiddennum,outdim,D,Ptrain,Ttrain); 
    [C,I]=min(fitness(:,1,j));  %第j代的最优适应值和最优微粒序号
    L(:,1,j)=fitness(:,1,j);     %第j代每个微粒的适应值
    B(1,1,j)=C;             %第j代全局最优适应值
    gbest(1,:,j)=X(I,:,j);      %第j代全局最优微粒的位置
    [GC,GI]=min(B(1,1,:));   %所有代的全局最优适应值赋给GC，代数赋给GI
    bestminimum(j)=GC;    %所有代的最优适应值赋给j代的bestminimum   
   
    % 判断是否符合条件
    if  GC<=minerr
       Gpos=gbest(1,:,GI);  %若满足均方误差条件，记录最优位置，停止迭代 
       break
    end
    if  j>=itmax
        break      %超过最大迭代次数时，退出
end   

    %计算历史全局最优位置
    if  B(1,1,j)<GC   
       gbest(1,:,j)=gbest(1,:,j);        
    else
       gbest(1,:,j)=gbest(1,:,GI);  
    end  
    for  p=1:N
        G(p,:,j)=gbest(1,:,j); 
    end
    %计算各微粒历史最优位置
    for  i=1:N;
        [C,I]=min(L(i,1,:));  %计算每个微粒的历史最优适应值，赋给C，代数赋给I
        if L(i,1,j)<=C       
            pbest(i,:,j)=X(i,:,j); 
        else
            pbest(i,:,j)=X(i,:,I);  
        end
    end
    V(:,:,j+1)=W(j)*V(:,:,j)+c1*rand*(pbest(:,:,j)-X(:,:,j))+c2*rand*(G(:,:,j)-X(:,:,j));  
    for ni=1:N
        for di=1:D
            if V(ni,di,j+1)>vmax
                V(ni,di,j+1)=vmax;
            else  if V(ni,di,j+1)<-vmax
                   V(ni,di,j+1)=-vmax;
                 else
                   V(ni,di,j+1)=V(ni,di,j+1);
            end
        end
    end
   X(:,:,j+1)=X(:,:,j)+V(:,:,j+1);     

  %2.3 将最优微粒（即最优权值和阈值）赋给神经网络
   if j=itmax
     Gpos=gbest(1,:,GI);
   end
  disp('要显示Gpos的值')
  disp(Gpos)
  wi=Gpos(1:hiddennum);                       %输入层-隐藏层权值
  wl=Gpos(hiddennum+1:2*hiddennum);           %隐藏层-输出层权值
  b1=Gpos(2*hiddennum+1:3*hiddennum);         %输入层-隐藏层阈值
  b2=Gpos(3*hiddennum+1:3*hiddennum+outdim);  %隐藏层-输出层阈值

end 
%三、神经网络训练部分
%****************************************************************************
[w,v]=size(testIn);  %w 返回行数，v返回列数
for k=1:v   % v是测试样本的个数
   for t=1:hiddennum  %计算隐藏层每个神经元的输入，输出
      hidinput=0;
      hidinput=wi(t)*testIn(k)-b1(t); 
      hidoutput(t)=tansig(hidinput);    
   end
   outinput=0; %used to calculate the value of output in outlayer
   for t=1:hiddennum
      outinput=outinput+wl(t)*hidoutput(t);  %输出层只有一个神经元时的情况    
   end   
   outVal(k)=purelin(outinput-b2);   %输出层的输出值
end

subplot(2,1,1)  %//调用窗口句柄
[AlltestIn,AlltestOut]=
postmnmx(testIn,minAlltestIn,maxAlltestIn,testOut,minAlltestOut,maxAlltestOut);  %反归一化
[ResVal]=postmnmx(outVal,minAlltestOut,maxAlltestOut);
trainError=abs(ResVal-AlltestOut);  %测试误差
for k=1:v
   SquareE(k)=(trainError(k)*trainError(k))/2;  %v个样本的误差数组
end
plot(AlltestIn,SquareE)
ylabel('Error')

subplot(2,1,2)
j=1:1:h;
plot(j,bestminimum(j))

set(gca,'XLim',[1 100000]);
set(gca,'XMinorTick','on');
set(gca,'XTick',[1 10 100 1000 10000 100000]);
set(gca,'YLim',[0.000001 1]);
set(gca,'YMinorTick','on');
set(gca,'YTick',[0.000001 0.00001 0.0001 0.001 0.01 0.1 1]);
set(gca,'yscale','log','xscale','log')
ylabel('training error')
xlabel('Iteration Number')

hold on


%适应度函数部分
function fitval = fitcal(X,indim,hiddennum,outdim,D,Ptrain,Ttrain) 
%三维矩阵：x 微粒数（X的行数）；y 微粒维数（X的列数）；z 代数（X的层数）
[x,y,z]=size(X);  
[w,v]=size(Ptrain);  %二维矩阵：w 训练样本维数，这里为1；v 训练样本个数

for  i=1:x   %x代表粒子数量，z代表代数 
wi=X(i,1:hiddennum,z);   
wl=X(i,1*hiddennum+1:2*hiddennum,z); 
    b1=X(i,2*hiddennum+1:3*hiddennum,z);  
    b2=X(i,3*hiddennum+1:3*hiddennum+outdim,z);
    error=0;
    for k=1:v   %训练样本总数
       for t=1:hiddennum
          hidinput=0;
          hidinput=wi(t)*Ptrain(k)-b1(t);  
          hidoutput(t)=tansig(hidinput);
       end
       outinput=0; 
       for t=1:hiddennum
          outinput=outinput+wl(t)*hidoutput(t);         
       end
       outval(k)=purelin(outinput-b2); 
       errval(k)=Ttrain(k)-outval(k);   %绝对误差
       error=error+errval(k)*errval(k);  %v个样本的误差平方求和
    end
    fitval(i,1,z)=error/v;   %均方和,返回值是第i个微粒第z代的误差
end

