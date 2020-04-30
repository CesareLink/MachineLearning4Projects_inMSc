
clc
clear all
%һ����ʼ������
%1.1 Ԥ������������
% ѡȡѵ������(x,y)
for  i=1:126
    x=0+0.0251*(i-1);            
    y(i)=(sin(x)+(x.^2/9+x/3)*exp((-0.5)*(x.^2)))/2;  % ���ƽ�����
end
AllSamIn=0:0.0251:pi;  %ѵ����������
AllSamOut=y;         %ѵ���������
%ѡȡ��������
for  i=1:125
    x=0.0125+0.0251*(i-1);  %������������
    ytest(i)=(sin(x)+(x.^2/9+x/3)*exp((-0.5)*(x.^2)))/2;  %�����������
end
AlltestIn=0.0125:0.0251:(pi-0.0125);
AlltestOut=ytest;
%��һ��ѵ����������������
[AlltestInn,minAlltestIn,maxAlltestIn,AlltestOutn,minAlltestOut,maxAlltestOut]= premnmx(AlltestIn,AlltestOut);  %��������
[AllSamInn,minAllSamIn,maxAllSamIn,AllSamOutn,minAllSamOut,maxAllSamOut]= premnmx(AllSamIn,AllSamOut); %ѵ������
testIn=AlltestInn;
testOut=AlltestOutn;
global Ptrain;
Ptrain = AllSamInn;
global Ttrain;
Ttrain = AllSamOutn;

%1.2 �������������
global indim;    %�������Ԫ����
indim=1;
global hiddennum; %���ز���Ԫ����
hiddennum=3;
global outdim;   %�������Ԫ����
outdim=1;
global Gpos;  

%1.3 ����΢��Ⱥ����
vmax=0.5;      % �ٶ�����
minerr=1e-7;    % Ŀ�����
wmax=0.95;
wmin=0.25;
global itmax;  % ����������
itmax=200;  
c1=1.5;
c2=1.5;
%Ȩֵ������������Եݼ��Ա�֤����
for  iter=1:itmax
    W(iter)=wmax-((wmax-wmin)/itmax)*iter; 
end 
a=-1; 
b=1;
m=-1;
n=1;
global N;  % ΢������
N=30;
global D;  % ÿ��΢����ά��
D=(indim+1)*hiddennum+(hiddennum+1)*outdim;  %����Ȩֵ����ֵ
% ��ʼ��΢��λ��
rand('state',sum(100*clock));  %������ʱ����ص������
global X;
X=a+(b-a)*rand(N,D,1);   %X��ֵ��a ��b֮�� 
%��ʼ��΢���ٶ�
V=m+(n-m)*rand(N,D,1);  %V��ֵ��m��n֮�� 


%����΢��Ⱥ���µ�������
%global net;
net=newff(minmax(Ptrain),[hiddennum,outdim],{'tansig','purelin'});
global gbest;  %ȫ������λ��
global pbest;  %�ֲ�����λ��  

%2.1��һ�ε���
fitness=fitcal(X,indim,hiddennum,outdim,D,Ptrain,Ttrain); %������Ӧֵ
[C,I]=min(fitness(:,1,1));  %��һ��������΢��Ⱥ����С��Ӧֵ��C����΢������Ÿ�I
L(:,1,1)=fitness(:,1,1);    %��һ����ÿ��΢������Ӧֵ
B(1,1,1)=C;            %��һ����ȫ��������Ӧֵ��B�洢��ǰ��������Ӧֵ��
bestminimum(1)=C;     % bestminimum�洢���д��е�ȫ����С��Ӧֵ  
gbest(1,:,1)=X(I,:,1);    %��һ����ȫ�����ŵ�΢��λ��

for  p=1:N     
    G(p,:,1)=gbest(1,:,1);  %G�����ٶȸ������㣨������ʽͳһ��
end
Gpos=gbest(1,:,1);  

for  i=1:N;
    pbest(i,:,1)=X(i,:,1);   %��Ϊ�ǵ�һ������ǰλ�ü�Ϊ��ʷ����λ�� 
end
V(:,:,2)=W(1)*V(:,:,1)+c1*rand*(pbest(:,:,1)-X(:,:,1))+c2*rand*(G(:,:,1)-X(:,:,1));  % �����ٶ�
% �ж��ٶ��Ƿ�Խ��
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
X(:,:,2)=X(:,:,1)+V(:,:,2);   %����λ��

%disp('ִ�е�����')
%2.2 ��2�ε����һ�ε���
for j=2:itmax 
h=j;
    disp('������������ǰ��ȫ�������Ӧֵ��������ǰ���д��е�ȫ�������Ӧֵ')
    disp(j-1)
    disp(B(1,1,j-1))         %j-1��ȫ��������Ӧֵ
    disp(bestminimum(j-1))  %j-1����ǰ���д��е�ȫ��������Ӧֵ
    disp('******************************') 
    fitness=fitcal(X,indim,hiddennum,outdim,D,Ptrain,Ttrain); 
    [C,I]=min(fitness(:,1,j));  %��j����������Ӧֵ������΢�����
    L(:,1,j)=fitness(:,1,j);     %��j��ÿ��΢������Ӧֵ
    B(1,1,j)=C;             %��j��ȫ��������Ӧֵ
    gbest(1,:,j)=X(I,:,j);      %��j��ȫ������΢����λ��
    [GC,GI]=min(B(1,1,:));   %���д���ȫ��������Ӧֵ����GC����������GI
    bestminimum(j)=GC;    %���д���������Ӧֵ����j����bestminimum   
   
    % �ж��Ƿ��������
    if  GC<=minerr
       Gpos=gbest(1,:,GI);  %��������������������¼����λ�ã�ֹͣ���� 
       break
    end
    if  j>=itmax
        break      %��������������ʱ���˳�
end   

    %������ʷȫ������λ��
    if  B(1,1,j)<GC   
       gbest(1,:,j)=gbest(1,:,j);        
    else
       gbest(1,:,j)=gbest(1,:,GI);  
    end  
    for  p=1:N
        G(p,:,j)=gbest(1,:,j); 
    end
    %�����΢����ʷ����λ��
    for  i=1:N;
        [C,I]=min(L(i,1,:));  %����ÿ��΢������ʷ������Ӧֵ������C����������I
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

  %2.3 ������΢����������Ȩֵ����ֵ������������
   if j=itmax
     Gpos=gbest(1,:,GI);
   end
  disp('Ҫ��ʾGpos��ֵ')
  disp(Gpos)
  wi=Gpos(1:hiddennum);                       %�����-���ز�Ȩֵ
  wl=Gpos(hiddennum+1:2*hiddennum);           %���ز�-�����Ȩֵ
  b1=Gpos(2*hiddennum+1:3*hiddennum);         %�����-���ز���ֵ
  b2=Gpos(3*hiddennum+1:3*hiddennum+outdim);  %���ز�-�������ֵ

end 
%����������ѵ������
%****************************************************************************
[w,v]=size(testIn);  %w ����������v��������
for k=1:v   % v�ǲ��������ĸ���
   for t=1:hiddennum  %�������ز�ÿ����Ԫ�����룬���
      hidinput=0;
      hidinput=wi(t)*testIn(k)-b1(t); 
      hidoutput(t)=tansig(hidinput);    
   end
   outinput=0; %used to calculate the value of output in outlayer
   for t=1:hiddennum
      outinput=outinput+wl(t)*hidoutput(t);  %�����ֻ��һ����Ԫʱ�����    
   end   
   outVal(k)=purelin(outinput-b2);   %���������ֵ
end

subplot(2,1,1)  %//���ô��ھ��
[AlltestIn,AlltestOut]=
postmnmx(testIn,minAlltestIn,maxAlltestIn,testOut,minAlltestOut,maxAlltestOut);  %����һ��
[ResVal]=postmnmx(outVal,minAlltestOut,maxAlltestOut);
trainError=abs(ResVal-AlltestOut);  %�������
for k=1:v
   SquareE(k)=(trainError(k)*trainError(k))/2;  %v���������������
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


%��Ӧ�Ⱥ�������
function fitval = fitcal(X,indim,hiddennum,outdim,D,Ptrain,Ttrain) 
%��ά����x ΢������X����������y ΢��ά����X����������z ������X�Ĳ�����
[x,y,z]=size(X);  
[w,v]=size(Ptrain);  %��ά����w ѵ������ά��������Ϊ1��v ѵ����������

for  i=1:x   %x��������������z������� 
wi=X(i,1:hiddennum,z);   
wl=X(i,1*hiddennum+1:2*hiddennum,z); 
    b1=X(i,2*hiddennum+1:3*hiddennum,z);  
    b2=X(i,3*hiddennum+1:3*hiddennum+outdim,z);
    error=0;
    for k=1:v   %ѵ����������
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
       errval(k)=Ttrain(k)-outval(k);   %�������
       error=error+errval(k)*errval(k);  %v�����������ƽ�����
    end
    fitval(i,1,z)=error/v;   %������,����ֵ�ǵ�i��΢����z�������
end

