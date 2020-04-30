
clear all

 %remove the neutral category, set up 5 cross validation dataset

 r1={'.xlsx'};
for ii=1:2   
    %nam1=strcat('den_cq_bei',r1{ii})
    %nam2=strcat('den_cq_kong',r1{ii})
    %nam4=strcat('den_cq_xi',r1{ii})
    nam1=['train_P_',r1{ii}];
    nam2=['train_Z_',r1{ii}];
    nam4=['train_N_',r1{ii}];    
    nam1T=['test_P_',r1{ii}];
    nam2T=['test_Z_',r1{ii}];
    nam4T=['test_N_',r1{ii}];  
    load(nam1)
    load(nam2)
    load(nam4)
    load(nam1T)
    load(nam2T)
    load(nam4T)
                        
x11([(ii-1)*600+1]:600*ii,:)=P;
x22([(ii-1)*600+1]:600*ii,:)=Z;
x44([(ii-1)*600+1]:600*ii,:)=N;
x10([(ii-1)*400+1]:400*ii,:)=P1;
x20([(ii-1)*400+1]:400*ii,:)=Z1;
x40([(ii-1)*400+1]:400*ii,:)=N1;
clear P N Z
end

rn1=randperm(size(x11,1));    
new_x11=x11(rn1,:);
rn2=randperm(size(x22,1));
new_x22=x22(rn2,:);
rn4=randperm(size(x44,1));
new_x44=x44(rn4,:);                   
num1=new_x11(1:855,:);
num2=new_x22(1:855,:);
num4=new_x44(1:855,:);                  
save('rn1.mat','rn1');
save('rn2.mat','rn2');
save('rn4.mat','rn4');

num1size=size(x11,1);
num2size=size(x22,1);
num4size=size(x44,1);
label1=ones(num1size,1)*[0 0 1];
label2=ones(num2size,1)*[0 1 0];
label4=ones(num4size,1)*[1 0 0];
label1=label1(rn1,:);
label2=label2(rn2,:);
label4=label4(rn4,:);
new_label1=label1(1:855,:);
new_label2=label2(1:855,:);
new_label4=label4(1:855,:);             
numsize=size(num1,1);               
k=10;              

kk=2;        %generate the k- train and test dataset               
numtrain=floor((kk-1)*numsize/k);
numtrain1=floor(kk*numsize/k);
train_x2=[num1([1:numtrain,numtrain1+1:end],:);num2([1:numtrain,numtrain1+1:end],:);num4([1:numtrain,numtrain1+1:end],:)];
train_y2=[new_label1([1:numtrain,numtrain1+1:end],:);new_label2([1:numtrain,numtrain1+1:end],:);new_label4([1:numtrain,numtrain1+1:end],:)];
test_x2=[num1(numtrain+1:numtrain1,:);num2(numtrain+1:numtrain1,:);num4(numtrain+1:numtrain1,:)];
test_y2=[new_label1(numtrain+1:numtrain1,:);new_label2(numtrain+1:numtrain1,:);new_label4(numtrain+1:numtrain1,:)];

%random shuffle
ran1=randperm(size(train_x2,1));
train_x2=train_x2(ran1,:);
train_y2=train_y2(ran1,:);
ran2=randperm(size(test_x2,1));
test_x2=test_x2(ran2,:);
test_y2=test_y2(ran2,:);
ran3=randperm(size(train_x2,1));
train_x2=train_x2(ran3,:);
train_y2=train_y2(ran3,:);
ran4=randperm(size(test_x2,1));
test_x2=test_x2(ran4,:);
test_y2=test_y2(ran4,:);

%save the dataset
save('train_x2.mat','train_x2');
save('train_y2.mat','train_y2');
save('test_x2.mat','test_x2');
save('test_y2.mat','test_y2');


kk=3;                      
numtrain=floor((kk-1)*numsize/k);
numtrain1=floor(kk*numsize/k);
train_x3=[num1([1:numtrain,numtrain1+1:end],:);num2([1:numtrain,numtrain1+1:end],:);num4([1:numtrain,numtrain1+1:end],:)];
train_y3=[new_label1([1:numtrain,numtrain1+1:end],:);new_label2([1:numtrain,numtrain1+1:end],:);new_label4([1:numtrain,numtrain1+1:end],:)];
test_x3=[num1(numtrain+1:numtrain1,:);num2(numtrain+1:numtrain1,:);num4(numtrain+1:numtrain1,:)];
test_y3=[new_label1(numtrain+1:numtrain1,:);new_label2(numtrain+1:numtrain1,:);new_label4(numtrain+1:numtrain1,:)];


ran1=randperm(size(train_x3,1));
train_x3=train_x3(ran1,:);
train_y3=train_y3(ran1,:);
ran2=randperm(size(test_x3,1));
test_x3=test_x3(ran2,:);
test_y3=test_y3(ran2,:);
ran3=randperm(size(train_x3,1));
train_x3=train_x3(ran3,:);
train_y3=train_y3(ran3,:);
ran4=randperm(size(test_x3,1));
test_x3=test_x3(ran4,:);
test_y3=test_y3(ran4,:);

save('train_x3.mat','train_x3');
save('train_y3.mat','train_y3');
save('test_x3.mat','test_x3');
save('test_y3.mat','test_y3');


kk=4;                    
numtrain=floor((kk-1)*numsize/k);
numtrain1=floor(kk*numsize/k);
train_x4=[num1([1:numtrain,numtrain1+1:end],:);num2([1:numtrain,numtrain1+1:end],:);num4([1:numtrain,numtrain1+1:end],:)];
train_y4=[new_label1([1:numtrain,numtrain1+1:end],:);new_label2([1:numtrain,numtrain1+1:end],:);new_label4([1:numtrain,numtrain1+1:end],:)];
test_x4=[num1(numtrain+1:numtrain1,:);num2(numtrain+1:numtrain1,:);num4(numtrain+1:numtrain1,:)];
test_y4=[new_label1(numtrain+1:numtrain1,:);new_label2(numtrain+1:numtrain1,:);new_label4(numtrain+1:numtrain1,:)];


ran1=randperm(size(train_x4,1));
train_x4=train_x4(ran1,:);
train_y4=train_y4(ran1,:);
ran2=randperm(size(test_x4,1));
test_x4=test_x4(ran2,:);
test_y4=test_y4(ran2,:);
ran3=randperm(size(train_x4,1));
train_x4=train_x4(ran3,:);
train_y4=train_y4(ran3,:);
ran4=randperm(size(test_x4,1));
test_x4=test_x4(ran4,:);
test_y4=test_y4(ran4,:);

save('train_x4.mat','train_x4');
save('train_y4.mat','train_y4');
save('test_x4.mat','test_x4');
save('test_y4.mat','test_y4');


kk=5;                  
numtrain=floor((kk-1)*numsize/k);
numtrain1=floor(kk*numsize/k);
train_x5=[num1([1:numtrain,numtrain1+1:end],:);num2([1:numtrain,numtrain1+1:end],:);num4([1:numtrain,numtrain1+1:end],:)];
train_y5=[new_label1([1:numtrain,numtrain1+1:end],:);new_label2([1:numtrain,numtrain1+1:end],:);new_label4([1:numtrain,numtrain1+1:end],:)];
test_x5=[num1(numtrain+1:numtrain1,:);num2(numtrain+1:numtrain1,:);num4(numtrain+1:numtrain1,:)];
test_y5=[new_label1(numtrain+1:numtrain1,:);new_label2(numtrain+1:numtrain1,:);new_label4(numtrain+1:numtrain1,:)];


ran1=randperm(size(train_x5,1));
train_x5=train_x5(ran1,:);
train_y5=train_y5(ran1,:);
ran2=randperm(size(test_x5,1));
test_x5=test_x5(ran2,:);
test_y5=test_y5(ran2,:);
ran3=randperm(size(train_x5,1));
train_x5=train_x5(ran3,:);
train_y5=train_y5(ran3,:);
ran4=randperm(size(test_x5,1));
test_x5=test_x5(ran4,:);
test_y5=test_y5(ran4,:);

save('train_x5.mat','train_x5');
save('train_y5.mat','train_y5');
save('test_x5.mat','test_x5');
save('test_y5.mat','test_y5');

kk=6;              
numtrain=floor((kk-1)*numsize/k);
numtrain1=floor(kk*numsize/k);
train_x6=[num1([1:numtrain,numtrain1+1:end],:);num2([1:numtrain,numtrain1+1:end],:);num4([1:numtrain,numtrain1+1:end],:)];
train_y6=[new_label1([1:numtrain,numtrain1+1:end],:);new_label2([1:numtrain,numtrain1+1:end],:);new_label4([1:numtrain,numtrain1+1:end],:)];
test_x6=[num1(numtrain+1:numtrain1,:);num2(numtrain+1:numtrain1,:);num4(numtrain+1:numtrain1,:)];
test_y6=[new_label1(numtrain+1:numtrain1,:);new_label2(numtrain+1:numtrain1,:);new_label4(numtrain+1:numtrain1,:)];


ran1=randperm(size(train_x6,1));
train_x6=train_x6(ran1,:);
train_y6=train_y6(ran1,:);
ran2=randperm(size(test_x6,1));
test_x6=test_x6(ran2,:);
test_y6=test_y6(ran2,:);
ran3=randperm(size(train_x6,1));
train_x6=train_x6(ran3,:);
train_y6=train_y6(ran3,:);
ran4=randperm(size(test_x6,1));
test_x6=test_x6(ran4,:);
test_y6=test_y6(ran4,:);

save('train_x6.mat','train_x6');
save('train_y6.mat','train_y6');
save('test_x6.mat','test_x6');
save('test_y6.mat','test_y6');

kk=7;                      
numtrain=floor((kk-1)*numsize/k);
numtrain1=floor(kk*numsize/k);
train_x7=[num1([1:numtrain,numtrain1+1:end],:);num2([1:numtrain,numtrain1+1:end],:);num4([1:numtrain,numtrain1+1:end],:)];
train_y7=[new_label1([1:numtrain,numtrain1+1:end],:);new_label2([1:numtrain,numtrain1+1:end],:);new_label4([1:numtrain,numtrain1+1:end],:)];
test_x7=[num1(numtrain+1:numtrain1,:);num2(numtrain+1:numtrain1,:);num4(numtrain+1:numtrain1,:)];
test_y7=[new_label1(numtrain+1:numtrain1,:);new_label2(numtrain+1:numtrain1,:);new_label4(numtrain+1:numtrain1,:)];


ran1=randperm(size(train_x7,1));
train_x7=train_x7(ran1,:);
train_y7=train_y7(ran1,:);
ran2=randperm(size(test_x7,1));
test_x7=test_x7(ran2,:);
test_y7=test_y7(ran2,:);
ran3=randperm(size(train_x7,1));
train_x7=train_x7(ran3,:);
train_y7=train_y7(ran3,:);
ran4=randperm(size(test_x7,1));
test_x7=test_x7(ran4,:);
test_y7=test_y7(ran4,:);

save('train_x7.mat','train_x7');
save('train_y7.mat','train_y7');
save('test_x7.mat','test_x7');
save('test_y7.mat','test_y7');

kk=8;                
numtrain=floor((kk-1)*numsize/k);
numtrain1=floor(kk*numsize/k);
train_x8=[num1([1:numtrain,numtrain1+1:end],:);num2([1:numtrain,numtrain1+1:end],:);num4([1:numtrain,numtrain1+1:end],:)];
train_y8=[new_label1([1:numtrain,numtrain1+1:end],:);new_label2([1:numtrain,numtrain1+1:end],:);new_label4([1:numtrain,numtrain1+1:end],:)];
test_x8=[num1(numtrain+1:numtrain1,:);num2(numtrain+1:numtrain1,:);num4(numtrain+1:numtrain1,:)];
test_y8=[new_label1(numtrain+1:numtrain1,:);new_label2(numtrain+1:numtrain1,:);new_label4(numtrain+1:numtrain1,:)];


ran1=randperm(size(train_x8,1));
train_x8=train_x8(ran1,:);
train_y8=train_y8(ran1,:);
ran2=randperm(size(test_x8,1));
test_x8=test_x8(ran2,:);
test_y8=test_y8(ran2,:);
ran3=randperm(size(train_x8,1));
train_x8=train_x8(ran3,:);
train_y8=train_y8(ran3,:);
ran4=randperm(size(test_x8,1));
test_x8=test_x8(ran4,:);
test_y8=test_y8(ran4,:);

save('train_x8.mat','train_x8');
save('train_y8.mat','train_y8');
save('test_x8.mat','test_x8');
save('test_y8.mat','test_y8');

kk=9;                     
numtrain=floor((kk-1)*numsize/k);
numtrain1=floor(kk*numsize/k);
train_x9=[num1([1:numtrain,numtrain1+1:end],:);num2([1:numtrain,numtrain1+1:end],:);num4([1:numtrain,numtrain1+1:end],:)];
train_y9=[new_label1([1:numtrain,numtrain1+1:end],:);new_label2([1:numtrain,numtrain1+1:end],:);new_label4([1:numtrain,numtrain1+1:end],:)];
test_x9=[num1(numtrain+1:numtrain1,:);num2(numtrain+1:numtrain1,:);num4(numtrain+1:numtrain1,:)];
test_y9=[new_label1(numtrain+1:numtrain1,:);new_label2(numtrain+1:numtrain1,:);new_label4(numtrain+1:numtrain1,:)];


ran1=randperm(size(train_x9,1));
train_x9=train_x9(ran1,:);
train_y9=train_y9(ran1,:);
ran2=randperm(size(test_x9,1));
test_x9=test_x9(ran2,:);
test_y9=test_y9(ran2,:);
ran3=randperm(size(train_x9,1));
train_x9=train_x9(ran3,:);
train_y9=train_y9(ran3,:);
ran4=randperm(size(test_x9,1));
test_x9=test_x9(ran4,:);
test_y9=test_y9(ran4,:);

save('train_x9.mat','train_x9');
save('train_y9.mat','train_y9');
save('test_x9.mat','test_x9');
save('test_y9.mat','test_y9');

kk=10;                    
numtrain=floor((kk-1)*numsize/k);
numtrain1=floor(kk*numsize/k);
train_x10=[num1([1:numtrain,numtrain1+1:end],:);num2([1:numtrain,numtrain1+1:end],:);num4([1:numtrain,numtrain1+1:end],:)];
train_y10=[new_label1([1:numtrain,numtrain1+1:end],:);new_label2([1:numtrain,numtrain1+1:end],:);new_label4([1:numtrain,numtrain1+1:end],:)];
test_x10=[num1(numtrain+1:numtrain1,:);num2(numtrain+1:numtrain1,:);num4(numtrain+1:numtrain1,:)];
test_y10=[new_label1(numtrain+1:numtrain1,:);new_label2(numtrain+1:numtrain1,:);new_label4(numtrain+1:numtrain1,:)];


ran1=randperm(size(train_x10,1));
train_x10=train_x10(ran1,:);
train_y10=train_y10(ran1,:);
ran2=randperm(size(test_x10,1));
test_x10=test_x10(ran2,:);
test_y10=test_y10(ran2,:);
ran3=randperm(size(train_x10,1));
train_x10=train_x10(ran3,:);
train_y10=train_y10(ran3,:);
ran4=randperm(size(test_x10,1));
test_x10=test_x10(ran4,:);
test_y10=test_y10(ran4,:);

save('train_x10.mat','train_x10');
save('train_y10.mat','train_y10');
save('test_x10.mat','test_x10');
save('test_y10.mat','test_y10');




kk=1;
numtrain=floor(1*numsize/k);
train_x1=[num1(numtrain+1:end,:);num2(numtrain+1:end,:);num4(numtrain+1:end,:)];
train_y1=[new_label1(numtrain+1:end,:);new_label2(numtrain+1:end,:);new_label4(numtrain+1:end,:)];
test_x1=[num1(1:numtrain,:);num2(1:numtrain,:);num4(1:numtrain,:)];
test_y1=[new_label1(1:numtrain,:);new_label2(1:numtrain,:);new_label4(1:numtrain,:)];

%
ran1=randperm(size(train_x1,1));
train_x1=train_x1(ran1,:);
train_y1=train_y1(ran1,:);
ran2=randperm(size(test_x1,1));
test_x1=test_x1(ran2,:);
test_y1=test_y1(ran2,:);
ran3=randperm(size(train_x1,1));
train_x1=train_x1(ran3,:);
train_y1=train_y1(ran3,:);
ran4=randperm(size(test_x1,1));
test_x1=test_x1(ran4,:);
test_y1=test_y1(ran4,:);

%save the dataset
save('train_x1.mat','train_x1');
save('train_y1.mat','train_y1');
save('test_x1.mat','test_x1');
save('test_y1.mat','test_y1');


