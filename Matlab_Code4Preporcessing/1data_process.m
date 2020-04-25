clear all
load 10_90.mat
num=zeros(1,9);
for i=1:9
nam_s=strcat('code',num2str(i),'0');
zzz=eval(nam_s);
aaa=length(zzz);
num(i)=fix(aaa/1000);   %1s为1个epoch
for j=1:num(i)
pilo=zzz(:,((j-1)*1000+1) :5: j*1000);  %隔点降采样至200hz
nam_s=strcat('cq_0',num2str(i),'_',num2str(j));
nam_s=strcat(nam_s,'.mat');
%xlswrite(nam_s,pilo);
save(nam_s,'pilo');
end
end