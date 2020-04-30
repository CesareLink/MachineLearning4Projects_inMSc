num=zeros(1,15);   %set up the data split array

%start the data load process
for i=1:15
%make the name of data
nam_s=strcat('djc_eeg',num2str(i));
zzz=eval(nam_s);

%split the data for 1s as a epoch
aaa=length(zzz);
num(i)=fix(aaa/200);  %the default sample frequency is 1000Hz, down-sample to 200Hz

%save the clips in mat
for j=1:num(i)
pilo=zzz(:,((j-1)*200+1) : j*200); 
nam_s=strcat('djc_0',num2str(i),'_',num2str(j),'_2');
nam_s=strcat(nam_s,'.mat');

%save in excel or not
%xlswrite(nam_s,pilo);
save(nam_s,'pilo');
end
end        