clear all
 clc
 zz=[235 233 206 238 185 195 237 216 265 237 235 233 235 238 206];    %the datanumber of each clips
 P=zeros(695,12400);  
N=zeros(681,12400);
Z=zeros(634,12400);  %training dataset
P1=zeros(475,12400);  
N1=zeros(470,12400);
Z1=zeros(439,12400); %test dataset
 
for tt=1:15 
    f=zeros(zz(tt),12400);   
    for cc=1:zz(tt)        
    pilo=zeros(62,200);
    dt2=zeros(1,12400);
    display(tt) 
    display(cc)     
    real=['ys_0',num2str(tt),'_',num2str(cc),'_2.mat'];   
    load (real)
    dt2=reshape(pilo',1,12400);
    f(cc,:)=dt2; 
    end
          
    if tt==1
        P(1:235,:)=f;           
    end 
  
    if tt==2
       Z(1:233,:)=f;       
    end

     if tt==3
       N(1:206,:)=f;  
    end

     if tt==4
        N(207:(206+238),:)=f;  
     end    

     if tt==5
        Z(234:(233+185),:)=f;       
     end  

     if tt==6
       P(236:(235+195),:)=f;             
     end 

     if tt==7
       N((206+239):(206+238+237),:)=f;              
     end

     if tt==8
       Z((233+186):(233+185+216),:)=f;              
     end  

     if tt==9
       P((235+196):(235+195+265),:)=f;         
     end 
 
     if tt==10
       P1(1:237,:)=f;         
     end 

     if tt==11
       Z1(1:235,:)=f;         
     end 

     if tt==12
       N1(1:233,:)=f;         
     end 

     if tt==13
       Z1(236:(235+235),:)=f;         
     end 

     if tt==14
       P1(238:(237+238),:)=f;
     end

     if tt==15
       N1(234:(233+206),:)=f; 
     end

end


name1=['train_P_2.mat'];
%xlswrite(name1,P);
save(name1,'P');
name2=['train_N_2.mat'];
%xlswrite(name2,N);
save(name2,'N');
name3=['train_Z_2.mat'];
%xlswrite(name3,zhong);
save(name3,'Z');
name1=['test_P_2.mat'];
%xlswrite(name1,P);
save(name1,'P1');
name2=['test_N_2.mat'];
%xlswrite(name2,N);
save(name2,'N1');
name3=['test_Z_2.mat'];
%xlswrite(name3,zhong);
save(name3,'Z1');

