clear all
 clc
%put the samples into the corresponding category
zz=[235 233 206 238 185 195 237 216 265 237 235 233 235 238 206];    %the sample number in each clips
P=zeros(1170,12400);  
N=zeros(1120,12400);
Z=zeros(1104,12400);  %
 
%input data shape is [x,62*200]
for tt=1:15 
    f=zeros(zz(tt),12400);  
    for cc=1:zz(tt)         
    pilo=zeros(62,200);
    dt2=zeros(1,12400);
    display(tt) 
    display(cc)     
    real=['train_0',num2str(tt),'_',num2str(cc),'_1.mat'];   
    load (real)
    dt2=reshape(pilo',1,12400);
    f(cc,:)=dt2; 
    end

%put the samples into the corresponding category
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
       P((235+195+266):(235+195+265+237),:)=f;         
     end 

     if tt==11
       Z((233+185+217):(233+185+216+235),:)=f;         
     end 

     if tt==12
       N((206+238+238):(206+238+237+233),:)=f;         
     end 

     if tt==13
       Z((233+185+216+236):(233+185+216+235+235),:)=f;         
     end 

     if tt==14
       P((235+195+265+238):(235+195+265+237+238),:)=f;
     end

     if tt==15
       N((206+238+237+234):(206+238+237+233+206),:)=f; 
     end

end

%save the train data for each category
name1=['train_P_1.mat'];
%xlswrite(name1,P);
save(name1,'P');
name2=['train_N_1.mat'];
%xlswrite(name2,N);
save(name2,'N');
name3=['train_Z_1.mat'];
%xlswrite(name3,zhong);
save(name3,'Z');

