clear all
 clc
 zz=[594 329 439 351 417 269 504 240 204];    %ÿһ��ӰƬ����Ŀ
 bei =zeros(863,6000);   
 xi =zeros(1119,6000);
 kong= zeros(948,6000);   %
 zhong=zeros(417,6000);
for tt=1:9 
    f=zeros(zz(tt),6000);   %����ÿһ����Ӱ���������������е���ʽ
    for cc=1:zz(tt)         %��ȡÿһ����Ӱ��ÿһ��������תΪ�е���ʽ
    pilo=zeros(30,200);
    dt2=zeros(1,6000);
    display(tt) 
    display(cc)     %tt��ʾ����film�ı��
    real=['ls_0',num2str(tt),'_',num2str(cc),'.mat'];   
    load (real)
    dt2=reshape(pilo',1,6000);
    f(cc,:)=dt2; 
    end
          %��ͬһ����������ϵ�һ��
    if tt==1
        bei(1:594,:)=f;
    end
    if tt==2
        xi(1:329,:)=f;
    end
    if tt==3
        xi(330:768,:)=f;
    end
    if tt==4
        xi(769:1119,:)=f;
    end
    if tt==5
        zhong(1:417,:)=f;
    end
    if tt==6
        bei(595:863,:)=f;
    end
    if tt==7
        kong(1:504,:)=f;
    end
    if tt==8
        kong(505:744,:)=f;
    end
    if tt==9
        kong(745:948,:)=f;
    end   
end
 name1=['ls_bei.xlsx'];
xlswrite(name1,bei);
name2=['ls_xi.xlsx'];
xlswrite(name2,xi);
name3=['ls_kong.xlsx'];
xlswrite(name3,kong);
name4=['ls_zhong.xlsx'];
xlswrite(name4,zhong);
  
