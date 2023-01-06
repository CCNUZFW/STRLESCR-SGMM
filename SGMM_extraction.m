clc;clear;close all;
disp('Code start running...');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% % Step1:Extract MFCC. % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ce_train=cell(45,514);
ce_test=cell(45,128);

for n=1:45
    name={'Apple_air1','Apple_air2','Apple_air2(1)','Honor9 STF-ALOO','huawei Novo2s',...
          'huawei_honor8(2)','huawei_honor10','huawei_nova','huawei_nova3e','huawei_p10',...
          'huawei_p20','huawei_TAG_AL00','huwei_honor7x','huwei_honor8','huwei_honor8(1)',...
          'huwei_honorV8','ipad7','IPHONE_SE','IPHONE6','IPHONE6(1)',...
          'IPHONE6(2)','IPHONE6(3)','IPHONE6S','IPHONE6S(1)','IPHONE6S(2)',...
          'IPHONE7plus','IPHONEX','nubia_Z11','oppo_R9s','Redmi 3S',...
          'Redmi Note4x','SAMSUNG_S8','SPH-D710','vivo_X3F','vivo_x7',...
          'vivo_Y11t','xiaomi note3','xiaomi_mix2','xiaomi2s','xiaomi5',...
          'xiaomi8','xiaomi8se','xiaomi8se(1)','ZTE C880A','ZTE G719C'};

   for i=1:642   
       train_file=char(strcat('C:\Users\fsx10\Desktop\proj_keras\CCNU_Mobile Dataset\',...
                 name(n),'\',name(n),'_',num2str(i),'.wav'));

       [sample,fs]=audioread(train_file);
       mfcc=melcepst(sample,fs,'E0dD',...
            12,floor(3*log(fs)),pow2(floor(log2(0.03*fs))),...
            pow2(floor(log2(0.03*fs)))/2,...
            0,0.5);
       
       mfcc=mfcc(:,1:640)';
       if i<=514
           ce_train{n,i}=mfcc;
       else
           ce_test{n,i-514}=mfcc;
       end
   end
end

csvwrite('MFCC_train.csv',ce_train);
csvwrite('MFCC_test.csv',ce_test);
disp('MFCC extraction completed.');
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% % Step2:Build UBM. % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load('ce.mat');
% ubm=gmm_em(ce(:),16,32,1,16); 
% %%% %(dataList, nmix, final_niter, ds_factor, nworkers)
load('UBM.mat');
disp('UBM build completed.');
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% % Step3:Extract GSV. % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ce_trains=cell(45,514*80);
ce_tests=cell(45,128*80);
for i=1:45
    for j=1:514
        for k=1:80
            ce_trains{i,(j-1)*80+k}=ce_train{i,j}(:,(k-1)*8+1:k*8);
        end
    end
    for j=1:128
        for k=1:80
            ce_tests{i,(j-1)*80+k}=ce_test{i,j}(:,(k-1)*8+1:k*8);
        end
    end
end
disp('Data reshape completed.');

disp('Adapt the UBM to each speaker');
map_tau = 10.0;
config = 'mwv';
pNum=45;
nSpeakers= pNum;
nChannels= 1;

gmm_train = cell(pNum*514*80, 1);
gmms_train = cell(pNum*514*80, 1);
gmms_trains = zeros(45*514*80,12,16);
gmms_trainss = zeros(45*514,12*16*80);
gmm_test = cell(pNum*128*80, 1);
gmms_test = cell(pNum*128*80, 1);
gmms_tests = zeros(45*128*80,12,16);
gmms_testss = zeros(45*128,12*16*80);
for s=1:nSpeakers
    disp(['for the ',num2str(s),' speaker...']);
    for i=1:514
        for k=1:80
            gmm_train{(((s-1)*514+i-1)*80+k)} = mapAdapt(ce_trains(s,(i-1)*80+k), ubm, map_tau, config);
            gmms_train{(((s-1)*514+i-1)*80+k)} = mapminmax(gmm_train{(((s-1)*514+i-1)*80+k)}.mu);
            gmms_trains(((s-1)*514+i-1)*80+k,:,:) = gmms_train{(((s-1)*514+i-1)*80+k)};
            for d=1:12
                gmms_trainss((s-1)*514+i,((k-1)*12+d-1)*16+1:((k-1)*12+d)*16) = gmms_trains((((s-1)*514+i-1)*80+k),d,:);
            end
        end
    end
    for j=1:128
        for k=1:80
            gmm_test{(((s-1)*128+j-1)*80+k)} = mapAdapt(ce_tests(s,(j-1)*80+k), ubm, map_tau, config);
            gmms_test{(((s-1)*128+j-1)*80+k)} = mapminmax(gmm_test{(((s-1)*128+j-1)*80+k)}.mu);
            gmms_tests((((s-1)*128+j-1)*80+k),:,:) = gmms_test{(((s-1)*128+j-1)*80+k)};
            for d=1:12
                gmms_testss((s-1)*128+j,((k-1)*12+d-1)*16+1:((k-1)*12+d)*16) = gmms_tests((((s-1)*128+j-1)*80+k),d,:);
            end
        end
    end
end

csvwrite('SGMM_train.csv',gmms_trainss);
csvwrite('SGMM_test.csv',gmms_testss);
disp('Finished');