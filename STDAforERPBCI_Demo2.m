%%% Spatial-temoporal discriminant analysis (STDA) for ERP classification demo %%%
% 
% by Yu Zhang, ECUST & RIKEN, June 2012. Email: yuzhang@ecust.edu.cn
% 
% Reference:
% Y. Zhang, G. Zhou, Q. Zhao, J. Jin, X. Wang, A. Cichocki, "Spatial-temporal 
%   discriminant analysis for ERP-based brain-computer interface," 
%   IEEE Trans. Neural Syst. Rehabil. Eng., vol. 21, no. 2, pp.
%   233-243, Mar. 2013.
% 
% Note: This demo requires the tensor_toolbox developed by Kolda
% (http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.5.html)
% 

clear; clc; close all;

%% Load data
%load ERPdata                                    % ERP data (samples x channels x points); [trials,channels,datapoints]
%load flash_id                                   % Labels (stimulus id x target id)
[ERPdata, flash_id] = makeDataset('sub09');

%% Classification
method={'LDA','FC'};  
%method={'LDA','FC','STDA'};                     % compared methods for ERP classification
                                                % LDA: linear discriminant analysis without dimensionality reduction
                                                % FC: Fisher criterion based dimensionality reduction + LDA
                                                % STDA: spatial-temporal discriminant analysis based dimensionality reduction + LDA
                                        
n_meth=length(method);
n_ch=size(ERPdata,2);                           % number of channels
n_point=size(ERPdata,3);                        % number of points in each channel
n_trial=5;                                      % number of trials average
n_sti=4;                                        % number of visual stimuli
trialstart=1:n_sti*n_trial:size(flash_id,1);

n_command=length(trialstart);


%load cvorder                                   % cross-validation order (100 times)
   
%cv
if exist('cvorder.mat', 'file')
    load cvorder;
    cv_num = size(cvorder,1);
    n_test= n_command/cv_num;                   % number of trials for training
    n_train=n_command-n_test;                   % number of trials for test

else
    cv_num = 10;
    n_test= n_command/cv_num;                   % number of trials for training
    n_train=n_command-n_test;                   % number of trials for test
    initOrder = randperm(n_command);
    cvorder = initOrder;
    for ii = 1:cv_num-1
        cvorder = [cvorder;circshift(initOrder,[1, ii*n_test])];
    end
    save('cvorder.mat','cvorder')
end


n_rep=size(cvorder,1);                          % number of cross-validation
correctClassify=zeros(n_rep,n_trial,n_meth);    % number of correctly classified stimuli
correctClassifyImage=zeros(n_rep,n_trial,n_meth,n_sti);

%%% Do Cross-validation for Classification %%%
for mth=1:n_meth
    for nrp=1:n_rep
        fprintf('%s processing..., No.cross-validation: %d\n',method{mth},nrp);
        % select training data from random 8 commands
        testsubtrial=trialstart(cvorder(nrp,1:n_test));
        trainsubtrial=trialstart(cvorder(nrp,n_test+1:end));
        
        % extract training feature
        n_subtrial=n_sti*n_trial;               % number of subtrial in each trial
        traindata=zeros(n_train*n_subtrial,n_ch,n_point);
        testdata=zeros(n_test*n_subtrial,n_ch,n_point);
        for i=1:n_train
            traindata(1+(i-1)*n_subtrial:i*n_subtrial,:,:)=ERPdata(trainsubtrial(i):trainsubtrial(i)+n_subtrial-1,:,:);
            idtrain(1+(i-1)*n_subtrial:i*n_subtrial,:)=flash_id(trainsubtrial(i):trainsubtrial(i)+n_subtrial-1,:);
        end
        for i=1:n_test
            testdata(1+(i-1)*n_subtrial:i*n_subtrial,:,:)=ERPdata(testsubtrial(i):testsubtrial(i)+n_subtrial-1,:,:);
            idtest(1+(i-1)*n_subtrial:i*n_subtrial,:)=flash_id(testsubtrial(i):testsubtrial(i)+n_subtrial-1,:);
        end
        trainlabel=idtrain(:,2);
        trainlabel(trainlabel==1)=1;
        trainlabel(trainlabel==0)=2;
        target_id=idtest(idtest(:,2)~=0,1);
        traindata=permute(traindata,[2,3,1]);
        testdata=permute(testdata,[2,3,1]);

        %%% Feature Extraction %%%
        switch method{mth}
            case 'STDA'
                itrmax=200;                     % number of iteration for STDA
                [STDAmode error]=STDA(traindata,trainlabel',itrmax);
                fea_train=STDAprojection(traindata,STDAmode);
                fea_test=STDAprojection(testdata,STDAmode);
                fea_train=fea_train';
                fea_test=fea_test';

            case 'FC'
                [W d]=FCsf(traindata(:,:,trainlabel==1),traindata(:,:,trainlabel==2));
                ncop=2;
                subW=W(:,1:ncop);               % projection matrix
                fea_train=zeros(size(traindata,3),size(subW,2)*n_point);
                for pp=1:size(traindata,3)
                    mid=subW'*traindata(:,:,pp);
                    mid=mid';
                    mid=mid(:);
                    fea_train(pp,:)=mid;
                end
                fea_test=zeros(size(testdata,3),size(subW,2)*n_point);
                for pp=1:size(testdata,3)
                    mid=subW'*testdata(:,:,pp);
                    mid=mid';
                    mid=mid(:);
                    fea_test(pp,:)=mid;
                end

            case 'LDA'
                fea_train=reshape(traindata,n_ch*n_point,size(traindata,3))';
                fea_test=reshape(testdata,n_ch*n_point,size(testdata,3))';
        end

        %%% Classification using LDA %%%
        BasicLDAmode=BasicLDA(fea_train,trainlabel);
        [cScore pScore]=LDAClassify(fea_test,BasicLDAmode);
        pScore=pScore';

        for tt=1:n_test
            score=zeros(1,n_sti);
            for i=1:n_trial
                flashblock=1+(tt-1)*n_sti*n_trial+(i-1)*n_sti:(tt-1)*n_sti*n_trial+i*n_sti;
                %idtest(flashblock,1)
                score(idtest(flashblock,1))=score(idtest(flashblock,1))+pScore(flashblock);
                [value idx]=max(score);
                if idx==target_id(tt*n_trial)
                    correctClassify(nrp,i,mth)=correctClassify(nrp,i,mth)+1;
                    correctClassifyImage(nrp,i,mth,idx) = correctClassifyImage(nrp,i,mth,idx)+1;
                end
            end
        end
    end
end
%% Plot Accuracy
marker={'k-*','b-o','r-^'};
accuracy=100*correctClassify/n_test;
Maccuracy=squeeze(mean(accuracy,1))

accuracyImage=100*correctClassifyImage/(n_test/n_sti);
MaccuracyImage=squeeze(mean(accuracyImage,1));


if n_meth==1
    Maccuracy=Maccuracy';
end
for mth=1:n_meth
    SD=std(accuracy(:,:,mth));      % standard deviation
    errorbar(1:5,Maccuracy(:,mth),SD,marker{mth},'linewidth',1);
    hold on;
end
xlim([0.75 5.25]);
ylim([0 100]);
set(gca,'xtick',1:1:5,'xticklabel',1:1:5);
set(gca,'ytick',0:10:100,'yticklabel',0:10:100);
xlabel('Number of trials average');
ylabel('Accuracy (%)');
title('\bf ERP Classification Accuracy Comparison');
grid;
legend(method,'location','SouthEast');

