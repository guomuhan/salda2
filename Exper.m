% fea_train = fea(1:7291,:);
% gnd_train = gnd(1:7291,:);
% 
% fea_test = fea(7292:end,:);
% gnd_test = gnd(7292:end,:);

% load('train.mat')
% load('test.mat')

% img = fea(11,:);
% imshow(reshape(img,16,16)') % label = numer+1
clear

load('yale.mat');

% disp('Performing PCA (keeping 99.5% of variance)');
% X = data';
% [u,v]=pca(X);
% d95=find(cumsum(v)./sum(v)<0.985);
% mtr=mean(X,2);
% X=10.*u(d95,:)*bsxfun(@minus,X,mtr); % column
 data = X';


iterNum = 20;
rADLDA = zeros(iterNum,2); rLATR2 = zeros(iterNum,2); rLATR3 =zeros(iterNum,2);
rLMNN = zeros(iterNum,2); rLDA = zeros(iterNum,2); rMCC = zeros(iterNum,2);
rNMMP = zeros(iterNum,2); rLSDA = zeros(iterNum,2); rLFDA = zeros(iterNum,2);
rRLDA = zeros(iterNum,2); rNDA = zeros(iterNum,2); rLLDA = zeros(iterNum,2);
rbaseline = zeros(iterNum,1);


for iter = 1:iterNum
    clc
    iter
    [xTr, lTr, xTe, lTe] = genTrTe(data, label, 8);
    
    acc = knnclass(diag(ones(size(xTr,2),1)),xTr,lTr,xTe,lTe,1);
    rbaseline(iter,:) = acc;
%     

    % % ADLDA
%     tic;
%     feature_num = 60;
%     [~, W1, obj, interval_set] = ADLDA(xTr, lTr, feature_num, 1e-6); 
%     t = toc;
%     acc = knnclass(W1,xTr,lTr,xTe,lTe,1);
%     rADLDA(iter,:) = [acc, t];
%     
% %     tic;
% %     [~, W, obj] = LATR(xTr, lTr, feature_num, 1e-20); % row
% %     t = toc;
% %     acc = knnclass(W,xTr,lTr,xTe,lTe,1);
% %     rLATR2(iter,:) = [acc, t];
% %     
% %     tic;
% %     [~, W, obj] = LATR(xTr, lTr, feature_num, 1e-30); % row
% %     t = toc;
% %     acc = knnclass(W,xTr,lTr,xTe,lTe,1);
% %     rLATR3(iter,:) = [acc, t];
%     
%     % % LDA
%     tic;
%     [~, W] = LDA(xTr, lTr, 14, 'RatioTrace');
%     t = toc;
%     acc = knnclass(W,xTr,lTr,xTe,lTe,1);
%     rLDA(iter,:) = [acc, t];
% 
%     % % MCC
%     tic;
%     [~, W] = MMC(xTr, lTr, feature_num);
%     t = toc;
%     acc = knnclass(W,xTr,lTr,xTe,lTe,1);
%     rMCC(iter,:) = [acc, t];
% 
%     % % LMNN
%     tic;
%     [~, mapping] = compute_mapping([lTr,xTr], 'LMNN',4);
%     t = toc;
%     W = mapping.M;
%     acc = knnclass(W',xTr,lTr,xTe,lTe,1);
%     rLMNN(iter,:) = [acc, t];
% 
%     
%     % % NMMP
%     tic
%     W = NMMP_ijcai(xTr, lTr, 5, 10, feature_num);
%     t = toc;
%     acc = knnclass(W,xTr,lTr,xTe,lTe,1);
%     rNMMP(iter,:) = [acc, t];
%     
%     % % LSDA
%     options.k = 2;
%     options.ReducedDim=feature_num;
%     tic
%     [W] = LSDA(xTr, lTr, options);
%     t = toc;
%     acc = knnclass(W,xTr,lTr,xTe,lTe,1);
%     rLSDA(iter,:) = [acc, t];
%     
%     % % LFDA
%     tic
%     [W,~] = LFDA(xTr', lTr, feature_num,'weighted',4);
%     t = toc;
%     acc = knnclass(W,xTr,lTr,xTe,lTe,1);
%     rLFDA(iter,:) = [acc, t];
% 
%     
%     % % RLDA
%     temp2 = [];
%     for i = min(label):max(label)
%         temp1 = find(lTr==i);
%         temp2 = [temp2,length(temp1)];
%     end
%     W = RLDA(xTr', temp2); 
%     acc = knnclass(W,xTr,lTr,xTe,lTe,1);
%     rRLDA(iter,:) = [acc, t];
% 
% 
%     % % NDA
%     train.mat = xTr';
%     train.labels = lTr; % 1-totclass
%     train.N = size(xTr,1);
%     train.dim = size(xTr,2);
%     train.totclass = length(unique(lTr));
%     tic
%     [W, ~] = NDA(train, 5, feature_num, 1);
%     t = toc;
%     acc = knnclass(W,xTr,lTr,xTe,lTe,1);
%     rNDA(iter,:) = [acc, t];
%     
%     % % LLDA
%     tic
%     avrate = LLDA(xTr, lTr, xTe, lTe, feature_num, 18); % K = nTr*0.15
%     t = toc;
%     rLLDA(iter,:) = [avrate,t];

end

save('iter30_trnum8_nn1.mat','rbaseline','rADLDA','rLDA','rLFDA',...
    'rLLDA','rLMNN','rLSDA','rMCC','rNDA','rNMMP','rRLDA');


%% compute average acc, std and time
[acc, std, t] = metric(rbaseline,'baseline')
% [acc, std, t] = metric(rADLDA,'base')
% [acc, std, t] = metric(rLDA,'base')
% [acc, std, t] = metric(rLFDA,'base')
% [acc, std, t] = metric(rLSDA,'base')
% [acc, std, t] = metric(rLLDA,'base')
% [acc, std, t] = metric(rNMMP,'base')
% [acc, std, t] = metric(rNDA,'base')
% [acc, std, t] = metric(rRLDA,'base')
% [acc, std, t] = metric(rLMNN,'base')
% [acc, std, t] = metric(rMCC,'base')




