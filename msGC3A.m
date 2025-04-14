

clear all;
close all;
% Please download the SSVEP benchmark dataset and BETA  dataset for this code

Fs=250; % sample rate

dataset_no=1;
if dataset_no==1
    str_dir='..\..\2016_Tsinghua_SSVEP_database\'; %更改成自己的数据集地址
    latencyDelay = round(0.14*Fs);      % latency
    num_of_subj=35;                     % Number of subjects (35 if you have the benchmark dataset)
    ch_used=[48 54 55 56 57 58 61 62 63]; %[48 54 55 56 57 58  62 ][  54 55 56 57 58 61 62 63][48  55  57  61 62 63] Pz, PO5, PO3, POz, PO4, PO6, O1,Oz, O2 (in SSVEP benchmark dataset)
    num_of_trials=5;                    % Number of training trials (1<=num_of_trials<=5)
    pha_val=[0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 ...
        0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5]*pi;
    sti_f=[8.0:1:15.0, 8.2:1:15.2,8.4:1:15.4,8.6:1:15.6,8.8:1:15.8];
    n_sti=length(sti_f);                     % number of stimulus frequencies
    [~,target_order]=sort(sti_f);
    sti_f=sti_f(target_order);
elseif dataset_no==2
    str_dir='..\..\BETA SSVEP dataset\';  %更改成自己的数据集地址
    latencyDelay = round(0.13*Fs);      % latency
    num_of_subj=70;
    ch_used=[48 54 55 56 57 58 61 62 63]; % Pz, PO5, PO3, POz, PO4, PO6, O1,Oz, O2 (in SSVEP benchmark dataset)
    num_of_trials=3;                    % Number of training trials (1<=num_of_trials<=3)
    pha_val=[0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 ...
        0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5]*pi;
    sti_f=[8.6:0.2:15.8,8.0 8.2 8.4];
    n_sti=length(sti_f);                     % number of stimulus frequencies
    [~,target_order]=sort(sti_f);
    sti_f=sti_f(target_order);
else
end


num_of_harmonics=5;                 % for all cca-based methods
num_of_signal_templates=11;         % for mscca (1<=num_of_signal_templates<=40) and msGC3A
num_of_subbands=5;                  % for filter bank analysis
FB_coef0=[1:num_of_subbands].^(-1.25)+0.25; % for filter bank analysis



% time-window length (min_length:delta_t:max_length)
min_length=0.3;
delta_t=0.1;
max_length=1.0;                           % [min_length:delta_t:max_length]

TW = min_length:delta_t:max_length;
TW_p = round(TW*Fs);
enable_bit=[1];                       % Select the algorithms
is_center_std=0;                      % 0: without , 1: with (zero mean, and unity standard deviation)


% Chebyshev Type I filter design
for k=1:num_of_subbands
    Wp = [(8*k)/(Fs/2) 90/(Fs/2)];
    Ws = [(8*k-2)/(Fs/2) 100/(Fs/2)];
    [N,Wn] = cheb1ord(Wp,Ws,3,40);
    [subband_signal(k).bpB,subband_signal(k).bpA] = cheby1(N,0.5,Wn);
end
%notch
Fo = 50;
Q = 35;
BW = (Fo/(Fs/2))/Q;

[notchB,notchA] = iircomb(Fs/Fo,BW,'notch');
seed = RandStream('mt19937ar','Seed','shuffle');


for tw_length=1:length(TW)
    sig_len=TW_p(tw_length);
    
    clear y_sb 
    
    for sn=1:num_of_subj 
        tic
        if dataset_no==1
            load(strcat(str_dir,'S',num2str(sn),'.mat'));
        elseif dataset_no==2
            load([str_dir 'S' num2str(sn) '.mat']);
            eegdata=data.EEG;
            data = permute(eegdata,[1 2 4 3]);
        else
%             load(strcat(str_dir,'\','exampleData.mat'));
        end
        
        %  pre-stimulus period: 0.5 sec
        %  latency period: 0.14 sec
        eeg=data(ch_used,floor(0.5*Fs)+1:floor(0.5*Fs+latencyDelay)+sig_len,:,:);
        
        [d1_,d2_,d3_,d4_]=size(eeg);
        d1=d3_;d2=d4_;d3=d1_;d4=d2_;
        no_of_class=d1;
        n_ch = d3;
        % d1: num of stimuli
        % d2: num of trials
        % d3: num of channels % Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
        % d4: num of sampling points
        if sn==1
            for sub_band=1:num_of_subbands
                subband_signal(sub_band).SSVEPdata = zeros(n_ch,sig_len,d2,d1);
                subband_signal(sub_band).signal_template = zeros(n_ch,sig_len,d1);
            end
        end
        
        for i=1:1:d1
            for j=1:1:d2
                y0=reshape(eeg(:,:,i,j),d3,d4);
%                 SSVEPdata(:,:,j,i)=reshape(y0,d3,d4,1,1);
                y = filtfilt(notchB, notchA, y0.'); %notch
                y = y.';
                for sub_band=1:num_of_subbands
                    
                    for ch_no=1:d3
                        tmp2=filtfilt(subband_signal(sub_band).bpB,subband_signal(sub_band).bpA,y(ch_no,:));
                        y_sb(ch_no,:) = tmp2(latencyDelay+1:latencyDelay+sig_len);
                    end
                    
                    subband_signal(sub_band).SSVEPdata(:,:,j,i)=reshape(y_sb,d3,length(y_sb),1,1);
                end
                
            end
        end
        
        clear eeg
        %% Initialization
%         n_ch=size(SSVEPdata,1);
        n_run=d2;                                % number of used runs
        
%         SSVEPdata=SSVEPdata(:,:,:,target_order);
        for sub_band=1:num_of_subbands
            subband_signal(sub_band).SSVEPdata=subband_signal(sub_band).SSVEPdata(:,:,:,target_order); % To sort the orders of the data as 8.0, 8.2, 8.4, ..., 15.8 Hz
        end        
        
        FB_coef=FB_coef0'*ones(1,n_sti);
        n_correct=zeros(1,1); % Count how many correct detection        
        
        seq_0=zeros(d2,num_of_trials);
        for run=1:d2
            %         % leave-one-run-out cross-validation
            if (num_of_trials==1)
                seq1=run;
            elseif (num_of_trials==d2-1)
                seq1=[1:n_run];
                seq1(run)=[];
            else
                % leave-one-run-out cross-validation
                % Randomly select the trials for training
                isOK=0;
                while (isOK==0)
                    seq=randperm(seed,d2);
                    seq1=seq(1:num_of_trials);
                    seq1=sort(seq1);
                    if isempty(find(sum((seq1'*ones(1,d2)-seq_0').^2)==0))
                        isOK=1;
                    end
                end
                
            end
            idx_traindata=seq1; % index of the training trials
            idx_testdata=1:n_run; % index of the testing trials
            idx_testdata(seq1)=[];
            
            for i=1:no_of_class

                for k=1:num_of_subbands
                    if length(idx_traindata)>1
                        subband_signal(k).signal_template(:,:,i)=mean(subband_signal(k).SSVEPdata(:,:,idx_traindata,i),3);
                    else
                        subband_signal(k).signal_template(:,:,i)=subband_signal(k).SSVEPdata(:,:,idx_traindata,i);
                    end
                end
  
            end
            
            
            for run_test=1:length(idx_testdata)   
                clear Xa Xa_train             
                test_signal=zeros(d3,sig_len);
                fprintf('Testing TW %fs, No.crossvalidation %d \n',TW(tw_length),idx_testdata(run_test));
 
                for i=1:no_of_class
                    
                    
                    for sub_band=1:num_of_subbands
                        test_signal=subband_signal(sub_band).SSVEPdata(:,1:TW_p(tw_length),idx_testdata(run_test),i);
                        if (is_center_std==1)
                            test_signal=test_signal-mean(test_signal,2)*ones(1,length(test_signal));
                            test_signal=test_signal./(std(test_signal')'*ones(1,length(test_signal)));
                        end
                        
                        for jj=1:no_of_class
                            
                            template=subband_signal(sub_band).signal_template(:,[1:sig_len],jj);
                            if (is_center_std==1)
                                template=template-mean(template,2)*ones(1,length(template));
                                template=template./(std(template')'*ones(1,length(template)));
                            end
                            
                            % Generate the sine-cosine reference signal
                            ref1=ref_signal_nh(sti_f(jj),Fs,pha_val(jj),sig_len,num_of_harmonics);
                    
                            
                            if (jj==1)&&(i==1)

                                W_msgcca(sub_band).wg1=[];
                                W_msgcca(sub_band).wg2=[];
                                %W_msgcca(sub_band).wg3=[];

                                W_gcca(sub_band).wg1_ref=[];
                                W_gcca(sub_band).wg2_ref=[];
                                %W_gcca(sub_band).wg3_ref=[];

                                for j=1:no_of_class
                                    % find the indices of neighboring templates
                                    d0=floor(num_of_signal_templates/2);
                                    if j<=d0
                                        template_st=1;
                                        template_ed=num_of_signal_templates;
                                    elseif ((j>d0) && j<(d1-d0+1))
                                        template_st=j-d0;
                                        template_ed=j+(num_of_signal_templates-d0-1);
                                    else
                                        template_st=(d1-num_of_signal_templates+1);
                                        template_ed=d1;
                                    end
                                    mscca_template=[];
                                    mscca_ref=[];
                                    template_seq=[template_st:template_ed];

                                    % Concatenation of the templates (or sine-cosine references)
                                    for n_temp=1:num_of_signal_templates
                                        template0=subband_signal(sub_band).signal_template(:,1:sig_len,template_seq(n_temp));
                                        if (is_center_std==1)
                                            template0=template0-mean(template0,2)*ones(1,length(template0));
                                            template0=template0./(std(template0')'*ones(1,length(template0)));
                                        end
                                        ref0=ref_signal_nh(sti_f(template_seq(n_temp)),Fs,pha_val(template_seq(n_temp)),sig_len,num_of_harmonics);
                                        mscca_template=[mscca_template;template0'];
                                        mscca_ref=[mscca_ref;ref0'];
                                    end
                                    % ========mscca spatial filter=====
                                    %第一层mscca空间滤波
                                    [Wx1,Wy1,cra]=canoncorr(mscca_template,mscca_ref(:,1:end));
                                    spatial_filter1(sub_band,j).wx1=Wx1(:,1)';                                   
                                    spatial_filter1(sub_band,j).wy1=Wy1(:,1)';

                                    % 产生替代正余弦谐波的模板
                                    sc=ref_signal_nh(sti_f(j),Fs,pha_val(j),sig_len,num_of_harmonics);  
                                    SS=[];

                                    for tr=1:num_of_trials
                                        X0=subband_signal(sub_band).SSVEPdata(:,1:sig_len,idx_traindata(tr),j);
                                        S0=Wx1(:,1:4)'*X0;   %保留前L个特征向量
                                        SS=[SS;S0]; 
                                    end                                
                                    spatial_filter1(sub_band,j).wq=SS;
                                    % ========msGC3A spatial filter=====

                                    X0=subband_signal(sub_band).SSVEPdata(:,1:sig_len,idx_traindata,j);

                                    X1 = reshape(X0, d3, sig_len*num_of_trials);
                                    X2 = repmat(squeeze(mean(X0,3)), 1, num_of_trials);
                                    X3 = repmat(SS, 1, num_of_trials);  %替代的模板
                                  
                                    X = [X1;X2;X3];
                                    D = blkdiag(X1*X1',X2*X2',X3*X3');
                                    [V, P] = eig(D\(X*X'));
                                    P = diag(P);
                                    [~, index] = sort(P, 'descend');
                                    w = V(:,index(1));
                                    spatial_filter1(sub_band,j).wg1 = w(1:d3, :)';
                                    spatial_filter1(sub_band,j).wg2 = w(d3+1:2*d3, :)';
                                    spatial_filter1(sub_band,j).wg3 = w(2*d3+1:end, :)';
                                    W_msgcca(sub_band).wg1=[ W_msgcca(sub_band).wg1;spatial_filter1(sub_band,j).wg1];
                                    W_msgcca(sub_band).wg2=[ W_msgcca(sub_band).wg2;spatial_filter1(sub_band,j).wg2];
                                    %W_msgcca(sub_band).wg3=[ W_msgcca(sub_band).wg3;spatial_filter1(sub_band,j).wg3];

                                    %egcca
                                    X3_ref = repmat(sc, 1, num_of_trials);
                                    X_ref = [X1;X2;X3_ref];
                                    D_ref = blkdiag(X1*X1',X2*X2',X3_ref*X3_ref');
                                    [V_ref, P_ref] = eig(D_ref\(X_ref*X_ref'));
                                    P_ref = diag(P_ref);
                                    [~, index_ref] = sort(P_ref, 'descend');
                                    w_ref = V_ref(:,index_ref(1));
                                    spatial_filter1(sub_band,j).wg1_ref = w_ref(1:d3, :)';
                                    spatial_filter1(sub_band,j).wg2_ref = w_ref(d3+1:2*d3, :)';
                                    spatial_filter1(sub_band,j).wg3_ref = w_ref(2*d3+1:end, :)';
                                    W_gcca(sub_band).wg1_ref=[ W_gcca(sub_band).wg1_ref;spatial_filter1(sub_band,j).wg1_ref];
                                    W_gcca(sub_band).wg2_ref=[ W_gcca(sub_band).wg2_ref;spatial_filter1(sub_band,j).wg2_ref];
                                    %W_gcca(sub_band).wg3_ref=[ W_gcca(sub_band).wg3_ref;spatial_filter1(sub_band,j).wg3_ref];
                                end

                            end

                            %mscca
                             cr1=corrcoef((spatial_filter1(sub_band,jj).wx1*test_signal)',(spatial_filter1(sub_band,jj).wy1*ref1)');
                            
                            %msGC3A
                            
                             cr2 = corrcoef(W_msgcca(sub_band).wg1*test_signal,W_msgcca(sub_band).wg2*template);
                           
                            %egcca
                             cr3  = corrcoef(W_gcca(sub_band).wg1_ref*test_signal,W_gcca(sub_band).wg2_ref*template);
                            
                            msGCCCA(sub_band,jj)=sign(cr1(1,2))*cr1(1,2)^2+sign(cr2(1,2))*cr2(1,2)^2+sign(cr3(1,2))*cr3(1,2)^2;

                        end
                    end
                    msGC3A1=sum((msGCCCA).*FB_coef,1);

                    [~,idx]=max(msGC3A1);
                    if idx==i
                        n_correct(1)=n_correct(1)+1;
                    end
                end
                %             end
            end
            idx_train_run(run,:)=idx_traindata;
            idx_test_run(run,:)=idx_testdata;
            seq_0(run,:)=seq1;
        end

        
        %% Save results
        toc
        accuracy=100*n_correct/n_sti/n_run/length(idx_testdata)
        all_sub_acc(tw_length,sn,:)=accuracy;
        
        % all_sub_acc(:,:,i)
   
        
            itrs(1) = itr(n_sti, accuracy(1)/100, TW(tw_length)+0.5);
            fprintf('ITR = %2.2f bpm\n', itrs(1));
            all_sub_itr(tw_length,sn,:)=itrs;
       
        disp(sn) 
        if dataset_no==1
           save save_all_sub_acc_th.mat all_sub_acc
           save save_all_sub_itr_th.mat all_sub_itr
        elseif dataset_no==2
           save save_all_sub_acc_beta.mat all_sub_acc
          save save_all_sub_itr_beta.mat all_sub_itr
        else
        end   
    end
end
