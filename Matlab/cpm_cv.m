function [y_predict,consist_pmasks,pmasks]=cpm_cv(x,y,pthresh,kfolds,lastmasks)
% Runs cross validation for CPM
% x            Predictor variable
% y            Outcome variable
% pthresh      p-value threshold for feature selection
% kfolds       Number of partitions for dividing the sample
% y_test       y data used for testing
% y_predict    Predictions of y data used for testing

% Split data
nsubs=size(x,2);
randinds=randperm(nsubs);
ksample=floor(nsubs/kfolds);
rs = zeros(kfolds,1);
ps = zeros(kfolds,1);
y_predict = zeros(nsubs, 1);
consist_pmasks = lastmasks;
% Run CPM over all folds
% We first try to find consistent pmasks, if not it will return the
% y_predict as original CPM
if size(consist_pmasks,1) == 1
    fprintf('\n# finding consist %1.0f Folds.\nPerforming fold no. ',kfolds);
    for leftout = 1:kfolds
        fprintf('%1.0f ',leftout);
        
        if kfolds == nsubs % doing leave-one-out
            testinds=randinds(leftout);
            traininds=setdiff(randinds,testinds);
        else
            si=1+((leftout-1)*ksample);
            fi=si+ksample-1;
            % here, to make it comparable with our 10-fold, we use less
            % data to train
            traininds=randinds(si:fi);
            testinds=setdiff(randinds,traininds);
            % testinds=randinds(si:fi);
            % traininds=setdiff(randinds,testinds);
        end
        
        % Assign x and y data to train and test groups 
        x_train = x(:,traininds);
        y_train = y(traininds);
        x_test = x(:,testinds);
        y_test = y(testinds);
        
        % Train Connectome-based Predictive Model
        [~, ~, pmask, mdl] = cpm_train(x_train, y_train,pthresh);
        
        % Test Connectome-based Predictive Model
        [y_predict(testinds)] = cpm_test(x_test,mdl,pmask);
        pmasks(:,leftout) = pmask;
        if size(consist_pmasks,1) == 1
            consist_pmasks = pmask;
        else
            consist_pmasks(pmask~=consist_pmasks) = 0;
        end
    end
else
    pmasks = consist_pmasks;
end
% if it can find consistent p_masks it will use such p_masks to redo the CV
% and replace the old y_predict with the improved y_predict
if ~all(consist_pmasks==0)
    nsubs=size(x,2);
    randinds=randperm(nsubs);
    ksample=floor(nsubs/kfolds);
    rs = zeros(kfolds,1);
    ps = zeros(kfolds,1);
    y_predict = zeros(nsubs, 1);
    % Run CPM over all folds
    fprintf('\n# Running over %1.0f Folds.\nPerforming fold no. ',kfolds);
    for leftout = 1:kfolds
        fprintf('%1.0f ',leftout);
        
        if kfolds == nsubs % doing leave-one-out
            testinds=randinds(leftout);
            traininds=setdiff(randinds,testinds);
        else
            si=1+((leftout-1)*ksample);
            fi=si+ksample-1;
        
            traininds=randinds(si:fi);
            testinds=setdiff(randinds,traininds);
            % testinds=randinds(si:fi);
            % traininds=setdiff(randinds,testinds);
        end
        
        % Assign x and y data to train and test groups 
        x_train = x(:,traininds);
        y_train = y(traininds);
        x_test = x(:,testinds);
        y_test = y(testinds);
        
        % Train Connectome-based Predictive Model
    
        
        % For each subject, summarize selected features 
        % size(x)
        summary_feature = [];
        for i=1:size(x_train,2)
            a = nanmean(x_train(consist_pmasks>0,i));
            b = nanmean(x_train(consist_pmasks<0,i));
            if (~isnan(a)) && (~isnan(b))
                summary_feature(i)=a-b;
            elseif ~isnan(a)
                summary_feature(i)=a;
            elseif ~isnan(b)
                summary_feature(i)=-b;
            else
                summary_feature(i)=nan;
            end
        end
        % size(summary_feature)
        % Fit y to summary features
        mdl=robustfit(summary_feature,y_train');
        
        % Test Connectome-based Predictive Model
        [y_predict(testinds)] = cpm_test(x_test,mdl,consist_pmasks);
    end
else
    fprintf('no consistent p value ');
    consist_pmasks = 0;
end
