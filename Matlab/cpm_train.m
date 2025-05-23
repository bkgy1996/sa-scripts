function [r,p,pmask,mdl]=cpm_train(x,y,pthresh)
% Train a Connectome-based Predictive Model
% x            Predictor variable
% y            Outcome variable
% pthresh      p-value threshold for feature selection
% r            Correlations between all x and y
% p            p-value of correlations between x and y
% pmask        Mask for significant features
% mdl          Coefficient fits for linear model relating summary features to y

% Select significant features
[r,p]=corr(x',y);
pmask=(+(r>0))-(+(r<0)); 
pmask=pmask.*(+(p<pthresh));

% For each subject, summarize selected features 
for i=1:size(x,2)
    a = nanmean(x(pmask>0,i));
    b = nanmean(x(pmask<0,i));
    if (~isnan(a)) && (~isnan(b))
        summary_feature(i)=nanmean(x(pmask>0,i))-nanmean(x(pmask<0,i));
    elseif ~isnan(a)
        summary_feature(i)=nanmean(x(pmask>0,i));
    elseif ~isnan(b)
        summary_feature(i)=-nanmean(x(pmask<0,i));
    else
        summary_feature(i)=nan;
    end
end
% Fit y to summary features
mdl=robustfit(summary_feature,y');
% size(mdl)
    
