%Use our eta in testing set as feature to robust fit just as CPM did.
rs_all = [];
for i = 10:10
    rs_list = [];
    for j = 1:10
        rs = [];
        for k = 1:i+1
            fold_ps_train = projection_pack_train_op{i,j}{k};
            fold_tags_train = tags_pack_train_op{i,j}{k};
            mdl=robustfit(fold_ps_train,fold_tags_train);
            fold_ps = projection_pack_test_op{i,j}{k};
            fold_tags = tags_pack_test_op{i,j}{k};
            y_predict = zeros(size(fold_tags));
            for m = 1:size(fold_ps,2)
                y_predict(m)=mdl(2)*fold_ps(m) + mdl(1); 
            end
            [r,p] = corr(y_predict(fold_tags>30)',fold_tags(fold_tags>30)');
            rs = [rs,abs(r)];
        end
        rs_list = [rs_list;rs];
    end
    rs_all = [rs_all;rs_list];
end


rs_alls = [];
lastmasks = [];
all_masks = [];
%10 realization plus 10 batch
% within each batch, we try to find the consistent pmap in cpm to improve robustness. If not, it
% will still use inconsistent pmaps
% Although we want to do that, we can not find any single consistent pmask. So
% this will not influence the results.
for k = 1:10
    rsc = [];
    lastmask = 0;
    for i = 1:10
        [y_predict,performance,pmasks,all_mask] = cpm_main(cpm_fcs',cpm_tags,'pthresh',0.1,'kfolds',10,'pmatrix',lastmask);
        if (lastmask == 0) & (size(pmasks,1) ~= 1)
            lastmask = pmasks;
            lastmasks = [lastmasks,lastmask];
        end
        all_masks = [all_masks,all_mask];
        rsc = [rsc,performance(1)];
    end
    rs_alls = [rs_alls;rsc];
end