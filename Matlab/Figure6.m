% Figure 6
max_all = [];
for k =9:9
    max_list = [];
    max_rs_list = [];
    for i = 1:10
        % eta0_pack and eta1_pack is the projections using eigenvectors of
        % testing sets
        eta0 = eta0_pack{k,i};
        eta1 = eta1_pack{k,i};
        mas = [];
        max_rs = [];
        for j =1:size(eta0,2)
            e0 = eta0{1,j};
            e1 = eta1{1,j};
            tag = eta_tags_pack{k,i}{1,j};
            n_e0 = e0;
            n_e1 = e1;
            rs = [];
            for m = 0:100
                alpha = m*0.01;
                e = n_e0.*alpha + n_e1.*(1-alpha);
                [r,p] = corr(e',tag');
                rs = [rs,abs(r)];
            end
            [m,ma] = max(abs(rs));
            max_rs = [max_rs;rs];
            mas = [mas,(ma-1)*0.01];
        end
        max_list = [max_list;mean(mas)];
        max_rs_list = [max_rs_list;max_rs];
    end
    max_all = [max_all,max_list];
end