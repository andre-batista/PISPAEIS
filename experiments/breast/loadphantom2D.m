function [epsilon_r,sigma_eff,dx,dy,mi] = loadphantom2D ...
    (filepath,f,epsilon_rb,sigma_b,new_res)

    % Constants
    omega = 2*pi*f;
    dx = .5e-3; dy = .5e-3; dz = .5e-3;
    epsilon_0 = 8.854187817e-12;
    
    % Cole-Cole parameters - 500MHz < f < 20GHz
    epsilon_inf = [2.293, 2.908, 3.140, 4.031, 9.941, 7.821, 6.151, 1.];
    epsilon_delta = [.141, 1.2, 1.708, 3.654, 26.6, 41.48, 48.26, 66.31];
    tau = 1e-12 * [16.4, 16.88, 14.65, 14.12, 10.9, 10.66, 10.26, 7.585];
    alpha = [.251, .069, .061, .055, .003, .047, .049, .063];
    sigma_s = [.002, .02, .036, .083, .462, .713, .809, 1.37];

    bi_file = fopen([filepath,'/breastInfo.txt'],'r');
    breastInfo = fscanf(bi_file,'%s');
    fclose(bi_file);

    m_file = fopen([filepath '/mtype.txt'],'r');
    mi = fscanf(m_file,'%f');
    fclose(m_file);
    
    p_file = fopen([filepath '/pval.txt'],'r');
    p_i = fscanf(p_file,'%f');
    fclose(p_file);
    
    epsilon_star = epsilon_inf ...
        + epsilon_delta./(1+(1j*omega*tau).^(1-alpha)) ...
        + sigma_s/1j/omega/epsilon_0;
    
    bound_epsilon_r = real(epsilon_star);
    bound_sigma_eff = -imag(epsilon_star)*omega*epsilon_0;
    
    epsilon_r = epsilon_rb * ones(size(mi));
    sigma_eff = sigma_b * ones(size(mi));
    
    tissues = [3.3, 3.2, 3.1, 2, 1.3, 1.2, 1.1];
    
    for i = 1:length(tissues)
        
        epsilon_r(mi==tissues(i)) = p_i(mi==tissues(i))*bound_epsilon_r(i+1)...
            +(1-p_i(mi==tissues(i)))*bound_epsilon_r(i);
        
        sigma_eff(mi==tissues(i)) = p_i(mi==tissues(i))*bound_sigma_eff(i+1)...
            +(1-p_i(mi==tissues(i)))*bound_sigma_eff(i);
        
    end
    
    epsilon_r(mi==-4) = max(bound_epsilon_r);
    sigma_eff(mi==-4) = max(bound_sigma_eff);
    
    epsilon_r(mi==-2) = mean(bound_epsilon_r);
    sigma_eff(mi==-2) = mean(bound_sigma_eff);
    
    i = strfind(breastInfo,'s1=');
    j = strfind(breastInfo,'s2=');
    k = strfind(breastInfo,'s3=');
    l = strfind(breastInfo,'class');

    I = str2double(breastInfo(i+3:j-1));
    J = str2double(breastInfo(j+3:k-1));
    K = str2double(breastInfo(k+3:l-1));
    
    epsilon_r = reshape(epsilon_r,[I,J,K]);
    sigma_eff = reshape(sigma_eff,[I,J,K]);
    
    epsilon_r = squeeze(epsilon_r(round(I/2),:,:));
    sigma_eff = squeeze(sigma_eff(round(I/2),:,:));
    
    mi = find(epsilon_r(:)==epsilon_rb & sigma_eff(:)==sigma_b);
    
    dx = dy;
    dy = dz;
    I = J;
    J = K;
    
    Lx = I*dx;
    Ly = J*dy;
    
    if ~isempty(new_res)
        if mod(new_res(1),1)==0 && mod(new_res(2),1)==0
            newi = new_res(1);
            newj = new_res(2);
            newdx = Lx/newi;
            newdy = Ly/newj;
        else
            newdx = new_res(1);
            newdy = new_res(2);
            newi = floor(Lx/newdx);
            newj = floor(Ly/newdy);
        end
        
        [y,x] = meshgrid((1:J)*dy,(1:I)*dx);
        [newy,newx] = meshgrid((1:newj)*newdy,(1:newi)*newdx);
        
        epsilon_r = interp2(y,x,epsilon_r,newy,newx);
        sigma_eff = interp2(y,x,sigma_eff,newy,newx);
        mi = find(epsilon_r(:)==epsilon_rb & sigma_eff(:)==sigma_b);
        dx = newdx;
        dy = newdy;
        
    end
    
    epsilon_r(isnan(epsilon_r)) = epsilon_rb;
    sigma_eff(isnan(sigma_eff)) = sigma_b;
