%% PHANTOM_READ - Obtain permittivity and conductivity information
% This function reads the permittivity and conductivity information as in
% [1].
% Inputs:
%   filepath:   String with path to the files (breastInfo,mtype,pval)
%   f:          Linear frequency [Hz]
% Outpus
%   eps:    Relative permittivity
%   sgma:   Conductivity [S/m]
%   mi:     Media indexes:
%       -1:  Immersion medium
%       -2:  Skin
%       -4:  Muscle
%       1.1: Fibroconnective/glandular-1
%       1.2: Fibroconnective/glandular-2
%       1.3: Fibroconnective/glandular-3
%       2:   Transitional
%       3.1: Fatty-1
%       3.2: Fatty-2
%       3.3: Fatty-3

function [eps,sgma,mi] = phantom_read(filepath,f)

    % Read main data
    bi = fileread([filepath '/breastInfo.txt']);
    be = strfind(bi, '=');
    s1 = str2double(bi(be(2)+1:be(2)+3));
    s2 = str2double(bi(be(3)+1:be(3)+3));
    s3 = str2double(bi(be(4)+1:be(4)+3));
    
    % Read media indexes
    mi = textread([filepath '/mtype.txt'],'%f');
    mi = reshape(mi,[s1 s2 s3]);

    % Read p_i data
    p_i = textread([filepath '/pval.txt'],'%f');
    p_i = reshape(p_i,[s1 s2 s3]);
    
    % Genreal parameters
    w = 2*pi*f;
    eps_0= 8.85e-12;
    
    if f>3e9 && f<=10e9 % ref paper : instruction manual
        % 3 adipose/fat , 1 fibroglandular
        %[min 3_low 3_med 3_high 1_low 1_med 1_high max skin mus];
        e_i = [2.293 2.848 3.116 3.987 12.99 13.81 14.2 23.2 15.93 21.66];
        e_d = [0.092 1.104 1.592 3.545 24.4 35.55 40.49 46.05 23.83 33.24];
        tau = 13e-12;
        s = [0.005 0.005 0.05 0.08 0.397 0.738 0.824 1.306 0.831 0.886];
    
    elseif f>0.5e9 && f<=3e9 % ref paper : jake medical physics 2010
        e_i = [2.28 2.74 3.11 4.09 16.8 17.5 18.6 29.1 15.3 ];
        e_d = [0.141 1.33 1.7 3.54 19.9 31.6 35.6 38.1 24.8 ];
        tau = 15e-12;
        s = [0.0023 0.0207 0.0367 0.0842 0.461 0.72 0.817 1.38 0.741];
    end
    
    ep = e_i+e_d/(1+w^2*tau^2);
    sig = e_d*w*tau/(1+w^2*tau^2)+s/(w*eps_0); % imaginary part is negative(-j).
    e_min = ep([1:3 7:-1:5 4]);
    e_max = ep([2:4 8:-1:6 5]);
    s_min = sig([1:3 7:-1:5 4 ]);
    s_max = sig([2:4 8:-1:6 5]);
    tissues = [3.1 3.2 3.3 1.1 1.2 1.3 2];
    
    epsmin = 22.9*ones(size(p_i));
    epsmax = 22.9*ones(size(p_i));
    sgmmin = zeros(size(p_i)) +.07;
    sgmmax = zeros(size(p_i)) +.07;
    
    for i=1:length(tissues)
        epsmin(mi==tissues(i))= e_min(i);
        epsmax(mi==tissues(i))= e_max(i);
        sgmmin(mi==tissues(i))= s_min(i);
        sgmmax(mi==tissues(i))= s_max(i);
    end
    
    eps = p_i.*epsmax+(1-p_i).*epsmin;
    sgma = p_i.*sgmmax+(1-p_i).*sgmmin;
    
end

% References
% [1]   M. J. Burfeindt, T. J. Colgan, R. O. Mays, J. D. Shea, N. Behdad, 
%       B. D. Van Veen, and S. C. Hagness, "MRI-derived 3D-printed breast 
%       phantom for microwave breast imaging validation," IEEE Antennas and
%       Wireless Propagation Letters, vol. 11, pp. 1610-1613, 2012.