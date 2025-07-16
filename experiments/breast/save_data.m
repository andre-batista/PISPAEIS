clc, clear

%% General parameters
f = 1e9;
nsamp = 100;
bd_epsr = [1,5];
bd_sig = [0,1];
img_size = [50,50];

%% Bounds
bd = [20,270]; filepath = './class1/phantom1';      % class1 - phantom1
% bd = [26,225]; filepath = './class1/phantom2';      % class1 - phantom2

%% Collecting data
[epsilon,sigma,mi] = phantom_read(filepath,f);
i = round(linspace(bd(1),bd(2),nsamp+2));
i([1,end]) = [];
epsilon = bd_epsr(1) + diff(bd_epsr)*epsilon(i,:,:)/max(epsilon(:));
sigma = bd_sig(1) + diff(bd_sig)*sigma(i,:,:)/max(sigma(:));
[~,J,K] = size(epsilon);
j = round(linspace(0,J+1,img_size(1)+2));
j([1,end]) = [];
k = round(linspace(0,K+1,img_size(2)+2));
k([1,end]) = [];
epsilon = epsilon(:,j,k);
sigma = sigma(:,j,k);
id = filepath(3:end);

%% Saving data
save([filepath,'/phantoms.mat'],'epsilon','sigma','bd_epsr','bd_sig','id')