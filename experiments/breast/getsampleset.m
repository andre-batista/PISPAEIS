clear

filepath = './class1/phantom1';
namefile = 'class_1_1';
frequency = 1e9;
epsilon_rb = 20; % 20;
sigma_b = 0;
new_res = [150,150];
slices = 5;

[aux_epsilon_r,aux_sigma,dx,dy,aux_mi] = loadphantom2Dset ...
    (filepath,frequency,epsilon_rb,sigma_b,new_res,slices);

for i = 1:slices
    epsilon_r = squeeze(aux_epsilon_r(i,:,:));
    sigma = squeeze(aux_sigma(i,:,:));
    mi = aux_mi{i};
    plotsample(epsilon_r,sigma,dx,dy)
    pause(1)
    save([namefile,'_',num2str(i),'.mat'],'dx','dy','epsilon_r','sigma','epsilon_rb','sigma_b','mi');
end