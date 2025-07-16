clear

filepath = './class3/phantom1';
frequency = 1e9;
epsilon_rb = 20;
sigma_b = 0;
new_res = [150,150];

[epsilon_r,sigma,dx,dy,mi] = loadphantom2D ...
    (filepath,frequency,epsilon_rb,sigma_b,new_res);

plotsample(epsilon_r,sigma,dx,dy)

save sample.mat dx dy epsilon_r sigma epsilon_rb sigma_b mi