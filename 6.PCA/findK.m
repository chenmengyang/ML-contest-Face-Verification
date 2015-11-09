function [ K ] = findK( Sigma, retain)
%FINDK Summary of this function goes here
%   Detailed explanation goes here
Sn = sum(sum(Sigma));
rate = 0;
ind = 0;
while rate < retain
    ind = ind + 1;
    Sk = 0;
    for i=1:ind
       Sk = Sk + Sigma(i,i);
    end
    rate = Sk/Sn;
end
K = ind;
end

