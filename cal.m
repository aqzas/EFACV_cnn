function res = cal( x1,x2,op)
%CAL Summary of this function goes here
%   Detailed explanation goes her
switch(op)
    case 1
        res=(x1+x2)/2;
    case 2
        res=x1;
    case 3
        res=x2;
    case 4
        res=min(x1,x2);
    case 5
        res=max(x1,x2);
    case 6
        res=(x1+x2+1)/2;
    case 7
        res=min(x1+x2,255);
    case 8
        res=max(0,x1-x2);
end



end

