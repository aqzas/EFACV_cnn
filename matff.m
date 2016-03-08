function Y = matff( mat,x)
%MATFF Summary of this function goes here
%   Detailed explanation goes here

   for i=1:mat.layers(1)
       mat.unit{1}{i}.res=x(i);
   end

    for i=2:numel(mat.layers)
        for j=1:mat.layers(i)
            mat.unit{i}{j}.res = cal ( mat.unit{i-1}{mat.unit{i}{j}.x1}.res,   mat.unit{i-1}{mat.unit{i}{j}.x2}.res, mat.unit{i}{j}.op );
        end
    end
    
    Y=mat.unit{numel(mat.layers)}{1}.res>mat.unit{numel(mat.layers)}{2}.res;

end
