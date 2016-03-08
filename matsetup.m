function mat= matsetup( mat )
%MATSETUP Summary of this function goes here
%   Detailed explanation goes here
for i=2:numel(mat.layers)
    for j=1:mat.layers(i)
        mat.unit{i}{j}.x1=unidrnd( mat.layers(i-1) );
        mat.unit{i}{j}.x2=unidrnd( mat.layers(i-1) );
        mat.unit{i}{j}.op=unidrnd( 8);
    end
end

end

