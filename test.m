r=double(data(:,1:1024))
b=double(data(:,1025:1025+1023));
g=double(data(:,2049:end));

train_x=zeros(10000,1024);

for i=1:10000
    for j=1:1024
        train_x(i,j)=double( r(i,j)*30+g(i,j)*59+b(i,j)*11+50 )/100;
    end
end