clear;
disp(['reading data']);
load data_batch_1;
r=double(data(:,1:1024));
b=double(data(:,1025:1025+1023));
g=double(data(:,2049:end));

train_x= zeros(10000,1024);
train_y = zeros(10,10000);

for i=1:10000
    for j=1:1024
        train_x(i,j)=( r(i,j)*30+g(i,j)*59+b(i,j)*11+50 )/100;
    end
    train_y(labels(i)+1,i)=1;
end

train_x = double(reshape(train_x',32,32,10000))/255;
x = align_data( train_x(:,:,1:1000) );

cae = cae_setup(1,32,5,2,0);

opts.batchsize = 500;
opts.shuffle = 1;
opts.alpha = 0.03;
opts.numepochs=1;

cae = cae_train( cae,x,opts );



%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error
rand('state',0)
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 32, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 64, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};
cnn = cae_setup_cnn(cae,cnn,x,train_y);

mat.layers=[size(cnn.ffW,2) 512 256 120 60 30 16 8 4 2];

mat=matsetup(mat);

opts.numepochs = 2;
opts.matrate=0.03;
opts.cnnrate=0.03;

[cnn,mat]=my_train( cnn, mat, train_x(:,:,1:1000) , train_y(:,1:1000) , opts);

