function [net,mat]=my_train( net, mat , x , y, opts )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    m = size(x, 3);
    numbatches = m / opts.batchsize;
     for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        tic;
        kk = randperm(m);
        res1=0;
        res0=0;
        vres1=0;
        vres0=0;
        
        m_net=net;
        m_mat=mat;
        %%±‰“Ï
        for ii=2:numel(mat.layers)
              for jj=1:mat.layers(ii)
                  if( rand()<opts.matrate )
                  m_mat.unit{ii}{jj}.x1=unidrnd( m_mat.layers(ii-1) );
                  end
                  
                   if( rand()<opts.matrate )
                  m_mat.unit{ii}{jj}.x2=unidrnd( m_mat.layers(ii-1) );
                    end
                  
                   if( rand()<opts.matrate )
                  m_mat.unit{ii}{jj}.op=unidrnd( 8);
                    end
              end
        end
        
       inputmaps = 1;
  

        for l = 1 : numel(net.layers)   %  layer
            if strcmp(net.layers{l}.type, 'c')
                for jj = 1 : net.layers{l}.outputmaps  %  output map
                    for ii = 1 : inputmaps  %  input map
                        net.layers{l}.k{ii}{jj} = net.layers{l}.k{ii}{jj}+(randn( size(net.layers{l}.k{ii}{jj}) )>opts.cnnrate).*( rand( size( net.layers{l}.k{ii}{jj} ) ) );
                        net.layers{l}.k{ii}{jj} =  net.layers{l}.k{ii}{jj}.*( net.layers{l}.k{ii}{jj}<10 );
                        net.layers{l}.k{ii}{jj} =  net.layers{l}.k{ii}{jj}.*( net.layers{l}.k{ii}{jj}>-10 );
                    end
                    net.layers{l}.b{jj} =  net.layers{l}.b{jj} + ( randn( size( net.layers{l}.b{jj}) )>opts.cnnrate ).*( rand( size( net.layers{l}.b{jj}) ) );
                    net.layers{l}.b{jj} =  net.layers{l}.b{jj} .* ( net.layers{l}.b{jj} < 10 );
                    net.layers{l}.b{jj} =  net.layers{l}.b{jj} .* ( net.layers{l}.b{jj} > -10);
                end
                inputmaps = net.layers{l}.outputmaps;
            end
        end
        %%
        for l = 1 : numbatches
            disp(['batch ',num2str(l)]);
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y=batch_y';

            net = cnnff(net, batch_x);       
            Ny=zeros(opts.batchsize,1);
            Nx=net.fv';
            for p=1:opts.batchsize
                Ny(p)=matff(  mat , Nx(p,:) );
                tmp=find(  batch_y(p,:) == max( batch_y(p,:) ) );
                tmp=tmp>5;
                if( tmp==Ny(p) )
                     if( tmp==1 ) res1=res1+1;
                     else
                         res0=res0+1;
                     end
                end
            end
            
            m_net = cnnff(m_net, batch_x);       
            Ny=zeros(opts.batchsize,1);
            Nx=m_net.fv';
            for p=1:opts.batchsize
                Ny(p)=matff(  m_mat , Nx(p,:) );
                tmp=find( batch_y(p,:)==max( batch_y(p,:) ) );
                tmp=tmp>5;
                if( tmp==Ny(p) )
                    if( tmp==1 ) vres1=vres1+1;
                    else
                        vres0=vres0+1;
                    end
                end
            end   
        end
        if(res0*res1>vres1*vres0)
            net=m_net;
            mat=m_met;
        end
        disp([num2str(max(res0*res1,vres1*vres0))]);
        toc;
    end

end

