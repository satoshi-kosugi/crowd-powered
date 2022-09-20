function output = enhanceAll()
  addpath(genpath('./'));
  dir_name = '../images/'
  listing = dir(strcat(dir_name, '*'));
  if not(exist('illumination'))
      mkdir 'illumination'
  end
  if not(exist('results'))
      mkdir 'results'
  end
  for n = 1:length(listing)
      disp(listing(n).name);
      if listing(n).name(1) ~= '._'
        % clc;close all;clear all;addpath(genpath('./'));
        %%

        filename = strcat(dir_name, listing(n).name);
        L = imresize(im2double(imread(filename)),1);

        %--------------------------------------------------------------
        post = true; % Denoising?
        % post = false; % Denoising?

        para.lambda = .15; % Trade-off coefficient
        % Although this parameter can perform well in a relatively large range,
        % it should be tuned for different solvers and weighting strategies due to
        % their difference in value scale.

        % Typically, lambda for exact solver < for sped-up solver
        % and using Strategy III < II < I
        % ---> lambda = 0.15 is fine for SPED-UP SOLVER + STRATEGY III
        % ......


        para.sigma = 2; % Sigma for Strategy III
        para.gamma = 0.7; %  Gamma Transformation on Illumination Map
        para.solver = 1; % 1: Sped-up Solver; 2: Exact Solver
        para.strategy = 3;% 1: Strategy I; 2: II; 3: III

        %---------------------------------------------------------------
        tic
        [I, T_ini,T_ref] = LIME(L,para);
        toc

        imwrite(T_ref, strcat('illumination/', listing(n).name));

        %% Post Processing
        if post
          YUV = rgb2ycbcr(I);
          Y = YUV(:,:,1);

          sigma_BM3D = 10;
          [~, Y_d] = BM3D(Y,Y,sigma_BM3D,'lc',0);

          I_d = ycbcr2rgb(cat(3,Y_d,YUV(:,:,2:3)));
          I_f = (I).*repmat(T_ref,[1,1,3])+I_d.*repmat(1-T_ref,[1,1,3]);

          imwrite(I_f, strcat('results/', listing(n).name));
        else
          imwrite(I, strcat('results/', listing(n).name));
        end

      end
  end
  output = 0
end
