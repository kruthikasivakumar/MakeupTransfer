function alpha_out = learning_matting(fn_im, fn_mask)

%% parameters to change according to your requests
%fn_im = 'input_cr.png';
%fn_mask = 'input_cr_trimap.png';
%fn_im='data/input_lowres/plasticbag.png';
%fn_mask='data/trimap_lowres/Trimap1/plasticbag.png';

%% configuration
addpath(genpath('./code'));

%% read image and mask
imdata=imread(fn_im);
mask=getMask_onlineEvaluation(fn_mask);

%% compute alpha matte
[alpha]=learningBasedMatting(imdata,mask);

[m,n] = size(alpha);
alpha_out = zeros(m,n,3);
alpha_out(:,:,1) = 255*alpha;
alpha_out(:,:,2) = 255*alpha;
alpha_out(:,:,3) = 255*alpha;
%% show and save results
%imshow(uint8(alpha_out));
%figure,subplot(2,1,1); imshow(imdata);
%subplot(2,1,2),imshow(uint8(alpha*255));

% imwrite(uint8(alpha*255),fn_save);