function alpha_out = demo_test(fn_im)

%% parameters to change according to your requests
%fn_im = 'input_cr.png';
%fn_mask = 'input_cr_trimap.png';
%fn_im='data/input_lowres/plasticbag.png';
%fn_mask='data/trimap_lowres/Trimap1/plasticbag.png';
temp = fn_im;
fn_mask = sprintf('%s_cr_trimap.png', fn_im);
fn_im = sprintf('%s_cr.png', fn_im);
%% configuration
addpath(genpath('./code'));

%% read image and mask
imdata=imread(fn_im);
mask=getMask_onlineEvaluation(fn_mask);

%% compute alpha matte
[alpha]=learningBasedMatting(imdata,mask);

%% show and save results
%imshow(uint8(alpha*255));
%imwrite(uint8(alpha*255), sprintf('%s_cr_trimap.png', fn_im));
alpha = uint8(alpha.*255);
[m, n] = size(alpha);
alpha_out = zeros(m,n,3);
alpha_out(:,:,1) = alpha;
alpha_out(:,:,2) = alpha;
alpha_out(:,:,3) = alpha;
%imwrite(alpha_out, sprintf('%s_cr_mask.png', temp));
%figure,subplot(2,1,1); imshow(imdata);
%subplot(2,1,2),imshow(uint8(alpha*255));

% imwrite(uint8(alpha*255),fn_save);
