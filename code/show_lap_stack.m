close all;
clear all;
test_in = imread('3_short.png');
[m,n,d] = size(test_in);
test_gray = rgb2gray(im2double(test_in));
mask = ones(m,n);
n_level = 6;
pyr = laplacian_pyramid(test_gray, n_level, false, mask);
figure;
i = 1;
j= 1;
for counter = 1:6
    subplot(2,3,counter);
    imshow(pyr{counter},[]);
    title(['level = ', num2str(counter)]);
end
%pyr_in = laplacian_pyramid(im_in(:,:,ch), nLevel, ...
                       % false, bin_alpha(mask_in(:,:,1)));