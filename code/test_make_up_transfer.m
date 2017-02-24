%function out=test_make_up_transfer(style_in, im_in_name, style_ex, im_ex_name, opt) 

clear all;
    close all;
  % Ex: style_transfer flickr2 2187287549_74951db8c2_o martin 0;
  addpath('../libs/image_pyramids/');

  if ~exist('opt')
    opt.write_output=true;
    opt.transfer_eye=true;
    opt.recomp=true; %Re-comppute the matching
    opt.show_match=true; %Visualize the warping
    opt.verbose=true;
  end
    
  % Reading data
  command_str1 = ['sudo python extract_crop_test.py input.png'];
  command_str2 = ['sudo python extract_crop_test.py example.png'];
  [status1, c_out1] = system(command_str1);
  [status2, c_out2] = system(command_str2);
  if status1 ~= 0 || status2 ~= 0
      display ('exitting! some error in the python')
      display(c_out1);
      display(c_out2);
  end
  
   in_image = im2double(imread('input_cr.png'));
   ex_image = im2double(imread('example_cr.png'));
  
  if(numel(in_image) > numel(ex_image))
      [to_m, to_n, d] = size(ex_image);
      in_image = imresize(in_image,[to_m, to_n]);
  else if(numel(in_image) < numel(ex_image))
          [to_m, to_n, d] = size(in_image);
          ex_image = imresize(ex_image, [to_m, to_n]);
      end
  end
  imwrite(in_image, 'input_cr.png', 'PNG');
  imwrite(ex_image, 'example_cr.png', 'PNG');
  
  command_str1 = 'sudo python test_form_face_masks.py input_cr.png';
  command_str2 = 'sudo python test_form_face_masks.py example_cr.png';
  [status1, c_out1] = system(command_str1);
  [status2, c_out2] = system(command_str2);
  if status1 ~= 0 || status2 ~= 0
      display ('exitting! some error in the python')
      display(c_out1);
      display(c_out2);
  end
  
  command_str1 = 'sudo python extract_trimap_test.py input_cr.png';
  command_str2 = 'sudo python extract_trimap_test.py example_cr.png';
  [status1, c_out1] = system(command_str1);
  [status2, c_out2] = system(command_str2);
  if status1 ~= 0 || status2 ~= 0
      display ('exitting! some error in the python')
      display(c_out1);
      display(c_out2);
  end
  im_in = in_image;
  im_ex = ex_image;
  %in_trimap = im2double(imread('input_cr_trimap.png'));
  %ex_trimap = im2double(imread('example_cr_trimap.png'));
  %in_trimap = in_trimap(:,:,1);
  %ex_trimap = ex_trimap(:,:,1);
  
  in_face = imread('input_cr_face_mask.png');
  ex_face = imread('example_cr_face_mask.png');
  
  in_face_k = zeros(size(in_face));
  ex_face_k = zeros(size(ex_face));
  yosh1 = find(in_face == 127);
  yosh2 = find(in_face == 255);
  yosh3 = find(in_face == 0);
  in_face_k(yosh1) = 0.7;
  in_face_k(yosh2) = 0;
  in_face_k(yosh3) = 1;
  
  yosh1 = find(ex_face == 127);
  yosh2 = find(ex_face == 255);
  yosh3 = find(ex_face == 0);
  ex_face_k(yosh1) = 0.7;
  ex_face_k(yosh2) = 0;
  ex_face_k(yosh3) = 1;
  
  beta_in = zeros(size(in_face));
  beat_ex = zeros(size(ex_face));
  
  [m,n] = size(in_face);
  sigma = min(m,n)/25;
  
  diff_sq_3 = [2 1 2; 1 0 1; 2 1 2]/(2*sigma);
  diff_sq_mat_3 = exp(-1.*diff_sq_3);
  diff_sq_5 = [8 5 4 5 8; 5 2 1 2 5; 4 1 0 1 4; 5 2 1 2 5; 8 5 4 5 8]/(2*sigma);
  diff_sq_mat_5 = exp(-1.*diff_sq_5);
  win_size = 2;
  for i = win_size+1:m - win_size
      for j = win_size+1:n - win_size
          k_mat_in = in_face_k(i-win_size:i+win_size, j-win_size:j+win_size);
          beta_in(i,j) = min(min(1 - k_mat_in.*diff_sq_mat_5));
          k_mat_ex = ex_face_k(i-win_size:i+win_size, j-win_size:j+win_size);
          beta_ex(i,j) = min(min(1 - k_mat_ex.*diff_sq_mat_5));
      end
  end
  
  %figure; imshow(beta_in,[]);
  %figure; imshow(beta_ex,[]);
  %pause
  %close all;
  
  mask_in = double(test_get_a_matt('input_cr', 'yosh', 'png'));
  mask_ex = double(test_get_a_matt('example_cr', 'yosh', 'png'));
  %mask_in = demo_test('input');
  %mask_ex = demo_test('example');
  %mask_in = knn_matting_user_input_image('input_cr.png', 'input_cr_trimap.png');
  %mask_ex = knn_matting_user_input_image('example_cr.png', 'example_cr_trimap.png');
  %mask_in = knn_matting(im_in, in_trimap, 100, 1);
  %mask_ex = knn_matting(im_ex, ex_trimap, 100, 1);
  %imwrite(uint8(mask_in), 'input_cr_mask.png', 'PNG');
  %imwrite(uint8(mask_ex), 'example_cr_mask.png', 'PNG');
  %mask_in = mask_in./255;
  %mask_ex = mask_ex./255;
  
  %im_in = im2double(imread('input.png'));
  %im_ex = im2double(imread('input.png'));
  %im_in = im2double(imread('input.png'));
  %[to_m, to_n, to_ch] = size(im_in);
  %im_ex = im2double(imread('example.png'));
  %im_ex = imresize(im_ex, [to_m, to_n]);
  %imwrite(im_ex, 'example.png');
  %im_in = im2double(imread(sprintf('../../data/%s/fgs/%s.png', style_in, im_in_name)));
  %im_ex = im2double(imread(sprintf('../../data/%s/imgs/%s.png', style_ex, im_ex_name)));
  %im_in=imresize(im2double(imread(sprintf('../../data/%s/imgs/%s.png', style_in, im_in_name))), 0.5);
  %im_ex=imresize(im2double(imread(sprintf('../../data/%s/imgs/%s.png', style_ex, im_ex_name))), 0.5);
  
  %im_ex = im2double(imread('karthi.jpg'));
  %[m_test, n_test, d_test] = size(im_in);
  %im_ex = imresize(im_ex, [m_test, n_test]);
  
  %mask_in = test_get_a_matt(im_in, im_in_name, 'png');
  %mask_ex = test_get_a_matt(im_ex, im_ex_name, 'png');
  %mask_in=im2double(imread(sprintf('../../data/%s/masks/%s.png', style_in, im_in_name)));
  %mask_ex=im2double(imread(sprintf('../../data/%s/masks/%s.png', style_ex, im_ex_name)));
  %mask_in = 255.*ones(to_m, to_n, to_ch);
  %mask_ex = 255.*ones(to_m, to_n, to_ch);
  %mask_in = im2double(imread('input_cr_trimap.png'))

  %bg_ex = im2double(imread(sprintf('../../data/%s/bgs/%s.jpg', style_ex, im_ex_name)));
  bg_ex = zeros(size(im_in));

  %%%%--- Dense matching ----%%%%
  disp('--- Dense matching ---');
  %print_v('Computing the correspondence ...\n', opt.verbose);
  if opt.recomp
    %[vxm vym] = morph(style_ex, im_ex_name, style_in, im_in_name);
    [vxm vym] = morph('','example_cr.png', '', 'input_cr.png');
    im_ex_w =warpImage(im_ex, vxm, vym);
    [vx vy]=sift_flow(im_in, im_ex_w);
    [vxf vyf]=thresh_v(vx+vxm, vy+vym);
    save('match.mat', 'vxf', 'vyf');
  else
    load('match.mat');
  end

  close all;
  if opt.show_match
    im_ex_wf =warpImage(im_ex, vxf, vyf);
    figure;imshow(0.5*(im_in+im_ex_wf));drawnow;
    pause
  end

  %%%% --- Local Matching ----%%%%
  %print_v('Local transfer ...\n', opt.verbose);
  
%   if strcmp(style_ex, 'martin')
%     nch=3;
%   else
%     nch=1;
%   end
%   nch = 3;
%   e_0 = 1e-4;
%   gain_max = 2.8;
%   gain_min = 0.9;
%   hist_transfer=true;

  % Replace the input background with example.
  im_in = mask_in.*im_in + (1-mask_in).*bg_ex;
  im_ex = mask_ex.*im_ex;
  im_ex = warpImage(im_ex, vxf, vyf);
  beta_ex_warp = warpImage(beta_ex, vxf, vyf);

  %im_in = RGB2Lab(im_in);
  %im_ex = RGB2Lab(im_ex);
  im_in = rgb2lab(im_in);
  im_ex = rgb2lab(im_ex);
  
  in_Light = im_in(:,:,1); in_col_a = im_in(:,:,2); in_col_b = im_in(:,:,3);
  ex_Light = im_ex(:,:,1); ex_col_a = im_ex(:,:,2); ex_col_b = im_ex(:,:,3);
  
  sigmar = 40;
  eps = 1e-3;
  sigmas = 3;
  [in_struct, in_Ng] = GPA(in_Light, sigmar, sigmas, eps, 'Gauss');
  [ex_struct, ex_Ng] = GPA(ex_Light, sigmar, sigmas, eps, 'Gauss');
  
  figure, imshow(in_struct, []);
  figure, imshow(ex_struct, []);
  pause
  
  in_detail = in_Light - in_struct;
  ex_detail = ex_Light - ex_struct;
  
  close all;
  figure, imshow(in_detail, []);
  figure, imshow(ex_detail, []);
  pause
  close all;
  
  in_delta = 1.0;
  ex_delta = 0.0;
  re_detail = in_delta.*in_detail + ex_delta.*ex_detail;
  
  gamma = 0.8;
  re_col_a = (1.0 - gamma).*in_col_a + gamma.*ex_col_a;
  re_col_b = (1.0 - gamma).*in_col_a + gamma.*ex_col_a;
  
  
  %[in_struct_dx, in_struct_dy] = imgradientxy(in_struct, 'sobel');
  %[ex_struct_dx, ex_struct_dy] = imgradientxy(ex_struct, 'sobel');
  
  [in_struct_gm, bis_] = imgradient(in_struct);
  [ex_struct_gm, bis_] = imgradient(ex_struct);
  %re_struct = ex_struct;
  
  nch = 1;
  e_0 = 1e-4;
  gain_max = 2.8;
  gain_min = 0.9;
  hist_transfer=true;
  out = zeros(size(in_struct));
  for ch = 1 : nch
    nLevel = 6;
    % Disabled mask-based Laplacian for now.
    pyr_in = laplacian_pyramid(in_struct, nLevel, ...
                        false, bin_alpha(mask_in(:,:,1)));
    pyr_ex = laplacian_pyramid(ex_struct, nLevel, ...
        false, bin_alpha(mask_in(:,:,1))|bin_alpha(mask_ex(:,:,1)));

    pyr_out = pyr_in;

    for i = 1 : nLevel-1
      r = 2*2^(i+1);

      l_in = pyr_in{i};
      l_ex = pyr_ex{i};
      %l_ex = warpImage(l_ex, vxf, vyf);
      e_in = imfilter(l_in.^2, fspecial('gaussian', ceil(6*[r r]), r));
      e_ex = imfilter(l_ex.^2, fspecial('gaussian', ceil(6*[r r]), r));
      gain = (e_ex./(e_in+e_0)).^0.5;

      % Clamping gain maps
      gain(gain>gain_max)=gain_max;
      gain(gain<gain_min)=gain_min;
      l_new = l_in.*gain;

      if hist_transfer
        minus = l_in <0;
        l_new = HistTransferOneD(abs(l_new), abs(l_ex));
        l_new(minus) = -1*l_new(minus);
      end
      pyr_out{i} = l_new; 
    end

    %pyr_out{end} = warpImage(pyr_ex{end}, vxf, vyf);
    pyr_out{end} = pyr_ex{end};
    if ch==1 && hist_transfer
      pyr_out{end} = HistTransferOneD(pyr_out{end}, pyr_ex{end});
    end
    out = sum_pyramid(pyr_out);
  end
  %out = Lab2RGB(out);
  %im_in = Lab2RGB(im_in);

  
  %res_struct_dx = in_struct_dx;
  %res_struct_dy = in_struct_dy;
  
  %yosh = find(beta_ex_warp.*ex_struct_gm > in_struct_gm);
  %res_struct_dx(yosh) = ex_struct_dx(yosh);
  %res_struct_dy(yosh) = ex_struct_dy(yosh);
  
  %[m_1,n_1] = size(in_struct_gm);
  %re_struct(yosh) = ex_struct(yosh);
  %re_struct = reshape(re_struct, m_1, n_1);
  
  %res_struct = poisson_solver_function(res_struct_dx, res_struct_dy, in_struct);
  %res_light = res_struct + re_detail;
  re_struct = out;
  res_light = re_struct + re_detail;
  [y1, y2] = size(res_light);
  res_lab = zeros(y1, y2, 3);
  res_lab(:,:,1) = res_light;
  res_lab(:,:,2) = re_col_a;
  res_lab(:,:,3) = re_col_b;
  res_test = lab2rgb(res_lab);
  imshow(res_test,[]);
  
  
  
% function output=bin_alpha(input) 
% output=input;
% % In case the mask is not perfec / too small
% output(output<0.5)=0;
% output(output>=0.5)=1;
% se = strel('disk', 71);  
% output = imdilate(output,se);
% 
% % Add a small number to avoid crazy
% eps=1e-2;
% output = output + eps;
% 
% function print_v(msg, verbose)
% if verbose
%   fprintf(msg);
% end
