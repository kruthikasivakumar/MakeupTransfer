function alpha = test_get_a_matt(img_name, img_name_without_type, type)

%     img_name_without_type = '3_short';
%     type = 'png';
    temp_name = img_name;
     img_name = strcat(img_name, strcat('.', type));
     %mod_img_name = strcat(img_name_without_type, strcat('_m.', type));

     mod_img_name = strcat(temp_name, strcat('_trimap.', type));

    img=im2double(imread(img_name));
    %get_pencil_curve(1,img, mod_img_name);
    %mod_img_raw=im2double(imread(mod_img_name));
    mod_img_raw = im2double(imread(mod_img_name));
    %mod_img = imresize(mod_img, [size(img, 1) size(img, 2)]);

    %figure, imshow(mod_img_raw);
    mod_img = refine_img(img, mod_img_raw);

    fg_and_bg=sum(abs(img-mod_img),3)>0.001;
    fg=rgb2gray(mod_img).*fg_and_bg;

    %figure, imshow([fg_and_bg, fg]);

    if(1)
        alpha_temp = generateAlpha(img,fg_and_bg,fg);
        beta_temp = 1-alpha_temp;
        alpha = zeros(size(img));
        alpha(:,:,1) = alpha_temp;
        alpha(:,:,2) = alpha_temp;
        alpha(:,:,3) = alpha_temp;
    %figure, imshow([alpha, beta]);
        imwrite(alpha, strcat(temp_name, strcat('_mask.'), 'png'));
        
    end

end

%if(1) 
    %foreground = img.*repmat(alpha,[1,1,3]);
    %background = img.*repmat(beta,[1,1,3]);
    %imwrite(img_name_without_type, strcat('_matt.', type));
    %figure, imshow([foreground,background]);
%end

