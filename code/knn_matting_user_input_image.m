%clear all;
function alpha = knn_matting_user_input_image(im_name, trimap_name)
run ('../vlfeat-0.9.20/toolbox/vl_setup');
lambda=100;
level=1;
l=10;
%im=double(imread('8.png'))/255;
im = double(imread(im_name))/255;
[m n d]=size(im);
nn=[10;2];
[a b]=ind2sub([m n],1:m*n);
feature=[reshape(im,m*n,d)';[a;b]/sqrt(m*m+n*n)*level+rand(2,m*n)*1e-6];
now=0;
for i=1:size(nn,1)
    ind=vl_kdtreequery(vl_kdtreebuild(feature),feature,feature,'NUMNEIGHBORS',nn(i),'MAXNUMCOMPARISONS',nn(i)*2);
    a=reshape(repmat(uint32(1:m*n),nn(i),1),[],1);
    b=reshape(ind,[],1);
    row(now+1:now+m*n*nn(i),:)=[min(a,b) max(a,b)];
    feature(d+1:d+2,:)=feature(d+1:d+2,:)/100;
    now=now+m*n*nn(i);
end
value=max(1-sum(abs(feature(1:d+2,row(:,1))-feature(1:d+2,row(:,2))))/(d+2),0);%1/(norm(x1-x2,1)+0.1);
A=sparse(double(row(:,1)),double(row(:,2)),value,m*n,m*n);
A=A+A';
D=spdiags(sum(A,2),0,n*m,n*m);

%%%% user input starts %%%%
avg=zeros(m,n,3);
for i=1:3
    avg(:,:,i)=mean(im(:,:,i:3:d-3+i),3);
end
% trimap = imread(trimap_name);
% trimap = trimap(:,:,1);
% bg_cord = [];
% fg_cord = [];
% aa = 1;
% bb = 1;
% [size_a, size_b] = size(trimap);
% for alpha = 1:size_a
%     for beta = 1:size_b
%         if (trimap(alpha,beta)==0)
%             bg_cord(aa) = alpha;
%             bg_cord(aa+1) = beta;
%             aa = aa + 2;
%         elseif (trimap(alpha,beta)==255)
%             fg_cord(bb) = alpha;
%             fg_cord(bb+1) = beta;
%             bb = bb + 2;
%         end
%     end
% end
% aa = aa - 1;
% bb = bb - 1;
% bg_cord = reshape(bg_cord, [aa/2,2]);
% fg_cord = reshape(fg_cord, [bb/2,2]);
% bg_cord1 = randperm(aa/2);
% fg_cord1 = randperm(bb/2);
% x_max = size(im,1) - l;
% y_max = size(im,2) - l;
% no_of_bgpts = 0;
% for something = 1:35
%     tempvar = bg_cord(bg_cord1(something),:);
%     if ((tempvar(1) > l && tempvar(1) < x_max) && (tempvar(2) > l && tempvar(2) < y_max))
%         no_of_bgpts = no_of_bgpts + 1;
%         bg_cord2(no_of_bgpts,:) = tempvar;
%     end
% end
% no_of_fgpts = 0;
% for something = 1:50
%     tempvar = fg_cord(fg_cord1(something),:);
%     if ((((tempvar(1)) > l) && ((tempvar(1)) < x_max)) && (((tempvar(2)) > l) && ((tempvar(2)) < y_max)))
%         no_of_fgpts = no_of_fgpts + 1;
%         fg_cord2(no_of_fgpts,:) = tempvar;
%     end
% end
% 
% x = [fg_cord2(:,1); 0; bg_cord2(:,1)];
% y = [fg_cord2(:,2); 0; bg_cord2(:,2)];
 figure('name','Left click on each layer and press Enter to terminate(Press Space to seperate layers)'),imagesc(avg);
 [x,y,BUTTON]=ginput;
%BUTTON = [ones(no_of_fgpts,1); 0; ones(no_of_bgpts,1)];
imwrite(min(max(avg,0),1),'avg.png','png');
a=round(x);
b=round(y);
map=zeros(m*n,1);
num=size(a,1);
for i=1:num
    if BUTTON(i)==1
        [x y]=meshgrid(a(i)-l:a(i)+l,b(i)-l:b(i)+l);
        for j=0:2
            avg(sub2ind([m n],a(i)-l:a(i)+l,b(i)-l:b(i)+l)+j*m*n)=(j==0);
            avg(sub2ind([m n],a(i)-l:a(i)+l,b(i)+l:-1:b(i)-l)+j*m*n)=(j==0);
        end
        map(sub2ind([m n],x,y))=1;
    end
end
%imagesc(avg);
imwrite(avg,'input_temp.png','png');
%imagesc(reshape(255.*map, [m,n]));
%%%%% map has all the points defined by the user %%%%%

tot=0;
alpha=zeros(n*m,num);
M=D-A+lambda*spdiags(map(:),0,m*n,m*n);
L=ichol(M);
i=1;
while i<=num
    %figure('name','alpha matte');
    val=zeros(m*n,1);
    while i<=num&&BUTTON(i)==1
        %[x y]=meshgrid(a(i),b(i));
        [x y]=meshgrid(a(i)-l:a(i)+l,b(i)-l:b(i)+l);
        val(sub2ind([m n],x,y))=1;
        i=i+1;
    end
    i=i+1;
    tot=tot+1;
    alpha(:,tot)=pcg(M,lambda*val(:),1e-10,2000,L,L');
    %imagesc(min(max(reshape(alpha(:,tot),m,n),0),1));
    %imwrite(min(max(reshape(alpha(:,tot),m,n),0),1),sprintf('%02d.png',tot),'png');
    %alpha = 255*min(max(reshape(alpha(:, tot), m, n), 0), 1);
    %imwrite(alpha, sprintf('%s_mask.png', im_name), 'png');
end

alpha = 255*min(max(reshape(alpha(:, tot), m, n), 0), 1);
imwrite(alpha, sprintf('%s_mask.png', im_name), 'png');