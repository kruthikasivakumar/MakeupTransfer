function output=bin_alpha(input) 
output=input;
% In case the mask is not perfec / too small
output(output<0.5)=0;
output(output>=0.5)=1;
se = strel('disk', 71);  
output = imdilate(output,se);

% Add a small number to avoid crazy
eps=1e-2;
output = output + eps;