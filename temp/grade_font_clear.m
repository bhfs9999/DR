clear;
clc;
path=pwd;
image=[];
image=[image;dir([path '\' '*.jpg'])];
for k=1:length(image)
    I=imread(image(k).name);
    [com]=font_clear(I);
    imwrite(com,image(k).name);
end