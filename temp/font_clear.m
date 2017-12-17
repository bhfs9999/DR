function [com] =font_clear(I)
    IR=I(:,:,1);
    IG=I(:,:,2);
    IB=I(:,:,3);
    [m,n]=size(IR);
    mask=zeros(m,n);
    temp=zeros(m,n);
    r=floor((1/4)*m);
    c=floor((1/4)*n);
    r1=floor((1/13)*m);
    c1=floor((1/13)*n);
    mask(1:r,1:c)=1;
    mask(1:r1,(n-c1):n)=1;
    ind=find(IR>225);
    temp(ind)=1;
    mask=mask.*temp;
    se=strel('disk',4);
    mask=imdilate(mask,se);
%     mask=im2bw(mask,0.5);
%     imtool(mask);
    ind=find(mask);
    IR(ind)=0;
    IG(ind)=0;
    IB(ind)=0;
    com=cat(3,IR,IG,IB);
end

